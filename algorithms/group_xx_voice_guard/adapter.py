"""Voice Guard 课程平台 HTTP 适配服务。

符合 lym-lab/algorithm_research_frontend 的算法接入规范（docs/算法接入规范.md）：

    POST /process   multipart/form-data
        defend_audio   文件     必填，要保护的音频（wav/mp3/flac 等）
        adv_audio      文件     可选，扰动方向音频（缺省自动选 least-likely 说话人）
        method_id      字符串   方法 ID
        parameters     JSON 字符串 例如 {"eps":0.05,"fast_mode":true,"target_speaker":"p225"}

    返回 200 JSON：
        {"message": "...", "artifacts": [{"kind":"audio","label":"保护后音频",
          "filename":"defended.wav","url":"http://host/files/<job>/defended.wav"}, ...]}

其他端点：
    GET  /health    健康检查
    GET  /files/... 结果文件（CORS 已放开，供浏览器预览/下载）

启动（GPU）：
    python platform/service/adapter.py \
        --model-dir checkpoints/adain-vc-official \
        --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
        --vctk data/VCTK22 --device cuda
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[2]
# In the course Fork, launch from the pinned Voice Guard source checkout.
sys.path.insert(0, str(SOURCE_ROOT if (SOURCE_ROOT / "vguard").is_dir() else Path.cwd()))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi import FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from vguard.audio import load_audio  # noqa: E402
from vguard.evaluation import identity_verdict  # noqa: E402

ARGS = None
PIPE = None
SPEAKER_FIRST: dict[str, str] = {}
GPU_LOCK = asyncio.Lock()
LOG = logging.getLogger(__name__)
DEFAULT_METHOD_ID = "group_xx_voice_guard"
MAX_UPLOAD = 30 * 1024**2
AUDIO_FORMATS = {"WAV", "WAVEX", "RF64", "FLAC", "OGG", "MP3", "MPEG"}
AUDIO_RESULTS = {
    "original_trimmed.wav": "原始音频",
    "defended.wav": "保护后音频",
    "delta.wav": "扰动音频",
    "conv_from_clean.wav": "对照：干净音频经 VC 转换",
    "conv_from_defended.wav": "被保护音频经 VC 转换",
}


class InvalidAudio(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def save_audio(upload, path, field):
    if upload.size is not None and upload.size > MAX_UPLOAD:
        raise InvalidAudio(f"{field} 超过 30 MiB", 413)
    with path.open("wb") as dest:
        chunk = upload.file.read(MAX_UPLOAD + 1)
        if len(chunk) > MAX_UPLOAD:
            raise InvalidAudio(f"{field} 超过 30 MiB", 413)
        dest.write(chunk)

    try:
        with sf.SoundFile(path) as audio:
            if audio.format not in AUDIO_FORMATS:
                raise InvalidAudio(f"{field} 仅支持 WAV、MP3、FLAC 或 Ogg/Vorbis", 415)
            if not (8000 <= audio.samplerate <= 96000 and 1 <= audio.channels <= 2):
                raise InvalidAudio(f"{field} 必须是 8–96 kHz 的单声道或双声道音频")
            if not .25 <= audio.frames / audio.samplerate <= 8:
                raise InvalidAudio(f"{field} 时长必须在 0.25–8 秒之间")
            samples = audio.read(dtype="float32", always_2d=True)
            if len(samples) != audio.frames or not np.isfinite(samples).all():
                raise InvalidAudio(f"{field} 音频不完整或包含无效采样值")
            if np.max(np.abs(samples.mean(axis=1)), initial=0) < 1e-5:
                raise InvalidAudio(f"{field} 没有可处理的语音信号")
    except sf.LibsndfileError as exc:
        raise InvalidAudio(f"无法解码 {field}，请上传有效的音频文件", 415) from exc
    return str(path)


def _build_pipeline(args):
    from vguard.pipeline import Pipeline
    pipe = Pipeline(args.model_dir, args.registry, args.attack_registry,
                    ecapa_dir=args.ecapa_dir, device=args.device)
    if args.vctk and os.path.isdir(args.vctk):
        for s in pipe.speakers:
            d = os.path.join(args.vctk, s)
            if os.path.isdir(d):
                wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
                if wavs:
                    SPEAKER_FIRST[s] = os.path.join(d, wavs[0])
    return pipe


def _auto_speaker(pipe, wav_path: str) -> str:
    """用评估 ASV（ECAPA）判上传音频的说话人，作为被保护说话人。"""
    wav = load_audio(wav_path, pipe.pp["sample_rate"])
    res = pipe._ecapa_asv(wav)  # noqa: SLF001
    return res["pred"]


app = FastAPI(title="Voice Guard 算法服务", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok" if PIPE is not None else "not_ready", "service": "voice-guard-adapter"}


@app.exception_handler(RequestValidationError)
async def validation_error(_request, _exc):
    return JSONResponse(status_code=400, content={"error": "请以 multipart/form-data 提交音频文件和表单字段"})


@app.get("/files/{job_id}/{filename}", include_in_schema=False)
def result_file(job_id: str, filename: str):
    if ARGS and re.fullmatch(r"[0-9a-f]{32}", job_id) and filename in (*AUDIO_RESULTS, "report.json"):
        path = Path(ARGS.out_dir) / job_id / filename
        if path.is_file():
            return FileResponse(path, media_type="application/json" if filename.endswith(".json") else "audio/wav")
    return JSONResponse(status_code=404, content={"error": "结果文件不存在或已清理"})


def parse_parameters(params):
    eps = params.get("eps", .05)
    if type(eps) not in (int, float) or not .02 <= eps <= .3 or not math.isfinite(eps):
        raise ValueError("eps 必须是 0.02–0.3 之间的数值")
    fast = params.get("fast_mode", True)
    context = params.get("preserve_context", False)
    if not isinstance(fast, bool) or not isinstance(context, bool):
        raise ValueError("fast_mode/preserve_context 必须是布尔值")
    method = params.get("method", "vg")
    if method not in ("vg", "vg-projected"):
        raise ValueError("method 必须为 vg 或 vg-projected")
    target = params.get("target_speaker", "")
    if not isinstance(target, str) or len(target) > 16:
        raise ValueError("target_speaker 必须是至多 16 个字符的字符串")
    return float(eps), fast, context, method


@app.post("/process")
async def process(
    defend_audio: UploadFile | None = File(default=None),
    adv_audio: UploadFile | None = File(default=None),
    method_id: str | None = Form(default=None),
    parameters: str = Form(default="{}"),
):
    expected = ARGS.method_id if ARGS else DEFAULT_METHOD_ID
    if method_id != expected:
        return JSONResponse(status_code=400, content={"error": f"method_id 必须为 {expected}"})
    try:
        params = json.loads(parameters or "{}")
        if not isinstance(params, dict):
            raise ValueError("parameters 必须是 JSON 对象")
        eps, fast, context, method = parse_parameters(params)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"parameters 解析失败：{e}"})

    if defend_audio is None:
        return JSONResponse(status_code=400, content={"error": "缺少必填素材 defend_audio"})
    if PIPE is None:
        return JSONResponse(status_code=503, content={"error": "模型尚未就绪"})

    job_id = uuid.uuid4().hex
    job_dir = Path(ARGS.out_dir) / job_id
    it1, it2 = (300, 200) if fast else (ARGS.iters1, ARGS.iters2)

    def compute():
        with TemporaryDirectory(prefix="voice-guard-upload-") as temp:
            inputs = Path(temp)
            defend_path = save_audio(defend_audio, inputs / "defend.audio", "defend_audio")
            target = params.get("target_speaker", "").strip()
            if target and target not in PIPE.speakers:
                raise ValueError("target_speaker 不在注册库，请填写已注册编号或留空")
            if not target:
                target = _auto_speaker(PIPE, defend_path)

            if adv_audio is not None:
                adv_path = save_audio(adv_audio, inputs / "reference.audio", "adv_audio")
            else:
                y = PIPE.least_likely(target)
                if y not in SPEAKER_FIRST:
                    raise ValueError(f"缺少 least-likely 说话人 {y} 的音频，请上传 adv_audio")
                adv_path = SPEAKER_FIRST[y]

            rep = PIPE.protect(defend_path, target, adv_path, eps=eps,
                               iters1=it1, iters2=it2, out_dir=str(job_dir), progress=False,
                               refine_steps=1000 if method == "vg-projected" else 0,
                               preserve_context=context)
            if not all((job_dir / filename).is_file() for filename in (*AUDIO_RESULTS, "report.json")):
                raise RuntimeError("Incomplete Voice Guard outputs")
            rep["files"] = {key: Path(value).name for key, value in rep["files"].items()}
            if "error" in rep.get("asv", {}):
                rep["asv"]["error"] = "身份评估未能完成，请查看服务日志"
            (job_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, rep

    try:
        async with GPU_LOCK:
            target, rep = await run_in_threadpool(compute)
    except ValueError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JSONResponse(status_code=e.status if isinstance(e, InvalidAudio) else 400, content={"error": str(e)})
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        LOG.exception("Voice Guard job %s failed", job_id)
        return JSONResponse(status_code=500, content={"error": "处理失败，请重试或将任务时间告知服务维护者"})

    artifacts = [
        {"kind": "audio", "label": label, "filename": name, "url": f"./files/{job_id}/{name}"}
        for name, label in AUDIO_RESULTS.items()
    ]
    artifacts.append({"kind": "json", "label": "处理报告（参数/判定/相似度）",
                      "filename": "report.json", "url": f"./files/{job_id}/report.json"})

    asv = rep.get("asv", {})
    imp_pred = asv.get("defended", {}).get("pred")
    conv_pred = asv.get("conv_def", {}).get("pred")
    assessed = identity_verdict(asv, target)
    ok = assessed["screened_identity_ok"]
    verdict = "身份维通过，质量尚需独立评估" if ok else "未通过筛选后的身份维检查（详见 report.json）"

    return {
        "message": f"处理完成：说话人 {target}，ε={eps}，{verdict}",
        "artifacts": artifacts,
        "meta": {"job_id": job_id, "method_id": method_id, "target_speaker": target,
                 "eps": eps, "iters": [it1, it2], "elapsed_s": rep.get("elapsed_s"),
                 "defended_pred": imp_pred, "converted_pred": conv_pred, "defense_ok": ok,
                 "identity_verdict": assessed, "quality_evaluated": False,
                 "method": method, "preserve_context": context},
    }


def main():
    global ARGS, PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--attack-registry", required=True)
    ap.add_argument("--vctk", default="")
    ap.add_argument("--ecapa-dir", default="checkpoints/ecapa-local")
    ap.add_argument("--out-dir", default="outputs/platform_jobs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters1", type=int, default=3000)
    ap.add_argument("--iters2", type=int, default=1500)
    ap.add_argument("--method-id", default=DEFAULT_METHOD_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ARGS = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", ARGS.method_id) or min(ARGS.iters1, ARGS.iters2) < 1:
        ap.error("invalid method ID or iteration budget")

    os.makedirs(ARGS.out_dir, exist_ok=True)
    PIPE = _build_pipeline(ARGS)

    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)


if __name__ == "__main__":
    main()
