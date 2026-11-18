# A036 · Voice Guard 语音隐私主动防御

| 项目 | 内容 |
| --- | --- |
| 课程组号 | 尚未分配 |
| 选题 | A036 Voice Guard（音频主动防御） |
| 方法 ID | `group_xx_voice_guard`（联调用） |
| 平台 Fork | [Abyss-sword/algorithm_research_frontend · group_xx_submission](https://github.com/Abyss-sword/algorithm_research_frontend/tree/group_xx_submission) |

| 成员 | GitHub | 分工 |
| --- | --- | --- |
| 石锐杰 | [miruikediyikeai](https://github.com/miruikediyikeai) | 华为云部署，与张书瑞共同完成复现 |
| 张书瑞 | [Abyss-sword](https://github.com/Abyss-sword) | 阅读和理解原论文，与石锐杰共同完成复现 |

## 论文与实现

- Jingyang Li、Dengpan Ye、Long Tang、Chuanxi Chen、Shengshan Hu，[*Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain*](https://www.ijcai.org/proceedings/2023/535)，IJCAI 2023。
- 参考 [论文作者的部分代码](https://github.com/WHUB401/Voice-Guard) 和 [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion)。本组[算法源码固定版本 `49ec85e`](https://github.com/Abyss-sword/Voice-Guard/tree/49ec85e862f91506ad7f70f5e9acf3053bc67263) 包含时域扰动、可微 mel、心理声学掩蔽和两阶段优化；另实现谱投影继续优化。第三方代码与权重按各来源条款使用。
- 本分支的 [HTTP 适配器](algorithms/group_xx_voice_guard/adapter.py) 调用上述版本的算法，校验素材和参数，并返回可预览、下载的本次结果。[方法配置](public/methods/group_xx_voice_guard.json) 通过[目录索引](public/methods/index.json)加载。

## 素材、参数与输出

| 素材 | 必填 | 格式与用途 |
| --- | --- | --- |
| `defend_audio` | 是 | 待保护的单人语音 |
| `adv_audio` | 否 | 扰动方向与转换内容参考；留空使用注册库中最不相似的说话人音频 |

每段限 30 MiB、0.25–8 秒、8–96 kHz、单或双声道。支持 WAV、MP3、FLAC、Ogg/Vorbis，以服务端实际解码结果为准。

| 参数 | 类型与默认值 | 作用 |
| --- | --- | --- |
| `eps` | 数值 `0.05`；范围 0.02–0.30 | 波形扰动强度 |
| `fast_mode` | 布尔 `true` | 300/200 步；关闭默认 3000/1500 步 |
| `target_speaker` | 字符串，最多 16 个字符；默认空 | 已注册说话人编号；留空做 ECAPA 闭集匹配 |
| `method` | 枚举 `vg`、`vg-projected`；默认 `vg` | 原始 VG 或额外优化 1000 步的实验方法 |
| `preserve_context` | 布尔 `false` | 保留裁剪区间外的原音 |

输出 5 个 WAV（原音、保护后音频、扰动、干净转换、保护后转换）和 1 份 JSON 报告（实际参数、时长与身份评估）。在线评估要求干净转换先识别为目标人，才判断保护后转换能否改变识别结果。在线不计算 MOS/WER；自动匹配仅覆盖 20 人注册库。

## 安装与联调

算法环境建议 Python 3.10–3.12，PyTorch/torchaudio 2.5.1。历史 GPU 环境为 Ubuntu 22.04、Tesla T4；CPU 可联调，预计更慢。前端用 Node.js 20+。在平台 Fork 目录执行：

```bash
FRONTEND_ROOT="$PWD"
git clone https://github.com/Abyss-sword/Voice-Guard.git ../voice-guard-source
cd ../voice-guard-source
git switch --detach 49ec85e862f91506ad7f70f5e9acf3053bc67263
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r deploy/requirements.txt
python scripts/reproduce.py models
python scripts/reproduce.py data-subset
python scripts/reproduce.py registry --device cuda
python "$FRONTEND_ROOT/algorithms/group_xx_voice_guard/adapter.py" \
  --model-dir checkpoints/adain-vc-official --ecapa-dir checkpoints/ecapa-local \
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
  --vctk data/VCTK22 --device cuda
```

CPU 环境将 wheel 索引末尾的 `cu124` 改为 `cpu`，注册库和适配器的 `--device` 也改为 `cpu`。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，并把上面的 Bash 续行命令合并成一行。模型与数据的固定来源、下载校验和注册步骤见[资源准备](https://github.com/Abyss-sword/Voice-Guard/blob/49ec85e862f91506ad7f70f5e9acf3053bc67263/README.md#环境与资源)。

1. 请求 `http://127.0.0.1:8020/health`，看到 `status: ok`。云端服务保持回环监听；验收电脑用 SSH 本地转发 `-L 8020:127.0.0.1:8020`，或由维护者提供可访问的 HTTPS 服务。
2. 在平台 Fork 根目录执行 `npm install`、`npm test`、`npm run build`、`npm run dev`。打开终端显示的地址，在方法集合中进入 **Voice Guard 语音隐私主动防御**。
3. 点击「用样例试一试」，参考音频留空。首次联调选择「原始 VG」、开启「快速模式」，然后点击「开始处理」，试听 WAV 并下载报告。也可以移除样例后上传自己的音频。
4. 服务把成功结果写到算法目录的 `outputs/platform_jobs/<任务ID>/`。原始上传用临时目录处理并在结束时清理；浏览器取消只停止等待。结果不会自动过期，维护者应在任务结束后定期清理不再需要的任务目录。

接口为同步 `POST /process`，发送表单文件、`method_id` 和 JSON 字符串 `parameters`；失败返回非 2xx 与 `error`。结果相对 URL 按接口地址解析。默认配置面向验收电脑本机和 SSH 隧道；从 HTTPS 页面远程验收时，使用 HTTPS 后端并更新配置的 `adapter.endpoint`。反向代理须允许 2 个 30 MiB 文件及表单开销，超时须覆盖同步处理时长。

## 真实样例与限制

[原音 p225_040](public/samples/group_xx_voice_guard/input.wav)、[保护后 PCM16](public/samples/group_xx_voice_guard/output_defended.wav) 和[保护后转换](public/samples/group_xx_voice_guard/output_converted.wav) 来自同一次云端离线实验。样例参数为 `eps=0.05`、完整 3000/1500 步、`target_speaker=p225`、`vg-projected`、`preserve_context=false`，参考内容为 `p226_002`。该条预测 MOSNet 3.275、Whisper WER 0，保护后闭集身份为 p225、转换后为 p246。在线若自动选到其他参考，结果可不同。

样例来自 [CSTR VCTK，CC BY 4.0](https://doi.org/10.7488/ds/2645)，使用 22.05 kHz 数据重采样、裁剪，并生成带扰动和转换的衍生音频。[来源、参数与文件摘要](public/samples/group_xx_voice_guard/sample-provenance.json) · [离线统计与逐句证据](https://github.com/Abyss-sword/Voice-Guard/tree/49ec85e862f91506ad7f70f5e9acf3053bc67263/docs/evidence/2026-11-04/quality/refine_dev_pcm16_eval)。

在线处理时间取决于设备、音频长度和迭代数；扩展实验已观察到新句的防护失败。单条样例不证明人耳不可感知或跨模型有效。本分支只公开样例和代码；权重按固定来源下载，用户上传和运行结果不进 Git。

## 附加资料与正式提测

- [本次联调验证](docs/VoiceGuard-联调验证.md)
- [华为云部署操作手册](https://github.com/Abyss-sword/Voice-Guard/blob/49ec85e862f91506ad7f70f5e9acf3053bc67263/docs/07_部署操作手册.md)
- [报告与实验记录](https://github.com/Abyss-sword/Voice-Guard/blob/49ec85e862f91506ad7f70f5e9acf3053bc67263/docs/20_最终复现总结.md)
- [课程接入规范](docs/算法接入规范.md) · [自查清单](docs/接入自查清单.md) · [提交指南](docs/算法提交指南.md)

组号分配后，更新 README、提交分支名、方法 ID、配置文件名、目录索引、样例和适配器目录路径；启动时以 `--method-id` 传入同一 ID。复测后用 `第<实际组号>组-A036 Voice Guard 语音隐私主动防御` 向 `lym-lab/algorithm_research_frontend:main` 发起一个 PR，正文链接本组提交分支的根 README，PR 保持打开。
