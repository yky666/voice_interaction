# Voice Interaction

这个目录是独立的语音交互服务，不会污染 `pi05_g1_flask_services`。

链路是：

`麦克风 -> ASR -> 记忆 -> LLM -> TTS -> 扬声器`

同时可以把动作类意图转发给：

- `http://127.0.0.1:8081/infer`
- 可选继续转发到 `http://127.0.0.1:8082/execute`
- 也可以把明确的 `skill_name` 直接路由到 `http://127.0.0.1:8082/run_program`

## 目录

```text
voice_interaction/
├── .env.example
├── README.md
├── requirements.txt
├── run_live_companion.sh
├── run_voice_interaction.sh
├── example_requests/
│   └── dialog_turn_request.json
└── src/
    ├── audio_io.py
    ├── common_utils.py
    ├── live_companion.py
    ├── dialog_memory.py
    ├── dialog_pipeline.py
    └── dialog_flask_service.py
```

## 环境要求

- 推荐 Python `3.10`
- Jetson 本地 ASR 建议在 CUDA 可用环境下运行
- 需要系统层音频工具：`arecord`、`aplay`
- 如果要播放更多音频格式，建议额外安装 `ffmpeg`

## 安装

```bash
cd /home/unitree/voice_interaction
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./run_voice_interaction.sh
```

如果你要在 Jetson 上启用本地 `faster-whisper` ASR，再额外安装：

```bash
pip install -r requirements-jetson-local-asr.txt
```

默认端口：

```text
http://127.0.0.1:8083
```

## 快速测试

健康检查：

```bash
curl http://127.0.0.1:8083/health
```

启动真机本地语音循环：

```bash
./run_live_companion.sh
```

这会直接做：

1. 麦克风录音
2. 基于能量阈值的说话检测
3. 把音频送给 `/turn`
4. 播放 TTS 音频
5. 播放时如果再次检测到你开口，会尝试打断当前播放

文本对话：

```bash
curl -X POST http://127.0.0.1:8083/turn \
  -H 'Content-Type: application/json' \
  -d @/home/unitree/voice_interaction/example_requests/dialog_turn_request.json
```

写入长期记忆：

```bash
curl -X POST http://127.0.0.1:8083/memory/fact \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-session-001","key":"preferred_name","value":"小张"}'
```

查看记忆：

```bash
curl http://127.0.0.1:8083/memory/demo-session-001
```

## 唤醒词

默认关闭。开启后可以要求用户先说唤醒词，例如 `小智`：

```bash
VOICE_WAKE_WORD_ENABLED=true
VOICE_WAKE_WORD=小智
VOICE_WAKE_DURATION_SEC=30
```

开启后：

- 当前句包含 `小智` 就会唤醒
- 唤醒后 30 秒内同一个 `session_id` 的后续句子默认继续生效
- 超时后需要再次说唤醒词

这是本地服务内实现的唤醒逻辑，不依赖宇树官方现成语音唤醒框架。

## 切换真实后端

### ASR

OpenAI-compatible 可直接部署版本：

```bash
VOICE_ASR_MODE=openai_compatible
VOICE_ASR_API_BASE=https://api.openai.com/v1
VOICE_ASR_API_KEY=你的key
VOICE_ASR_MODEL=gpt-4o-mini-transcribe
VOICE_ASR_LANGUAGE=zh
```

- `VOICE_ASR_MODE=command`
- `VOICE_ASR_COMMAND="python /path/to/asr_server.py"`

或者：

- `VOICE_ASR_MODE=http`
- `VOICE_ASR_HTTP_URL=http://127.0.0.1:9001/asr`

Jetson 本地版 `faster-whisper`：

```bash
VOICE_ASR_MODE=faster_whisper_local
VOICE_ASR_MODEL=large-v3-turbo
VOICE_ASR_LANGUAGE=zh
VOICE_ASR_DEVICE=cuda
VOICE_ASR_COMPUTE_TYPE=float16
VOICE_ASR_BEAM_SIZE=5
```

说明：

- 这一路不依赖外部云 ASR
- 首次加载模型会比较慢
- 在 Orin 16GB 上更建议先从 `small` 或 `medium` 验证，再上 `large-v3-turbo`
- Python 版本更建议 `3.10`，不要优先用 `3.13`

### LLM

OpenAI-compatible：

```bash
VOICE_LLM_MODE=openai_compatible
VOICE_LLM_API_BASE=https://api.openai.com/v1
VOICE_LLM_API_KEY=...
VOICE_LLM_MODEL=gpt-4o-mini
```

### TTS

OpenAI-compatible 可直接部署版本：

```bash
VOICE_TTS_MODE=openai_compatible
VOICE_TTS_API_BASE=https://api.openai.com/v1
VOICE_TTS_API_KEY=你的key
VOICE_TTS_MODEL=gpt-4o-mini-tts
VOICE_TTS_VOICE=alloy
VOICE_TTS_AUDIO_FORMAT=wav
```

- `VOICE_TTS_MODE=command`
- `VOICE_TTS_COMMAND="python /path/to/tts_server.py"`

或者：

- `VOICE_TTS_MODE=http`
- `VOICE_TTS_HTTP_URL=http://127.0.0.1:9002/tts`

## 风格修改

直接改 `.env` 里的这几个字段即可：

- `VOICE_SYSTEM_PROMPT`
- `VOICE_STYLE_PROMPT`
- `VOICE_RESPONSE_FORMAT_PROMPT`

如果你要“固定人设+固定语气”，优先改这三个，而不是去改用户输入文本。

更像陪伴机器人的一组示例：

```bash
VOICE_SYSTEM_PROMPT="You are a warm companion robot assistant."
VOICE_STYLE_PROMPT="Speak gently, offer reassurance, notice emotions, and prefer short comforting sentences."
```

## 陪伴式记忆和情绪

现在这版已经补了几类陪伴逻辑：

- 会保存长期事实记忆
- 能记住类似“我伤心的时候想听爱错”这种偏好
- 会识别简单情绪：难过、焦虑、开心、愤怒
- 当检测到难过时，会优先给安慰性回复
- 如果记忆里有“难过时想听的歌”，会主动提起
- 如果你在 `VOICE_MEDIA_LIBRARY_JSON` 里把歌名映射到本地音频文件，`live_companion` 还能直接播放那首歌
- 如果开启 `VOICE_ENABLE_COMFORT_ACTIONS=true`，检测到明显难过语句时会生成明确 skill：`comfort_hug`
- `comfort_hug` 默认会通过 `VOICE_G1_PROGRAM_URL` 路由到 `g1_flask_service` 的 `/run_program`
- 默认映射是 `comfort_hug -> sdk_arm_hug`，也就是复用宇树官方 SDK 里的拥抱动作 RPC

默认 skill 映射示例：

```bash
VOICE_SKILL_PROGRAM_MAP='{"comfort_hug":"sdk_arm_hug","handshake":"sdk_arm_shake_hand","high_five":"sdk_arm_high_five","greet_bow":"greet_bow"}'
```

如果你暂时只想验证链路、不让真机真动：

```bash
VOICE_SKILL_EXECUTE_MOTION=false
```

如果你已经完成安全确认，才改为：

```bash
VOICE_SKILL_EXECUTE_MOTION=true
```

歌曲播放示例：

```bash
VOICE_MEDIA_LIBRARY_JSON='{"爱错":"/home/unitree/media/爱错.wav"}'
```

## 真机部署建议

建议顺序：

1. 先启动 `pi05_flask_service` 和 `g1_flask_service`
2. 再启动 `run_voice_interaction.sh`
3. 先用 `/turn` 做 API 测试
4. 再运行 `./run_live_companion.sh`
5. 最后再考虑是否打开 `VOICE_ENABLE_G1_EXECUTE=true` 或 `VOICE_SKILL_EXECUTE_MOTION=true`

更稳的陪伴式动作路径建议优先级：

1. `skill_name -> /run_program`
2. `action_prompt -> /infer -> /execute`

原因是前者更可控，也更适合“拥抱、挥手、鞠躬”这类固定社交动作。

## 打断能力说明

这版打断是工程可用版：

- 播放 TTS 时额外监听麦克风
- 连续检测到语音能量超过阈值时，尝试终止当前播放

不是完整的流式端到端 barge-in，但已经适合真机部署和继续迭代。

## 关于宇树自带语音能力

从当前本机代码和公开资料来看，可以确认两件事：

- 宇树公开页面确实提到 G1/比赛版支持多模态交互、ASR、TTS、音频灯光服务，以及 UnifoLM 大模型能力。
- 但你这台机器里可见的公开 Python SDK，当前只直接暴露了比较基础的 VUI 控制项，比如音量、亮度、开关，没有看到“内建对话大模型 / 唤醒 / 唱歌 / 情绪陪伴”的公开高层 API。

所以更稳妥的判断是：

- 宇树演示里的“小智/唱歌/联网回复”大概率走的是它们自家的闭环链路：麦阵列 + 唤醒 + 云端或内置对话服务 + 内部动作层。
- 这条链路在你当前可见的开放 SDK 里并没有完整暴露出来，至少本地代码里没有现成可复用的高层接口。
- 能不能复用，取决于你的固件版本、官方 App、以及是否拿到了对应的内部接口文档；仅凭当前公开 SDK，不能直接接进我们这套自定义服务。

公开网页可参考：

- Unitree G1 官方页：<https://www.unitree.com/en/g1>
- Unitree G1-Comp 页面，里面明确写了支持 `TTS`、`ASR`、音频灯光服务、多模态交互：<https://www.unitree.com/cn/robocup/>

如果你的目标是“先跑通、再逐步增强”，当前这套自建链路通常比等官方闭源接口更可控。

## License

本仓库默认使用 `MIT License`。

## 建议的真机测试顺序

1. 先保持 `VOICE_ASR_MODE=mock`、`VOICE_TTS_MODE=mock`，只测 `/turn` 和记忆。
2. 再切到 `VOICE_ASR_MODE=faster_whisper_local`，只验证听写质量和延迟。
3. 再接真实 TTS。
4. 最后才打开 `VOICE_SKILL_EXECUTE_MOTION=true`，验证 `comfort_hug` 是否能稳定触发 `sdk_arm_hug`。

建议的接口检查：

```bash
curl http://127.0.0.1:8082/programs
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8083/health
```

陪伴动作 dry-run：

```bash
curl -X POST http://127.0.0.1:8083/turn \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","user_text":"我今天很难过，能抱抱我吗？"}'
```

你应该在返回里看到：

- `llm_result.skill_name = "comfort_hug"`
- `skill_result.program_id = "sdk_arm_hug"`

## 语料和微调建议

第一阶段先不要训练，先做提示词和记忆层：

- 把标准答复风格写进 `VOICE_STYLE_PROMPT`
- 把常见术语、称呼、禁语做成长期记忆或规则
- 收集真实对话日志，再决定是否微调

第二阶段再做监督微调：

- 准备 `用户输入 -> 理想回复(JSON结构)` 语料
- 重点覆盖你的语气、称呼、动作意图抽取方式
- 先微调聊天模型，不建议先动动作策略模型
