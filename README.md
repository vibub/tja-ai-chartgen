# tja-ai-chartgen

用于太鼓模拟器的 AI 辅助 TJA 谱面草稿生成器。

英文版文档见：[README_EN.md](README_EN.md)。

许可证：MIT。详见 [LICENSE](LICENSE)。

## 环境要求

- Python 3.11+
- ffmpeg

## 安装

```bash
pip install -e ".[dev]"
```

## 开发验证

运行代码检查和完整测试：

```bash
ruff check .
pytest
```

真实音频流水线测试会使用 ffmpeg 转换项目自带的程序化 4/4 点击轨，并继续执行 librosa 分析、小节特征构建和 TJA 导出：

```bash
pytest tests/test_audio_pipeline_integration.py -v
```

测试音频位于 `tests/fixtures/audio/`，完全由项目代码合成，不包含第三方录音或版权音乐。可以使用以下命令确定性重建：

```bash
python tests/fixtures/audio/rebuild_click_fixtures.py
```

## 使用方式

查看已安装的 CLI 版本：

```bash
tja-ai-chartgen version
```

生成规则谱面草稿：

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title"
```

只生成前 N 小节，便于快速检查：

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --max-bars 16
```

控制规则谱面草稿密度和风格模板：

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --density high --style hybrid
```

`--density` 可选值为 `auto`、`low`、`medium`、`high`、`max`。`auto` 会根据分析得到的小节能量自动选择密度。`--style` 可选值为 `technical`、`stamina`、`hybrid`、`performance`，会影响规则模板和 AI prompt。

允许规则生成器和 AI 使用简单滚奏、气球音符：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --density high \
  --special-notes
```

启用后规则生成器会在高能量小节中少量使用 `5...8` 滚奏和 `7...8` 气球，并在 `.tja` 中输出对应 `BALLOON:` 头。

在需要人工校准时覆盖自动分析得到的 BPM 和 OFFSET：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --max-bars 16 \
  --bpm 220.588 \
  --offset 0.725
```

覆盖或增强拍号分析，支持 `4/4`、`3/4` 和 `6/8`：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --time-signature 3/4
```

`3/4` 和 `6/8` 会以 12 格小节生成，并在 `.tja` 中通过 `#MEASURE 3/4` 输出；`4/4` 仍使用默认 16 格小节。

尝试使用可选 BeatNet 增强 downbeat、meter 和小节起点分析：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-beatnet
```

BeatNet 不是默认依赖。如需启用，请先安装 `BeatNet`；当 BeatNet 未安装或分析失败时，CLI 会继续使用默认 librosa 分析结果。

一次生成 Easy、Normal、Hard、Oni 四个难度：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --all-courses
```

启用 AI 辅助生成：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --course Oni \
  --level 10 \
  --style technical \
  --use-ai
```

AI 调用通过 LiteLLM 接入，并使用 OpenAI 兼容协议。推荐在 `.env` 中配置模型和连接信息，避免把 API key 写进命令历史：

```env
MODEL=openai/custom-model
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=sk-...
```

变量说明：

- `MODEL`：要调用的模型名。可以是兼容网关暴露的模型名，也可以是 LiteLLM provider 前缀形式，例如 `openai/custom-model`。
- `OPENAI_BASE_URL`：OpenAI 兼容接口地址，例如 `https://llm.example.com/v1`。
- `OPENAI_API_KEY`：OpenAI 兼容接口密钥。

配置好 `.env` 后，只需要加 `--use-ai`：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai
```

也可以用命令行参数临时覆盖 `.env` 中的配置：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai \
  --model openai/custom-model \
  --ai-base-url https://llm.example.com/v1 \
  --ai-api-key sk-... \
  --ai-request-timeout 300 \
  --ai-transport-retries 1
```

`generation_config.json` 会记录最终解析使用的 `model`、单次请求超时和 transport 重试次数，但不会记录 `OPENAI_BASE_URL` 或 `OPENAI_API_KEY`。复跑 AI 生成时，请继续通过 `.env`、环境变量或命令行参数提供连接配置。

每次 AI provider transport 尝试默认超时 300 秒，可用 `--ai-request-timeout` 在 1–600 秒范围内调整。超时、连接失败、HTTP 429 和主要 5xx 错误默认最多重试 1 次，可用 `--ai-transport-retries 0|1` 调整；LiteLLM 内部重试会被关闭，项目自行记录每次 transport 尝试的耗时和异常类别。认证或参数错误不会重试。该重试与内容修复独立：AI 输出若未通过 JSON、bar 数量、notes 长度、字符或质量校验，默认最多修复重试 2 次，可用 `--ai-repair-retries` 调整。任一阶段最终失败都会回退到规则生成器。

通过已保存的生成配置复跑：

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

启动 Web UI MVP：

```bash
tja-ai-chartgen web
tja-ai-chartgen web --host 0.0.0.0 --allow-remote
```

Web UI 支持上传音频、预览 BPM/OFFSET/小节能量分析、手动覆盖 BPM/OFFSET/拍号，并按指定小节范围重新生成规则或 AI 增强谱面片段。生成完成后会进入游玩预览，支持播放 OGG、自动演奏谱面、拖动进度条以及播放咚/咔命中音效。折叠的 AI 参数区可设置单次请求超时和 0–1 次 transport 重试；全曲生成会保存这两项非敏感设置供结果页沿用，局部 regenerate 的同步 AI 调用在线程池中执行，不阻塞 FastAPI 事件循环。默认监听 `127.0.0.1:8000`，任务文件写入 `output/web/`。监听非回环地址时必须显式提供 `--allow-remote`；该选项只确认暴露风险，不提供认证或多用户数据隔离，公开部署仍需额外的反向代理认证和访问控制。单个上传文件最大为 100 MiB；远程模式禁用请求方指定服务器目录的导出能力，任务下载端点只公开预览 OGG 和生成的 TJA，不公开 AI sidecar 或内部状态文件。

## 输出文件

```txt
output/
├─ song.ogg
├─ song.tja
├─ analysis.json
├─ generation_config.json
├─ ai_input.json
├─ ai_attempts.json
├─ ai_output.json
├─ report.txt
└─ web/
   └─ <job-id>/
      ├─ <uploaded-audio>
      ├─ <stem>.ogg
      ├─ analysis.json
      ├─ chart_bars.json
      ├─ chart_options.json
      ├─ progress.json
      ├─ preview.tja
      ├─ result.html
      ├─ ai_input_<start>_<end>.json
      ├─ ai_attempts_<start>_<end>.json
      ├─ ai_output_<start>_<end>.json
      └─ regenerated_<start>_<end>.tja
```

## 可复现性

每次执行 `generate` 都会在 TJA 输出旁写入 `generation_config.json`。该文件记录输入路径、元数据、难度、是否生成全难度、风格、密度、`--max-bars`、BPM/OFFSET 覆盖值、拍号覆盖值、BeatNet 开关、特殊音符开关、AI 开关、最终解析使用的模型名、AI 内容修复次数、请求超时和 transport 重试次数。后续可以使用 `generate-from-config` 用同一组参数重新生成谱面。`ai_attempts*.json` 会分别记录内容校验尝试和实际 transport 尝试，并在最终失败时写入稳定的 fallback 原因。API key 和 base URL 不会写入 `generation_config.json`；如需复跑 AI 生成，请继续通过 `.env`、环境变量或命令参数提供连接配置。

## 限制

- 更适合 BPM 稳定的歌曲。
- 当前默认按 `4/4` 拍处理；可通过 `--time-signature 3/4|6/8` 或 `--use-beatnet` 使用非 4/4 小节。
- 生成结果是谱面草稿，仍需要人工检查和调整。
- OFFSET 可能需要在 OpenTaiko 或其他模拟器中继续微调。
- `--max-bars` 主要用于快速检查，会将生成谱面截断到前 N 小节。
- `--bpm` 和 `--offset` 会覆盖自动分析结果，用于人工校准。
- `--time-signature 4/4|3/4|6/8` 会覆盖分析得到的拍号，并影响小节长度、AI prompt 和 `.tja` 的 `#MEASURE` 输出。
- `--all-courses` 会分别输出 `<stem>_easy.tja`、`<stem>_normal.tja`、`<stem>_hard.tja`、`<stem>_oni.tja`，并使用内置等级与密度预设。
- `--density auto|low|medium|high|max` 会控制规则生成器密度；启用 `--use-ai` 时也会传入 AI prompt。
- `--style technical|stamina|hybrid|performance` 会选择风格模板，影响规则谱面模式、特殊音符插入节奏和 AI prompt。
- `--special-notes` 会允许简单滚奏和气球音符；当前只做少量模板化插入，仍不支持复杂滚奏演出或分支语法。
- `--use-beatnet` 需要额外安装 `BeatNet`；如果 BeatNet 不可用或分析失败，会自动保留默认 librosa 分析结果。
- `--use-ai` 仍然可能因为模型不可用、输出多次修复失败或凭据配置问题回退到规则生成器。
- Web UI 是本地 MVP，支持上传、分析、游玩预览、局部重新生成和结果导出，但不提供账号、持久任务管理或完整的全曲谱面编辑器。
- MVP 不支持 BPM 变化、分歧谱面、复杂滚奏演出、滚动演出等复杂语法。

## 后续方向

以下内容仍未实现，适合后续版本继续迭代：

- 支持变 BPM、复杂滚奏演出、分歧谱面和滚动演出等更完整的 TJA 语法。
- 改进音频结构分析，减少对 BPM 稳定歌曲的依赖。
- 增强 Web UI 的任务管理、谱面编辑和长期保存能力。
- 沉淀更多可复现的参考谱面评测样例，便于比较不同生成策略。
