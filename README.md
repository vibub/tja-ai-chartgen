# tja-ai-chartgen

用于太鼓模拟器的 AI 辅助 TJA 谱面草稿生成器。

英文版文档见：[README_EN.md](README_EN.md)。

## 环境要求

- Python 3.11+
- ffmpeg

## 安装

```bash
pip install -e ".[dev]"
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

控制规则谱面草稿密度：

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --density high
```

`--density` 可选值为 `auto`、`low`、`medium`、`high`、`max`。`auto` 会根据分析得到的小节能量自动选择密度。

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
  --ai-api-key sk-...
```

`generation_config.json` 会记录最终解析使用的 `model`，但不会记录 `OPENAI_BASE_URL` 或 `OPENAI_API_KEY`。复跑 AI 生成时，请继续通过 `.env`、环境变量或命令行参数提供连接配置。

AI 输出会先做严格格式校验；如果 JSON、bar 数量、notes 长度或字符不符合 MVP 约束，CLI 会自动向模型发送修复提示并重试。默认最多修复重试 2 次，可以用 `--ai-repair-retries` 调整。修复仍失败时会回退到规则生成器。

通过已保存的生成配置复跑：

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

## 输出文件

```txt
output/
├─ song.ogg
├─ song.tja
├─ analysis.json
├─ generation_config.json
├─ ai_input.json
├─ ai_output.json
└─ report.txt
```

## 可复现性

每次执行 `generate` 都会在 TJA 输出旁写入 `generation_config.json`。该文件记录输入路径、元数据、难度、风格、密度、`--max-bars`、BPM/OFFSET 覆盖值、拍号覆盖值、BeatNet 开关、AI 开关、最终解析使用的模型名和 AI 修复重试次数。后续可以使用 `generate-from-config` 用同一组参数重新生成谱面。API key 和 base URL 不会写入 `generation_config.json`；如需复跑 AI 生成，请继续通过 `.env`、环境变量或命令参数提供连接配置。

## 限制

- 更适合 BPM 稳定的歌曲。
- 当前默认按 `4/4` 拍处理；可通过 `--time-signature 3/4|6/8` 或 `--use-beatnet` 使用非 4/4 小节。
- 生成结果是谱面草稿，仍需要人工检查和调整。
- OFFSET 可能需要在 OpenTaiko 或其他模拟器中继续微调。
- `--max-bars` 主要用于快速检查，会将生成谱面截断到前 N 小节。
- `--bpm` 和 `--offset` 会覆盖自动分析结果，用于人工校准。
- `--time-signature 4/4|3/4|6/8` 会覆盖分析得到的拍号，并影响小节长度、AI prompt 和 `.tja` 的 `#MEASURE` 输出。
- `--density auto|low|medium|high|max` 会控制规则生成器密度；启用 `--use-ai` 时也会传入 AI prompt。
- `--use-beatnet` 需要额外安装 `BeatNet`；如果 BeatNet 不可用或分析失败，会自动保留默认 librosa 分析结果。
- `--use-ai` 仍然可能因为模型不可用、输出多次修复失败或凭据配置问题回退到规则生成器。
- MVP 不支持 BPM 变化、分歧谱面、滚奏、气球音符或滚动演出等复杂语法。

## 后续版本规划

以下内容计划在后续版本中实现，不属于 MVP 范围。

### v0.2

- 支持用户手动指定 BPM。✅ 已在 MVP 迭代中通过 `--bpm` 实现。
- 支持用户手动指定 OFFSET。✅ 已在 MVP 迭代中通过 `--offset` 实现。
- 支持 `--max-bars` 只生成前 N 小节，方便测试。✅ 已在 MVP 迭代中实现。
- AI 输出自动修复重试。✅ 已实现，默认最多修复重试 2 次，可通过 `--ai-repair-retries` 调整。
- 加入 `--density low|medium|high|max`。✅ 已在 MVP 迭代中通过 `--density auto|low|medium|high|max` 实现。

### v0.3

- 尝试集成 BeatNet：✅ 已通过可选 `--use-beatnet` 增强实现，失败时回退到 librosa。
  - downbeat
  - meter
  - 更准的小节开始
- 支持 `3/4`、`6/8`。✅ 已通过 `--time-signature`、BeatNet meter、12 格小节和 `#MEASURE` 输出实现。
- 支持简单滚奏和气球。

### v0.4

- 支持多难度：
  - Easy
  - Normal
  - Hard
  - Oni
- 支持风格模板：
  - technical
  - stamina
  - hybrid
  - performance

### v0.5

- 做 Web UI。
- 上传音频。
- 在线预览分析结果。
- 手动调 BPM / OFFSET。
- 重新生成指定小节。
