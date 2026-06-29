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

启用 AI 辅助生成：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --course Oni \
  --level 10 \
  --style technical \
  --use-ai
```

AI 调用通过 LiteLLM 接入。默认读取 `MODEL`，也可以通过 `--model` 指定模型。使用 OpenAI 兼容接口时，模型名通常需要使用 LiteLLM 的 `openai/` 前缀，并传入自定义 base URL 和 API key：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai \
  --model openai/custom-model \
  --ai-base-url https://llm.example.com/v1 \
  --ai-api-key sk-...
```

也可以在 `.env` 中配置，避免把 API key 写进命令历史：

```env
MODEL=openai/custom-model
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=sk-...
```

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

每次执行 `generate` 都会在 TJA 输出旁写入 `generation_config.json`。该文件记录输入路径、元数据、难度、风格、密度、`--max-bars`、BPM/OFFSET 覆盖值、AI 开关、最终解析使用的模型名和 AI 修复重试次数。后续可以使用 `generate-from-config` 用同一组参数重新生成谱面。API key 和 base URL 不会写入 `generation_config.json`；如需复跑 AI 生成，请继续通过 `.env`、环境变量或命令参数提供连接配置。

## 限制

- 更适合 BPM 稳定的歌曲。
- 当前默认按 `4/4` 拍处理。
- 生成结果是谱面草稿，仍需要人工检查和调整。
- OFFSET 可能需要在 OpenTaiko 或其他模拟器中继续微调。
- `--max-bars` 主要用于快速检查，会将生成谱面截断到前 N 小节。
- `--bpm` 和 `--offset` 会覆盖自动分析结果，用于人工校准。
- `--density auto|low|medium|high|max` 会控制规则生成器密度；启用 `--use-ai` 时也会传入 AI prompt。
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

- 尝试集成 BeatNet：
  - downbeat
  - meter
  - 更准的小节开始
- 支持 `3/4`、`6/8`。
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
