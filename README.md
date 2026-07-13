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

`--density` 可选值为 `auto`、`low`、`medium`、`high`、`max`。规则 fallback 以 course 和 level 建立按小节时长归一化的难度负荷，density 只在当前难度范围内调整；`auto` 会结合小节能量、瞬态、持续 activity、真实乐句进度和 `build_up` / `peak` / `breakdown` 等结构角色决定目标音符数。落点优先跟随 onset、strength 和 accent，瞬态不足时使用 beat/downbeat 补充骨架，并按难度限制高 BPM 短小节的 notes/sec 和格点占用率。`--style` 可选值为 `technical`、`stamina`、`hybrid`、`performance`，主要影响节奏倾向、咚咔配色、特殊音符节奏和 AI prompt。

允许规则生成器和 AI 使用简单滚奏、气球音符：

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --density high \
  --special-notes
```

启用后，规则生成器只会在结构分析确认的活跃 `fill_candidate` 小节中使用单小节 `5...8` 滚奏和 `7...8` 气球；仅仅处于 `phrase_end`、`song_end` 或第 4/8 小节并不足以触发特殊音符。起止格点会跟随小节特征，持续时间按实际格点跨度与小节时长计算；气球击打数再根据持续秒数、course 和 level 动态生成，并在 `.tja` 中输出对应 `BALLOON:` 头。质量报告会单独记录滚奏/气球数量、持续时间、气球目标击打数和 hits/sec，普通 notes/sec 不计算 `5`、`7`、`8` 标记。

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

音频特征会先统一量化到“每四分音符 12 tick”的 canonical 网格：`4/4` 每小节 48 tick，`3/4` 和 `6/8` 每小节 36 tick。随后 `ResolutionPlan` 根据可靠 onset 的量化误差选择输出分辨率：`4/4` 为 16/24/48 格，`3/4` 和 `6/8` 为 12/18/36 格。默认仍使用全曲稳定的 `song-global-v1`；当 `structure-v1` 找到完整、高置信乐句且局部节奏确实需要更细表达时，`phrase-stable-v2` 只会在乐句边界整段升级，切换过多则回退为全曲稳定高分辨率，不会逐小节抖动。更高分辨率只增加三连音、十六分和混合细分的表达能力，不会自动提高难度，实际负荷仍由 course、level、density 和音乐证据控制。非 `4/4` 小节会在 `.tja` 中通过 `#MEASURE 3/4` 输出。

`structure-v1` 不再按固定四小节循环硬切乐句。它结合能量 percentile、前后窗口变化、onset/activity 密度、12 维节奏轮廓和 `spectral-v1` 频谱语义，生成非固定长度的 `phrase_id`、`phrase_progress`、可复用的 `section_id`、边界置信度、`stable` / `build_up` / `peak` / `drop` / `cadence` / `breakdown` 角色和 `fill_candidate_score`。这些字段会写入 `analysis.json`，同时进入规则生成器与 AI 的紧凑 payload，用于渐进 build-up、高潮对比、breakdown 骨架、回归段落 motif 和真实收束 fill。

`spectral-v1` 使用 HPSS 分离 harmonic/percussive 成分，并提取低/中/高频 onset、spectral flux、brightness、harmonic novelty、texture novelty 和 percussive ratio。逐帧特征会汇总为小节结构字段，可靠的瞬态位置还会以稀疏 `spectral_events` 进入 AI payload；规则 fallback 会把频谱瞬态作为落点辅助，并将低频攻击视为弱咚证据、高频攻击视为弱咔证据。短音频或 librosa 频谱步骤失败时不会中断现有 onset/RMS 流水线，而会产生 `spectral-analysis-fallback` notice，并继续生成可检查的谱面草稿。

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

每次 AI provider transport 尝试默认超时 300 秒，可用 `--ai-request-timeout` 在 1–600 秒范围内调整。超时、连接失败、HTTP 429 和主要 5xx 错误默认最多重试 1 次，可用 `--ai-transport-retries 0|1` 调整；LiteLLM 内部重试会被关闭，项目自行记录每次 transport 尝试的耗时和异常类别。认证或参数错误不会重试。该重试与内容修复独立：AI 使用 `tja-ai-chartgen-compact-v2` 输入并返回 canonical tick 上的稀疏 `hits` / `long_notes` 事件，不直接决定 notes 字符串长度；统一事件编码器会校验 tick 范围、目标 resolution 可表示性、冲突和长音结构。旧 `notes` 输出 schema、JSON/bar 数量错误或质量门控失败都会进入内容修复，默认最多重试 2 次，可用 `--ai-repair-retries` 调整。任一阶段最终失败都会回退到规则生成器。

运行时 AI prompt 会按 course 和接近的 level，从仓库内的匿名连续参考窗口中选取 intro、peak、cadence 各一段。当前静态数据由 5 组本地参考、23 个 course 离线生成，共 69 个连续窗口；窗口只保留匿名 source ID、course/level、每小节 resolution/measure/BPM/GOGO 和稀疏事件，不保存标题、WAVE、外部绝对路径或完整 notes 字符串。运行时不会访问原始参考目录；静态窗口不可读时才回退到旧的均匀抽样参考片段。

通过已保存的生成配置复跑：

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

启动 Web UI MVP：

```bash
tja-ai-chartgen web
tja-ai-chartgen web --host 0.0.0.0 --allow-remote
```

Web UI 支持上传音频、预览 BPM/OFFSET/小节能量分析、手动覆盖 BPM/OFFSET/拍号，并按指定小节范围重新生成规则或 AI 增强谱面片段。生成完成后会进入游玩预览，支持播放 OGG、自动演奏谱面、拖动进度条以及播放咚/咔命中音效。分析置信度不足、BeatNet 未生效、`spectral-v1` fallback、基础 resolution 降级、AI fallback 和最终写入错误会通过结构化 `GenerationNotice` 同时显示在进度页与结果页，并持久化为 `generation_notices*.json`；CLI 也会写入同类 sidecar 和 `report.txt`。折叠的 AI 参数区可设置单次请求超时和 0–1 次 transport 重试；全曲生成会保存这两项非敏感设置供结果页沿用，局部 regenerate 的同步 AI 调用在线程池中执行，不阻塞 FastAPI 事件循环。默认监听 `127.0.0.1:8000`，任务文件写入 `output/web/`。监听非回环地址时必须显式提供 `--allow-remote`；该选项只确认暴露风险，不提供认证或多用户数据隔离，公开部署仍需额外的反向代理认证和访问控制。单个上传文件最大为 100 MiB；远程模式禁用请求方指定服务器目录的导出能力，任务下载端点只公开预览 OGG 和生成的 TJA，不公开 AI sidecar 或内部状态文件，且公开进度/结果会隐藏服务器路径、连接地址和 provider 响应详情。

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
├─ generation_notices.json
├─ report.txt
└─ web/
   └─ <job-id>/
      ├─ <uploaded-audio>
      ├─ <stem>.ogg
      ├─ analysis.json
      ├─ chart_bars.json
      ├─ chart_options.json
      ├─ progress.json
      ├─ generation_notices.json
      ├─ preview.tja
      ├─ result.html
      ├─ ai_input_<start>_<end>.json
      ├─ ai_attempts_<start>_<end>.json
      ├─ ai_output_<start>_<end>.json
      ├─ generation_notices_<start>_<end>.json
      └─ regenerated_<start>_<end>.tja
```

## 可复现性

每次执行 `generate` 都会在 TJA 输出旁写入 `generation_config.json`。该文件记录输入路径、元数据、难度、是否生成全难度、风格、密度、`--max-bars`、BPM/OFFSET 覆盖值、拍号覆盖值、BeatNet 开关、特殊音符开关、AI 开关、最终解析使用的模型名、AI 内容修复次数、请求超时和 transport 重试次数。后续可以使用 `generate-from-config` 用同一组参数重新生成谱面。`ai_attempts*.json` 会分别记录内容校验尝试和实际 transport 尝试，并在最终失败时写入稳定的 fallback 原因。API key 和 base URL 不会写入 `generation_config.json`；如需复跑 AI 生成，请继续通过 `.env`、环境变量或命令参数提供连接配置。

## 质量评测

规则生成器的四难度负荷已使用匿名真实四难度谱面聚合结果进行初步校准，并通过项目代码生成的 120 BPM sparse、180 BPM dense、16/24/48 自适应分辨率，以及稳定段→build-up→peak→drop 的结构 WAV 夹具执行可重复真实音频回归；同一流水线还验证 `spectral-v1` envelope 对齐、非空 flux、小节/稀疏格点映射和失败降级。`quality_report*.json` 除总体平均和单小节峰值 notes/sec 外，还记录活跃小节平均 notes/sec、最长连续流数量/时长、重音覆盖率、归一化 pattern 重复率、结构密度相关性、高潮对比、build-up 斜率一致性、收束变化率、fill 候选精度、重复 section motif 一致性，以及 resolution 切换/量化误差。普通 hit 数、NPS、连续流、重音和重复统计按真实时间或小节相对位置计算，不会因等价节奏编码成 16/24/48 格而改变。普通负荷只统计 `1`–`4`，特殊音符继续单独统计；新增结构指标目前只用于报告，不参与统一加权总分或 AI repair 门控。

可用 `python tools/analyze_reference_dataset.py <reference-dir>` 离线分析用户提供的同 stem 音频/TJA 目录，输出多 course、BPMCHANGE、MEASURE、GOGO 和可变 resolution 的匿名聚合指标；也可用 `python tools/build_reference_windows.py <reference-dir> --output src/tja_ai_chartgen/ai/reference_windows.json` 离线重建连续参考窗口。运行时生成不读取该目录，也不会把参考音频、标题、原始 notes 或绝对路径写入仓库。自动门槛覆盖静音保护、四难度梯度、稀疏音乐不过度填充、高 BPM 上限、dense Hard/Oni 区分、结构角色、乐句级稳定分辨率、确定性和结构化 preflight。配色、手感、fill 趣味性和星级体感仍需人工游玩与试听。指标定义、匿名校准范围、复现流程和人工检查清单见 [docs/quality-evaluation.md](docs/quality-evaluation.md)。

## 限制

- 更适合 BPM 稳定的歌曲。
- 当前默认按 `4/4` 拍处理；可通过 `--time-signature 3/4|6/8` 或 `--use-beatnet` 使用非 4/4 小节。
- 生成结果是谱面草稿，仍需要人工检查和调整。
- OFFSET 可能需要在 OpenTaiko 或其他模拟器中继续微调。
- `--max-bars` 主要用于快速检查，会将生成谱面截断到前 N 小节。
- `--bpm` 和 `--offset` 会覆盖自动分析结果，用于人工校准。
- `--time-signature 4/4|3/4|6/8` 会覆盖分析得到的拍号，并影响小节长度、AI prompt 和 `.tja` 的 `#MEASURE` 输出。
- `--all-courses` 会分别输出 `<stem>_easy.tja`、`<stem>_normal.tja`、`<stem>_hard.tja`、`<stem>_oni.tja`，并使用 Easy 3、Normal 5、Hard 7、Oni 10 的内置等级建立 BPM 感知的负荷梯度。
- `--density auto|low|medium|high|max` 会在当前 course/level 难度范围内调整规则生成器密度；`--all-courses` 的四张谱面共享用户选择的 density。启用 `--use-ai` 时 density 也会传入 AI prompt。
- `--style technical|stamina|hybrid|performance` 会选择风格模板，影响规则谱面的节奏倾向、咚咔配色、特殊音符候选门槛和 AI prompt；普通落点仍优先跟随分析得到的音乐特征。
- `--special-notes` 只会在结构分析确认的活跃 fill 候选中生成单小节滚奏和动态击打数气球；仍不支持跨小节滚奏、复杂滚奏演出或分支语法。
- `--use-beatnet` 需要额外安装 `BeatNet`；如果 BeatNet 不可用或分析失败，会自动保留默认 librosa 分析结果。
- `--use-ai` 仍然可能因为模型不可用、输出多次修复失败或凭据配置问题回退到规则生成器。
- Web UI 是本地 MVP，支持上传、分析、游玩预览、局部重新生成和结果导出，但不提供账号、持久任务管理或完整的全曲谱面编辑器。
- MVP 不支持 BPM 变化、分歧谱面、复杂滚奏演出、滚动演出等复杂语法。

## 后续方向

以下内容仍未实现，适合后续版本继续迭代：

- 支持变 BPM、复杂滚奏演出、分歧谱面和滚动演出等更完整的 TJA 语法。
- 使用更多真实歌曲和人工游玩结果校准 `spectral-v1` 与结构阈值，并评估哪些频谱/结构指标适合进入 AI repair 或 CI 门槛。
- 增强 Web UI 的任务管理、谱面编辑和长期保存能力。
- 沉淀更多可复现的参考谱面评测样例，便于比较不同生成策略。
