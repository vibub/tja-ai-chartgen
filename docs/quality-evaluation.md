# 谱面质量评测

本文说明规则或 AI 谱面草稿的可重复质量评测方法。自动指标用于发现明显回归和解释生成结果，不替代人工游玩、试听或官方星级判断。

## 样本边界

### 项目合成音频

项目使用 `tests/fixtures/audio/rebuild_click_fixtures.py` 通过 Python 标准库确定性生成 17 个 WAV，不包含第三方录音：

- `click_4_4.wav`：120 BPM、4/4 基础四分音符点击轨，用于 BPM、offset、beat/downbeat 和完整流水线基线。
- `click_4_4_leadin.wav`：带前导静音的基础点击轨，用于确认首个实际事件与分析起点处理。
- `sparse_120.wav`：包含单瞬态稀疏小节和低活动段，用于 sparse/rest、密度上限和无证据 note 评测。
- `dense_180.wav`：稳定八分瞬态、局部十六分瞬态、重拍增强和短 break，用于高 BPM 密度与连续流评测。
- `transient_noise_intro_120.wav`：首部静音内含低电平毛刺，用于 onset 误检和静音区违规 note 评测。
- `straight_120.wav`：每拍四等分，用于 16 格 resolution 和直拍量化基线。
- `triplet_120.wav`：每拍三等分，用于 24 格 resolution 和三连音量化基线。
- `mixed_120.wav`：直拍与三连音小节交替，用于确认低分辨率不足时升级到 48 格。
- `structure_build_up_120.wav`：12 小节稳定段→逐步加密→peak→drop，用于真实乐句边界和结构密度响应。
- `syncopated_120.wav`：包含非主拍切分事件，用于 off-beat onset 与谱面响应评测。
- `pickup_120.wav`：首个 onset 早于 first downbeat，用于真正弱起检测。
- `band_attacks_120.wav`：低频主拍和高频反拍，用于 low/high band onset 和咚咔软证据评测。
- `harmonic_sparse_120.wav`：持续 harmonic 背景叠加稀疏 percussive onset，用于区分持续活动与瞬态。
- `fill_burst_120.wav`：包含两个明确短 fill burst，用于 fill onset 和谱面 fill 响应率。
- `meter_3_4_120.wav`：3/4 拍号与 36 canonical tick 基线。
- `meter_6_8_120.wav`：6/8 复拍子、四分音符 BPM 和两个复拍重音基线。
- `tempo_ambiguity_120.wav`：半小节重音明显、间拍较弱，用于半速/倍速歧义评测。

每个 `<stem>.wav` 都有同目录的 `<stem>.events.json`，由同一组 `FixtureSpec` 同步生成 BPM、拍号、时长、onset、强 onset、beat、downbeat、静音区、频带事件、fill 区间和结构段落。完整 fixture 集覆盖 ffmpeg 转换、librosa/onset-grid、canonical `BarFeature`、`spectral-v1`、`structure-v1`、`ResolutionPlan`、四难度规则生成、`QualityReport`、结构化 preflight 和 CP932 TJA 写入。

### Phase 0 自动基线

音频分析基线使用：

```bash
python tools/benchmark_audio_fixtures.py
```

该命令生成 `tests/fixtures/audio/audio_benchmark_baseline.json` 和 `.md`。`audio-alignment-v2` 使用一对一容差匹配统计 onset、强 onset、beat、downbeat、low/high band、pickup 和 fill 的 precision/recall/F1，并记录 BPM、拍号、first downbeat、静音误检及 ResolutionPlan 的可表达事件比例、量化误差、under-resolved 和不必要高 resolution 小节。默认容差为 onset 50 ms、beat/downbeat 70 ms，默认不启用 BeatNet，也不访问网络。

谱面对齐基线使用：

```bash
python tools/benchmark_chart_alignment.py
```

该命令为每个 fixture 生成 Easy 3、Normal 5、Hard 7 和 Oni 10，共 68 张固定 `technical` / `auto` 规则谱面，并写入 `chart_alignment_baseline.json` 和 `.md`。`chart-alignment-v1` 统计 note/onset 对齐、强 onset 响应、downbeat 响应、fill onset 响应、无证据 note、静音区违规和完整 `ChartBar` 重复生成确定性。高难度谱面允许使用 beat、activity 或结构证据补充不直接对应 ground-truth onset 的可玩落点，因此 note/onset precision 只作为修改前后 A/B 指标，不是统一阻断阈值。

正式阶段验收使用：

```bash
python tools/verify_phase_zero.py
```

`phase-zero-acceptance-v1` 在两个临时目录中重建全部 35 个 fixture/ground-truth 产物，在禁用 socket 连接时分别运行两次音频与谱面对齐 benchmark，并检查 baseline schema、覆盖范围、同环境稳定性和持久化隐私。验收结果写入 `phase_zero_acceptance.json` 和 `.md`；跨 librosa、BeatNet 或平台版本不要求 baseline 数值逐字节一致，但同一环境重复运行必须稳定。

Phase 1 节奏显著性验收使用：

```bash
python tools/verify_phase_one.py
```

`phase-one-acceptance-v1` 在禁用 `socket.connect` / `connect_ex` 时，对全部 17 个合成 fixture 独立运行两次 librosa/onset-grid、canonical `BarFeature`、`structure-v1` 和完整 hit/accent/don-ka salience 流水线。验收检查所有稀疏点位于 canonical grid 且无重复或乱序、两次输出完全一致、首尾静音小节全零、confidence 与稳定 fallback reason 一致，并以 70 ms 容差对 `hit >= 0.45` 的峰值和 ground-truth onset 做一对一匹配；阻断门槛为 precision ≥ 0.98、recall ≥ 0.95，稀疏点占 canonical dense grid 的比例不得超过 0.50。结果写入 `phase_one_acceptance.json` 和 `.md`。当前 Phase 1 不增加 AI payload 字段，fallback generator、AI prompt 和质量报告的消费迁移仍分别属于后续阶段。

### 匿名真实谱面校准

项目曾使用 5 组包含 Easy、Normal、Hard、Oni 的本地参考谱面完成初步校准。参考文件不是项目依赖，未复制到仓库；项目不保存其绝对路径、歌曲名、原始小节或逐曲 notes。

现在可通过稳定离线工具重新生成匿名聚合：

```bash
python tools/analyze_reference_dataset.py <reference-dir>
python tools/analyze_reference_dataset.py <reference-dir> --output reference_benchmark.json
```

目录中的每个 `.tja` 必须存在同 stem 音频。解析器支持多 course、`#BPMCHANGE`（包括小节中途变化）、`#MEASURE`、`#GOGOSTART/#GOGOEND`、barline 状态和可变 notes resolution；遇到分支、delay、scroll 等尚未支持的命令会显式失败。输出只包含内容哈希 ID、course/level 数量、resolution 分布与切换数、普通 NPS、GOGO NPS 和特殊音符聚合，不包含标题、WAVE、绝对路径或原始 notes。

连续 AI 参考窗口使用独立离线命令生成：

```bash
python tools/build_reference_windows.py <reference-dir> \
  --output src/tja_ai_chartgen/ai/reference_windows.json \
  --window-size 8
```

当前静态产物来自 5 个匿名 source、23 个 course，共 69 个 intro/peak/cadence 窗口。每小节只保存相对编号、source resolution、measure、BPM/BPMCHANGE、GOGO、普通 hit 数、稀疏 `[source_index, note]` 事件和气球计数；不保存完整 notes 字符串。运行时按 course 与相近 level 选择最多 3 个窗口，不访问参考目录。

初步聚合结果如下，普通 notes/sec 只统计 `1`–`4`：

| Course | 样本数 | Level 范围 | 全曲平均 NPS 范围 | 单小节峰值 NPS 范围 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 5 | 2–4 | 0.887–1.558 | 1.667–3.083 |
| Normal | 5 | 3–6 | 1.324–2.438 | 2.958–5.133 |
| Hard | 5 | 5–7 | 2.825–3.932 | 5.325–6.417 |
| Oni | 5 | 7–9 | 4.242–6.024 | 7.100–10.021 |

这些范围用于校准负荷方向和数量级，不是逐曲拟合目标。参考谱面含休止、变拍、BPM change 和项目暂不支持的语法，因此全曲平均值不能直接作为每个活跃小节的生成目标。

## 自适应分辨率评测

内部特征统一使用每四分音符 12 tick 的 canonical 网格，因此 `4/4` 为 48 tick，`3/4` 和 `6/8` 为 36 tick。输出候选为：

- `4/4`：16、24、48；
- `3/4` 和 `6/8`：12、18、36；
- 无可靠 onset 时回退到 16/12 的基础分辨率，并产生 `resolution-evidence-fallback` notice。

`ResolutionDecision.candidate_errors` 记录每个候选相对 canonical tick 的 onset 强度加权平均量化误差。算法选择第一个误差不高于 0.2 canonical tick 的候选；如果较低候选无法表达三连音或混合细分，则升级到 24/48 或 18/36。`evidence_count`、`confidence` 和 `reason` 用于解释选择，不作为谱面难度分数。

`phrase-stable-v2` 只在满足以下条件时替代默认 `song-global-v1`：结构分析得到至少两个完整乐句、乐句至少 2 小节、局部有足够可靠 onset、基础 resolution 的局部误差至少 0.35，并且切换发生在边界置信度不低于 0.4 的乐句边界。同一 `section_id` 的回归段落复用不低于首次出现的 resolution。切换数超过 `max(2, bar_count // 8)` 时，改用全曲稳定的最高所需 resolution；因此不会逐小节抖动。

离线参考 benchmark 的 `resolution_counts` 和 `resolution_change_count` 用于观察真实谱面的分布与切换习惯；生成回归使用 straight/triplet/mixed 验证 16/24/48 的确定性选择，并使用 straight→triplet→straight 乐句验证 `[16]×4 + [24]×4 + [16]×4` 只在边界切换。高 resolution 只表示更细时间坐标，不应增加目标 hit 数、格点占用上限或 course 难度。

## 难度负荷校准结果

### Course/level 基线

规则生成器继续由 course/level 决定绝对难度，density 只做当前难度内偏移。校准后的 profile 范围是活跃音乐上的目标，不等同于整首参考谱面的最终平均值：

| Course | 目标 NPS 范围 | Speed cap | 最大格点占用率 |
| --- | ---: | ---: | ---: |
| Easy | 1.1–2.1 | 3.0 | 0.38 |
| Normal | 1.8–3.4 | 5.0 | 0.50 |
| Hard | 3.2–5.8 | 7.5 | 0.70 |
| Oni | 4.0–7.0 | 10.0 | 0.82 |

level 在各 course 的范围内线性插值。最终 hit 数还会受到小节时长、音乐丰富度、density hint、speed cap 和 occupancy cap 限制。

### Density 偏移

| Density | NPS 乘数 |
| --- | ---: |
| low | 0.75 |
| medium | 0.90 |
| auto | 1.00 |
| high | 1.10 |
| max | 1.20 |

旧的 `low=0.3` 会使同一 course 内跨度过大。收窄后，density 表达编谱偏好，而不是替代 course。

### 音乐证据上限

- `silent` 和 `rest` 始终优先，不因难度增加音符。
- `sparse` 继续使用严格上限。
- `normal` 小节的可用 hit 上限同时参考 onset、beat 和持续 activity 数量；高难度可以增加骨架，但不能只因 profile 较高而铺满弱格点。
- `dense` 和 `fill` 的通用上限放宽，使 Hard/Oni 不再被同一个低上限完全压平。
- 所有小节仍受 course speed cap、occupancy cap 和格点数限制。

## 结构分析评测

`structure-v1` 以完整歌曲的小节向量工作，而不是固定每 4 小节切分：

- 小节向量包含 energy percentile/delta、onset density/strength、activity mean/peak/active ratio、accent density、`spectral-v1` 小节汇总，以及 12 维 rhythm/activity profile；
- 边界分数结合前后 3 小节窗口的 energy、onset、activity、rhythm profile 与相邻 drive 变化，静音↔音乐边界至少为 0.9；
- 最短乐句为 2 小节，4/8 小节长度只提供 0.02/0.04 的弱先验，超过 16 小节才强制软切分；
- transition role 包含 `stable`、`build_up`、`peak`、`drop`、`cadence`、`breakdown`；
- 完整 phrase signature 用于复用 `section_id`，相似阈值当前为 0.82；
- fill 只在真实 phrase end 评分，阈值为 0.60，候选数量最多为 `ceil(bar_count / 6)`。

结构 fixture 必须检测到 build-up、peak 和 drop，且第 4/8 小节不能仅因固定编号成为 phrase end。3、4、6、8、12 小节合成结构测试用于确认可变长度乐句；返回段落测试用于确认 `section_id` 复用。

## 频谱语义评测

`audio/spectral.py` 的 `spectral-v1` 在不改变现有 onset/RMS 失败边界的前提下增加以下证据：

- HPSS 分离 harmonic/percussive waveform；
- 基于 percussive 成分的低/中/高频 onset strength 和整体 spectral flux；
- 归一化 spectral centroid 作为 brightness；
- harmonic chroma 相邻帧变化作为 harmonic novelty；
- brightness、spectral flatness 与 percussive ratio 的联合变化作为 texture novelty；
- harmonic/percussive RMS 比得到的 percussive ratio。

所有 envelope 使用相同 hop length 并对齐为相同帧数，数值限制在 0–1。小节汇总中，多频带 onset、flux 和 novelty 使用局部峰值，brightness 与 percussive ratio 使用小节均值；逐格只保留低/中/高频 onset 和 flux 不低于 0.05 的稀疏事件，避免 `analysis.json` 和 AI prompt 为每格重复写零值。结构边界在原 energy/onset/activity/rhythm 分数不足时，可由强 timbre 差异与 novelty 补强；build-up、peak、breakdown 和 fill score 也会使用频谱趋势或 balance，但原有结构证据仍是主边界。

数字静音返回对齐的全零 `complete` 结果；短于一个 2048-sample 分析窗、参数无效或 librosa 频谱步骤异常时返回 `fallback`，不抛出到主生成流程。`analysis.json` 记录 `spectral_feature_version`、状态和稳定 reason，CLI/Web 通过 `spectral-analysis-fallback` notice 提醒用户，同时继续使用 onset、RMS、节拍和规则 fallback。单元测试覆盖静音、短音频、库异常、频谱 envelope 对齐、小节/格点映射、频谱结构边界、build-up/breakdown 角色以及低频咚/高频咔弱证据；真实 ffmpeg fixture 验证 OGG 上可产生非空 spectral flux 和稀疏格点。

## 人声与乐器语义评测

可选 `instrument-v1` 默认关闭。启用前需要安装 `.[instrument]`，并显式执行 `tja-ai-chartgen prepare-instrument-models`；固定版本的 Demucs `htdemucs`、AST AudioSet 和 manifest 默认写入项目根目录 `models/instrument-v1/`。`models/` 被 `.gitignore` 忽略，普通生成只允许本地加载，不隐式联网。

分析分为两层：

- Demucs 输出 vocals、drums、bass、other，并从各 stem 提取对齐的 RMS 活动、相对能量和 onset envelope；
- AST 使用 4 秒窗口、2 秒 hop 分析 mix 与 other stem，将固定 AudioSet 标签归并到 guitar、piano/keyboard、strings、brass、woodwind、synth、organ 和 other instrument；小节分数按 `0.65 × other + 0.35 × mix` 融合并受绝对活动门控。

每小节持久化人声 activity/presence、鼓/贝斯/伴奏 activity、主导声部、主导乐器、置信度和稳定分类分数；逐格只保存 vocal/drum/bass/accompaniment onset 至少为 0.05 的稀疏事件。首尾静音保护优先，边缘静音小节的阶段 C 特征全部归零。结构层只在活动与置信度达到门槛时使用人声进出、主导声部切换、合奏趋势和人声收束后的鼓组爆发，不允许单一标签独立决定 section 或 transition role。

状态语义：

- `complete`：Demucs 与 AST 均成功；
- `partial`：Demucs 等部分证据可用，但分类层失败；
- `fallback`：没有阶段 C 证据，基础谱面仍继续生成；
- `unavailable`：本次未启用。

默认 CI 不下载模型，使用 fake separator/classifier、合成 tensor 和固定 logits 覆盖设备选择、窗口确定性、taxonomy、mix/other 融合、canonical 映射、静音清零、结构/AI/fallback 消费以及 Web notice。准备本地模型后可运行 `pytest -m instrument_model -v` 做非默认真实推理 smoke。第一版不把识别指标作为阻断阈值，因为 source separation 泄漏、AST 标签混淆和不同曲风的门槛仍需真实歌曲与人工游玩校准。

## QualityReport 指标

### 密度与静音

- `density_compliance_rate`：满足对应 density hint 最小/最大 hit 范围的小节比例。
- `silent_bar_note_count`：歌曲首尾连续静音小节中的实际音符事件数。
- `longest_empty_bar_run`：最长连续空小节数量。

特殊音符标记参与结构活动判断，但普通 NPS 只统计 `1`–`4`。

### 重复和配色

- `repeated_bar_rate`：非空小节中，按“小节相对位置 + 音符字符”归一化后完全相同 pattern 的重复比例。
- `ka_ratio`：普通咚咔音符中咔音符的比例。
- `longest_monochrome_run`：跨小节连续同色普通音符的最长长度。

相对位置使用 `index / len(notes)` 表达，因此等价的 16/24/48 格 pattern 会得到同一签名。该指标只能发现机械重复，不能衡量节奏主题的合理复现。

### 总体和活跃负荷

- `playable_note_count`：普通音符 `1`–`4` 总数。
- `playable_duration_seconds`：可可靠计算时长的小节总时长，包括休止小节。
- `average_notes_per_second`：普通音符数除以可评估总时长。
- `peak_bar_notes_per_second`：单个可评估小节的最高普通 NPS。
- `active_duration_seconds`：至少含一个普通音符的小节时长总和。
- `active_average_notes_per_second`：普通音符数除以活跃小节时长。

总体 NPS 适合描述整首负荷；活跃 NPS 避免长静音或 break 掩盖实际演奏强度。普通 hit 直接按事件数量统计，不再使用 `16 / len(notes)` 缩放，因此同一节奏改用 24/48 格编码不会改变 NPS。两者都不能单独代表难度。

### 连续流

- `longest_note_stream_count`：相邻普通音符时间间隔不超过 0.3 秒时形成的最长连续流 hit 数。
- `longest_note_stream_seconds`：该连续流首个和末个 hit 的时间跨度。

连续流按绝对时间跨小节计算；`5`、`7`、`8` 不作为普通 hit。固定 0.3 秒阈值用于稳定比较，不表示唯一正确的演奏学定义。

### 重音覆盖

- `accent_candidate_count`：canonical `accent_grids` 与 `downbeat_grid` 去重后的候选数量。
- `accent_hit_count`：把候选的相对小节位置映射到实际 output resolution 后，存在普通音符的数量。
- `accent_coverage_rate`：命中数除以候选数；没有候选时记为 1.0。

该映射使等价节奏在 16/24/48 格下保持一致。指标只说明是否回应已检测到的重音，不判断使用咚、咔或大音符是否合适。

### 特殊音符

- 滚奏、气球及特殊音符数量；
- 各类持续时间；
- 气球目标击打数；
- 气球 hits/sec。

特殊音符负荷与普通 NPS 分开报告。

### 结构与分辨率

- `structure_density_correlation`：小节 energy percentile 与普通 NPS 的相关性。
- `peak_contrast`：peak 小节平均 NPS 与 stable/breakdown 基线之差。
- `build_up_slope_agreement`：含 build-up 角色的乐句中，phrase progress 与 NPS 的平均相关性。
- `cadence_variation_rate`：cadence pattern 与前一小节不同的比例。
- `fill_candidate_precision`：实际 fill/特殊结构中落在分析候选上的比例；没有实际 fill 时记为 1.0。
- `section_motif_consistency`：相同 `section_id` 回归时的基础 motif 一致性。
- `section_return_variation`：回归 section 在保留 motif 时产生受控变化的程度。
- `highlight_note_contrast`：peak/drop/cadence/fill 候选与普通小节的 NPS 对比。
- `base_resolution`、`resolution_change_count`、`resolution_changes_per_100_bars`、`high_resolution_bar_ratio`、`resolution_quantization_error`、`avoided_resolution_changes`：描述 ResolutionPlan 的稳定性与表达误差。
- `drum_onset_hit_coverage`：可靠鼓组瞬态所在 canonical 位置被普通音符回应的比例。
- `bass_downbeat_alignment`：位于 beat/downbeat 的可靠贝斯攻击被普通音符回应的比例。
- `vocal_phrase_response`：乐句起止或 cadence 中可靠人声攻击被回应的比例，不要求逐音节映射。
- `instrument_transition_response`：高置信主导声部/乐器切换处谱面节奏 pattern 发生响应的比例。
- `instrument_confident_bar_ratio`：存在阶段 C 活动证据的小节中，置信度不低于 0.4 的比例。
- `instrument_fill_support`：有鼓组或伴奏爆发证据的 fill candidate 中，后半小节出现普通或特殊活动的比例。

这些指标当前只写入报告，不参与 AI repair、CI 统一总分或阻断输出。阈值仍需结合更多真实歌曲和人工游玩校准。

## 可重复比较流程

1. 在临时目录或确认允许覆盖后重建 fixture：`python tests/fixtures/audio/rebuild_click_fixtures.py`。
2. 运行 `python tools/benchmark_audio_fixtures.py`，检查 `audio-alignment-v2` 的 17-fixture JSON/Markdown 报告。
3. 运行 `python tools/benchmark_chart_alignment.py`，检查 `chart-alignment-v1` 的 68-chart JSON/Markdown 报告。
4. 运行 `python tools/verify_phase_zero.py`，确认 fixture 双重建、禁网双 benchmark、稳定性和隐私检查全部通过。
5. 运行 `python tools/verify_phase_one.py`，确认 `rhythmic-salience-v1` canonical grid、确定性、首尾静音、onset 峰值对齐、confidence/reason 和稀疏输出验收通过。
6. 对同一 WAV 使用相同分析参数生成 `spectral-v1`、canonical `BarFeature`、`structure-v1` 和 `ResolutionPlan`，确认频谱状态为 `complete` 或存在匹配的 fallback notice。
7. 对 straight/triplet/mixed 确认 16/24/48 选择；对乐句混合 fixture 确认 `phrase-stable-v2` 只在真实边界切换。
8. 对 `structure_build_up_120.wav` 确认 build-up、peak、drop，并确认没有固定第 4/8 小节切分。
9. 固定 course、level、style、density 和特殊音符开关；生成两次，确认 `ChartBar` 和 `QualityReport` 完全一致。
10. 比较四难度普通音符总量、总体/活跃 NPS、峰值、连续流、重音覆盖、归一化重复、结构指标、resolution 指标、谱面对齐和静音违规。
11. 需要重新评估外部参考时，运行匿名 benchmark；需要更新 AI 示例时，再离线重建 `reference_windows.json` 并检查其中无标题、WAVE、绝对路径或完整 notes。
12. 使用版本控制中的历史版本运行同一流程，得到修改前后证据；不要把跨平台 baseline 数值差异直接视为回归。
13. 完成自动检查后执行 Web 播放和人工试听清单，并确认 `GenerationNotice` 与实际降级一致。

## CI 硬门槛

- 17 个 WAV、事件 JSON 和 ground truth schema 可以从同一组 `FixtureSpec` 确定性重建，文件集合与引用一一对应。
- `audio-alignment-v2`、`chart-alignment-v1`、`phase-zero-acceptance-v1` 和 `phase-one-acceptance-v1` 的 schema、版本、fixture/chart 覆盖范围和指标边界合法；跨环境不要求 baseline 数值逐值相等。
- Phase 1 验收中 salience 点全部位于 canonical grid，禁网双运行一致，首尾静音 0 违规，onset peak precision ≥ 0.98、recall ≥ 0.95，稀疏点比例 ≤ 0.50。
- sparse/dense、straight/triplet/mixed 与 structure build-up 真实音频流水线可运行，常规 fixture 的 `spectral-v1` 状态为 `complete` 且能映射非空 spectral flux。
- straight/triplet/mixed 稳定选择 16/24/48；乐句级混合节奏只在高置信边界切换，不逐小节抖动。
- 结构 fixture 检测到 build-up、peak、drop，且不机械使用第 4/8 小节边界。
- 同一输入重复生成结果一致。
- silent/rest 不出现违规音符。
- 内置 Easy 3、Normal 5、Hard 7、Oni 10 的普通负荷严格递增。
- dense 样本中 Oni 的活跃 NPS和最长连续流高于 Hard。
- sparse 样本的低 onset 小节保持严格上限。
- 重音覆盖率不低于宽松阈值。
- 最终谱面通过结构化 preflight 并可写入 TJA。

## 人工试听清单

- [ ] 判定线与音乐拍点对齐。
- [ ] 静音和 break 保留呼吸，没有为满足 NPS 强行补点。
- [ ] Easy/Normal 可读且不过密。
- [ ] Hard/Oni 在活跃段有明显但不过载的梯度。
- [ ] 连续流跟随持续节奏，而不是无依据铺格。
- [ ] 重拍和乐句收束有可感知回应。
- [ ] fill、滚奏和气球位于合理位置。
- [ ] 咚咔配色和换手体感可接受。
- [ ] 重复 pattern 是有意主题复现，而非机械复制。

人工检查应记录生成参数、播放环境和主观备注，但不转换为统一加权总分。

## 已知局限

- 5 组匿名参考只能提供初步数量级，不能覆盖全部曲风、BPM 和星级。
- `phrase-stable-v2` 依赖 `structure-v1` 边界和可靠 onset；弱 onset、误检或错误 section 聚类仍可能导致升级缺失或偏高，切换上限触发时会保守回退为全曲稳定高 resolution。
- TJA 参考统计对复杂变拍、跨小节持续音符和未知音符采用保守处理。
- 平均 NPS 不表达换手、复合节奏、视觉密度、大音符或局部爆发的完整难度。
- 连续流阈值是工程定义，不等同于官方难度算法。
- onset、activity、accent 和结构边界误检会传递到生成及质量指标。
- `spectral-v1` 的 HPSS、频带边界和 per-song 归一化是工程近似；密集混音、弱低频、无明确和声或极短音频可能得到不稳定语义，fallback notice 只表示本次频谱增强未使用，不表示基础节拍分析失败。
- `instrument-v1` 会受到 Demucs stem 泄漏、分离 artifact、AST AudioSet 标签粒度和硬件差异影响；其“人声/乐器”结果是概率性编谱提示，不是音乐转录或乐器鉴定。
- 当前结构、频谱和人声/乐器指标只用项目合成 fixture、mock 模型和少量匿名参考做方向性验证，尚未校准为阻断阈值。
- 自动指标无法判断谱面的趣味性、手感和音乐叙事，仍需人工游玩与试听。
