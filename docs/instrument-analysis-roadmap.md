# 节奏优先的音频分析升级路线图

日期：2026-07-15

## 1. 文档目的

本文重新定义 `tja-ai-chartgen` 的音频分析升级方向。近期工作的核心目标不是识别歌曲中具体使用了吉他、钢琴、弦乐或管乐，而是：

> 让谱面 note 的时间、密度、重音、咚咔倾向和结构变化更可靠地对齐音频。

路线图采用“节奏优先、声部可选、具体乐器暂缓”的策略：

1. 默认路径继续依赖 librosa onset/RMS、`spectral-v1`、HPSS、`structure-v1` 和 `ResolutionPlan`。
2. 建立统一、可解释的节奏显著性特征，融合 beat、downbeat、瞬态、频带攻击、持续活动和结构信息。
3. 优先改进普通 note、重音、咚咔和 fill 与音频证据的时间对齐。
4. BeatNet 继续作为可选分析器，但必须与 librosa/onset-grid 进行证据仲裁，而不是输出合法就直接采用。
5. `htdemucs` 仅作为可选的粗粒度声部增强，提供 vocals、drums、bass、other 的活动与 onset。
6. AST 和具体乐器 taxonomy 不再属于近期主线；已有能力保持兼容，后续可根据明确收益决定保留、移除或重新研究。
7. 不要求维护者自行标注真实歌曲数据；评测以程序生成的已知事件 fixture、自动指标和最终人工试听为主。

本文是专题路线图，不直接改变当前运行时行为。每个阶段都应拆成独立设计、实现计划和提交。

### 状态维护规范

所有可拆分目标都通过状态表跟踪。开发过程中直接更新对应行的“状态”“完成详情”和“验证记录”。

状态字段统一使用：

- `未开始`：尚未进入设计或实现。
- `设计中`：正在确认边界、数据模型或验收标准。
- `开发中`：已经开始编写测试或实现。
- `部分完成`：已有可复用基础或部分子项完成，但尚未满足全部验收条件。
- `验证中`：实现已完成，正在运行 benchmark、回归或人工试听。
- `已完成`：实现、验证和必要文档均已完成。
- `已阻塞`：存在依赖、模型、平台或评测问题。
- `已取消`：评测证明收益不足或不再符合项目目标。

状态改为 `已完成` 时，必须同时记录实际完成内容、关键决策、执行过的测试以及结果。不得只修改状态而保留“尚未实施”。

---

## 2. 为什么改为节奏优先

### 2.1 太鼓谱面真正需要的信息

普通 note 是否对齐音乐，主要取决于：

- BPM 与拍号；
- 第一小节 downbeat；
- beat、弱拍和细分位置；
- 鼓点、拨弦、钢琴攻击、人声重读等瞬态；
- 低、中、高频攻击；
- harmonic/percussive 平衡；
- 持续活动和静音；
- build-up、peak、drop、cadence、breakdown 和 fill；
- 当前分辨率能否表达检测到的节奏位置。

这些信息不要求准确知道声音来自吉他、钢琴还是弦乐。

### 2.2 具体乐器识别的收益有限

具体乐器分类可能帮助 AI 设计不同 motif 或理解配器叙事，但它不是 note 时间对齐的基础条件。当前 AST 路线还存在：

- 通用 AudioSet 目标与音乐乐器识别不完全一致；
- 4 秒窗口、2 秒 hop 对小节变化过粗；
- `other` stem 中多个乐器、分离泄漏和 artifact 混合；
- taxonomy 映射与阈值需要额外校准；
- 增加模型下载、推理成本和 AI 上下文；
- 如果要可靠比较模型，需要额外乐器标注或公开数据 benchmark。

在谱面基础节奏仍有提升空间时，继续投入精细乐器分类的优先级较低。

### 2.3 不自行标注真实歌曲

本路线图不要求维护者为真实歌曲逐小节标注乐器、进入/退出或 downbeat 数据集。

真实歌曲只用于最终试听和游玩抽查，不建立必须长期维护的人工 ground truth。自动回归主要依赖：

- 程序生成且事件时间已知的音频；
- 现有许可明确的 fixture；
- 音频事件与谱面事件之间可自动计算的对齐指标；
- 同一输入修改前后的确定性比较。

---

## 3. 当前基线

### 3.1 默认音频分析

当前默认路径已经包含：

- librosa onset strength 与 onset time；
- RMS activity 与原始 RMS；
- librosa beat tracking；
- onset-grid BPM/OFFSET 重估与置信度拒绝；
- `spectral-v1` 的 HPSS；
- 低/中/高频 onset；
- spectral flux；
- brightness；
- harmonic novelty；
- texture novelty；
- percussive ratio；
- canonical `BarFeature`；
- `structure-v1`；
- 12/18/36 与 16/24/48 `ResolutionPlan`。

这些能力已经覆盖节奏优先路线的大部分原始证据，但目前仍分散在不同字段和消费者中，缺少统一的“某个格点多值得放 note”表达。

### 3.2 可选 BeatNet

BeatNet 当前使用：

```python
model=1
mode="offline"
inference_model="DBN"
```

它提供 beat number、downbeat 和 meter，但当前只要输出结构合法，就会进入新的 BPM/OFFSET 基线。onset-grid 可以二次校正，但没有完整比较 BeatNet 与原始 librosa/onset-grid 哪一个更符合音频。

### 3.3 可选 instrument-v1

当前 `instrument-v1` 使用：

```text
htdemucs → vocals / drums / bass / other
AST AudioSet → mix / other 乐器 taxonomy
```

其中真正与节奏直接相关的是：

- vocal activity/onset；
- drum activity/onset；
- bass activity/onset；
- accompaniment activity/onset。

具体乐器 taxonomy 继续兼容现有 JSON、AI payload 和质量报告，但不再作为近期升级主线。

---

## 4. 目标与非目标

### 4.1 目标

1. 建立统一、可解释、可测试的节奏显著性表示。
2. 提高普通 note 与可靠音频瞬态的对齐率。
3. 降低没有 beat、onset、activity 或结构依据的无证据落点。
4. 保证静音、break 和 breakdown 不因高难度被无依据铺点。
5. 改进 downbeat、强拍、fill burst 和 cadence 的谱面响应。
6. 使用频带和 percussive 证据提供咚咔软倾向，而不是硬映射。
7. 让 BeatNet 只在证据更可靠时改善 BPM、OFFSET、meter 或 downbeat。
8. 保留可选 `htdemucs` stem activity/onset 增强，但不依赖具体乐器分类。
9. 不要求人工维护真实歌曲标注集。
10. 保持所有自动分析结果可追踪、可降级和可复现。

### 4.2 非目标

近期不实现：

- 吉他、钢琴、弦乐、铜管、木管等精细乐器分类升级；
- EfficientAT/DyMN、OpenMIC 或其他乐器分类器微调；
- 自建真实歌曲乐器标注数据集；
- 歌手身份、歌词、音素或语言识别；
- 多乐器 MIDI 转录；
- 鼓件级完整转录；
- 精确音高与和弦转录；
- 把每个音频 onset 机械转换成 note；
- 让任何声学特征突破静音、density、NPS、occupancy、resolution 和 TJA preflight 硬约束；
- 在本路线图中实现变速 tempo map 或自动 `#BPMCHANGE`；
- 仅凭一个统一加权分数替代人工试听和游玩。

---

## 5. 目标架构

### 5.1 默认路径

```text
输入音频
  ↓
librosa onset / RMS / activity
  ↓
spectral-v1
  ├─ HPSS percussive/harmonic
  ├─ low/mid/high onset
  ├─ spectral flux
  ├─ brightness/novelty
  └─ percussive ratio
  ↓
librosa / onset-grid / 可选 BeatNet 候选仲裁
  ↓
canonical BarFeature
  ↓
RhythmicSalience
  ├─ hit salience
  ├─ accent salience
  ├─ don/ka preference
  ├─ sustained activity
  └─ confidence/reason
  ↓
structure-v1 / ResolutionPlan
  ↓
规则 fallback 或 AI
  ↓
谱面对齐 QualityReport
```

### 5.2 可选声部增强

```text
htdemucs
  ├─ vocals activity/onset
  ├─ drums activity/onset
  ├─ bass activity/onset
  └─ other activity/onset
        ↓
作为 RhythmicSalience 的附加证据
```

可选声部增强不改变默认路径的可用性。模型缺失、设备不可用或推理失败时继续使用基础 onset、spectral 和 beat 证据。

### 5.3 具体乐器识别的位置

```text
AST / EfficientAT / DyMN / OpenMIC / MERT / CLAP
        ↓
远期可选研究
```

具体乐器分类不得成为默认节奏对齐流水线的依赖。

---

# Phase 0：建立无人工标注的自动对齐基线

**优先级：P1**
**预计工作量：M–L**
**依赖：无**

## 6. 阶段目标

通过程序生成、事件时间已知的音频建立自动 ground truth，使后续 onset、BeatNet、salience 和谱面落点修改都能量化比较，不需要人工标注真实歌曲。

### Phase 0 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 6.1 盘点并复用现有 fixture | 已完成 | 已盘点 9 个由标准库脚本确定性生成的 WAV，明确其节拍、密度、resolution、结构和边缘静音复用范围，并记录后续缺口。 | 2026-07-15：`pytest tests/test_audio_pipeline_integration.py -v`，10 passed。 |
| 6.2 新增节奏事件 ground truth | 已完成 | 已建立 schema version 1，为当前 17 个 WAV 同步生成事件 JSON，持久化 BPM、拍号、时长、onset、强 onset、beat、downbeat、静音区、频带事件、fill 区间和结构段落。 | 2026-07-15：`pytest tests/test_audio_fixture_ground_truth.py -v`；确定性重建后的 WAV、schema 和事件 JSON 与仓库文件逐字节一致。 |
| 6.3 扩展合成节奏类型 | 已完成 | 新增 8 个 WAV，覆盖切分、真正弱起、低频主拍/高频反拍、持续 harmonic 背景、短 fill burst、3/4、6/8 和半速/倍速歧义；fixture 总数增至 17。 | 2026-07-15：`pytest tests/test_audio_fixture_ground_truth.py -v`，4 passed；完整重建逐字节一致。 |
| 6.4 建立音频分析 benchmark | 已完成 | `audio-alignment-v2` 已覆盖全部 17 个 fixture，补齐频带、弱起、fill 与 resolution 指标，并同时生成机器可读 JSON baseline 和逐 fixture Markdown 报告。 | 2026-07-15：`python tools/benchmark_audio_fixtures.py`；`pytest tests/test_audio_benchmark.py -v`，9 passed。 |
| 6.5 建立谱面对齐 baseline | 已完成 | `chart-alignment-v1` 已对全部 17 个 fixture 的 Easy 3、Normal 5、Hard 7、Oni 10 规则谱面计算 note/onset、强 onset、downbeat、无证据 note、静音违规、fill 响应和确定性指标，并生成 JSON/Markdown 基线。 | 2026-07-15：`python tools/benchmark_chart_alignment.py`；`pytest tests/test_chart_alignment_benchmark.py -v`，7 passed。 |
| 6.6 阶段退出条件 | 已完成 | `phase-zero-acceptance-v1` 已验证 fixture/ground truth 双重建、离线 benchmark、两类 baseline 完整性、同环境双运行稳定性和持久化隐私，全部退出条件通过。 | 2026-07-15：`python tools/verify_phase_zero.py`，PASS；`pytest tests/test_phase_zero_acceptance.py -v`，7 passed。 |

## 6.1 现有 fixture 盘点与复用结论

6.1 盘点时已有的 9 个 WAV 全部由 `tests/fixtures/audio/rebuild_click_fixtures.py` 仅使用 Python 标准库确定性生成，可继续作为 Phase 0 基线，不需要引入真实歌曲或第三方录音。

| Fixture | 已知模式 | 当前覆盖 | Phase 0 复用方向 |
| --- | --- | --- | --- |
| `click_4_4.wav` | 120 BPM、4/4、16 个四分音符，首拍位于 0.25 秒 | ffmpeg 转换、BPM/offset、onset-grid、spectral、BarFeature、TJA 写入 | 基础 onset、beat、downbeat 与首拍误差基线 |
| `click_4_4_leadin.wav` | 与基础点击轨相同，但首个实际事件位于 1.25 秒 | 带前导静音的 BPM/相位稳定性 | 前导静音与实际 first downbeat 的独立校验；不得把它误作弱起 fixture |
| `sparse_120.wav` | 120 BPM、8 小节；第 3、5 小节仅保留 downbeat，其余为四分音符 | 四难度确定性、稀疏段密度上限、accent、静音违规 | sparse/rest、谱面负载和无证据 note 基线 |
| `dense_180.wav` | 180 BPM、8 小节；八分音符为主，第 3、7 小节为十六分音符，第 5 小节仅保留 downbeat | 四难度密度递增、NPS、连续流 | dense/flow、强弱密度对比和谱面响应基线 |
| `transient_noise_intro_120.wav` | 2.25 秒后进入 120 BPM 点击轨，前部含两个低电平瞬态 | 首部连续静音、低电平毛刺、density hint 和静音区零 note | 静音区 onset 误检与违规 note 基线 |
| `straight_120.wav` | 120 BPM、4 小节、每拍四等分 | 16 格 resolution 稳定选择 | straight subdivision 和量化误差基线 |
| `triplet_120.wav` | 120 BPM、4 小节、每拍三等分 | 24 格 resolution 稳定选择 | triplet subdivision 和量化误差基线 |
| `mixed_120.wav` | 120 BPM、4 小节，直拍与三连音逐小节交替 | 48 格 resolution 稳定选择 | mixed subdivision、可表达事件比例和高 resolution 必要性基线 |
| `structure_build_up_120.wav` | 120 BPM、12 小节，细分由稀到密后骤降 | `build_up`、`peak`、`drop` 与 phrase 边界 | 结构密度趋势、高潮对比和 drop 响应基线 |

现有复用入口集中在 `tests/test_audio_pipeline_integration.py`：基础点击轨覆盖完整真实音频流水线，sparse/dense 覆盖四难度规则生成与 QualityReport，straight/triplet/mixed 覆盖 ResolutionPlan，structure fixture 覆盖结构角色，transient fixture 覆盖首部静音保护。`tests/test_resolution.py` 中的内存事件 fixture 继续补充 3/4、6/8 和 phrase-stable resolution 单元测试，但它们不是可用于真实音频 benchmark 的 WAV。

6.1 盘点确认的持久化 ground truth 缺口已由 6.2 补齐；切分、真正弱起、低频主拍/高频反拍、持续 harmonic 背景、明确 fill burst、3/4 WAV、6/8 WAV 和半速/倍速歧义模式已由 6.3 补齐。当前共有 17 个可确定性重建的 WAV。

## 6.2 Fixture ground truth 格式

正式格式使用 `schema_version=1`，由 `tests/fixtures/audio/ground_truth.schema.json` 以 JSON Schema Draft 2020-12 定义。每个 `<stem>.wav` 对应同目录的 `<stem>.events.json`：

```json
{
  "schema_version": 1,
  "audio": "click_4_4.wav",
  "bpm": 120.0,
  "time_signature": "4/4",
  "duration": 8.25,
  "first_downbeat": 0.25,
  "onsets": [0.25, 0.75, 1.25, 1.75],
  "strong_onsets": [0.25],
  "beats": [0.25, 0.75, 1.25, 1.75],
  "downbeats": [0.25],
  "low_band_onsets": [],
  "high_band_onsets": [],
  "silent_ranges": [[0.0, 0.25]],
  "fill_ranges": [],
  "sections": []
}
```

时间字段统一使用相对 WAV 起点的秒数。`onsets` 表示实际合成瞬态，`beats` 和 `downbeats` 表示理论节拍网格，二者不得因某一拍没有声音而混用；`strong_onsets` 记录合成器明确加重的事件，通常包含 downbeat，也可包含 6/8 的第二个复拍或半速/倍速歧义 fixture 的半小节重音。尚无对应证据的频带事件和 fill 区间必须写为空数组，不使用推测值。`transient_noise_intro_120` 的低电平毛刺仍位于语义静音区内，供误检指标使用；`structure_build_up_120` 额外记录 stable、build_up、peak 和 drop 段落。

`rebuild_click_fixtures.py` 现在从同一组 `FixtureSpec` 同时生成 WAV 和事件 JSON，默认完整重建；`--ground-truth-only` 只刷新 schema 与事件文件，`--output-dir` 可在临时目录执行无副作用的确定性校验，重复使用 `--fixture <name.wav>` 可只生成指定 fixture。`tests/test_audio_fixture_ground_truth.py` 检查字段契约、时间范围、事件包含关系、音频引用，并在临时目录完整重建后逐字节比较全部 WAV、schema 和事件 JSON，防止音频与标注漂移。

## 6.3 新增 fixture 类型

已覆盖：

- [x] 主拍四分音符、八分音符和十六分音符；
- [x] 三连音，以及直拍与三连音混合；
- [x] 切分音：`syncopated_120.wav`；
- [x] 真正弱起：`pickup_120.wav` 的首个 onset 早于 first downbeat；
- [x] 低频主拍与高频反拍：`band_attacks_120.wav`，ground truth 分别写入 `low_band_onsets` 和 `high_band_onsets`；
- [x] 持续 harmonic 背景加稀疏 percussive onset：`harmonic_sparse_120.wav`；
- [x] 短 fill burst：`fill_burst_120.wav`，两个末拍 burst 写入 `fill_ranges`；
- [x] 静音和低电平毛刺：`transient_noise_intro_120.wav`；
- [x] build-up → peak → drop：`structure_build_up_120.wav`；
- [x] 3/4：`meter_3_4_120.wav`；
- [x] 6/8：`meter_6_8_120.wav`，每小节 6 个八分 onset、3 个四分音符 BPM 基准 beat，并标记两个复拍重音；
- [x] 半速/倍速容易混淆的模式：`tempo_ambiguity_120.wav`，半小节重音明显、四分音符间拍较弱。

频带 fixture 使用可配置频率和振幅的短衰减点击；harmonic fixture 在不改变已知 percussive onset 的前提下叠加带淡入淡出的 220 Hz/330 Hz 持续音。旧 fixture 继续按原参数生成，完整重建测试同时比较 17 个 WAV 和事件 JSON，确认扩展生成器没有改变既有音频字节。

## 6.4 自动指标

完整实现为 `audio-alignment-v2`：`tools/benchmark_audio_fixtures.py` 离线读取全部 `<stem>.events.json`，调用默认 `analyze_audio()`，再由 `evaluation/audio_benchmark.py` 做一对一事件匹配、ResolutionPlan 评测与跨 fixture 聚合。默认容差为 onset 50 ms、beat 70 ms、downbeat 70 ms，频带峰值阈值为 0.25；均可通过 CLI 参数覆盖，并可显式启用 `--use-beatnet`。默认路径不访问网络。

机器可读结果写入 `tests/fixtures/audio/audio_benchmark_baseline.json`，schema version 2 除基础 onset/beat/downbeat、BPM、拍号、first downbeat 和静音误检外，还记录：

- 仅对带标注 fixture 生效的 low/high band onset、pickup onset 和 fill onset precision/recall/F1；
- 每个 fixture 的 spectral 状态及 downbeat 来源；
- ResolutionPlan policy、base/bar resolution、切换次数；
- ground-truth onset 到目标 resolution 的平均/最大量化误差和可表达事件比例；
- under-resolved bar、不必要高 resolution bar、高 resolution 占比；
- 实际 resolution 分布与 ground truth 所需最小 resolution 分布；
- first downbeat 之前无法归入完整小节的 pickup 事件数量。

同一次命令还会原子写入 `tests/fixtures/audio/audio_benchmark_baseline.md`，提供聚合表和 17 个 fixture 的逐项对照，便于无需手工解析 JSON 即可进行修改前后 A/B。默认分析没有独立 downbeat 输出时，benchmark 按估计 offset、四分音符 BPM 和拍号推导并标记为 `derived-offset-meter`；BeatNet 提供结果时标记为 `analyzer`。

当前 17-fixture baseline 的 onset/beat/downbeat F1 分别为 0.996421、0.805740、0.418033；low/high band onset F1 均为 0.666667，pickup recall 和 fill recall 均为 1.0。ResolutionPlan 对 ground-truth 事件的可表达比例为 1.0、量化误差为 0，但 114 个已评测小节全部选择 48 格且都高于 ground truth 所需最小 resolution，明确暴露出真实检测 onset 抖动导致的过度升级基线。该结果只用于修改前后 A/B 和回归趋势，不作为跨 librosa、BeatNet 或平台版本必须逐值相等的 CI 阈值；CI 校验 schema、fixture 完整性、指标范围和 Markdown 报告覆盖。

### Onset

- [x] precision、recall 和 F1；
- [x] 平均/最大时间误差；
- [x] 强 onset recall；
- [x] 静音区误检数量；
- [x] 有标注 fixture 的 low/high band、pickup 和 fill onset 指标。

### Beat/downbeat

- [x] BPM 绝对/相对误差；
- [x] 半速/倍速错误；
- [x] beat/downbeat F-measure；
- [x] first downbeat 误差；
- [x] meter accuracy；
- [x] analyzer 与 derived downbeat 来源区分。

### Resolution

- [x] ground-truth onset 到目标 resolution 的量化误差；
- [x] 可表达事件比例；
- [x] under-resolved 与不必要高 resolution 小节；
- [x] high resolution 比例和 resolution 切换次数；
- [x] selected 与 expected minimum resolution 分布。

### 谱面

- [x] note/onset 对齐率；
- [x] 强 onset 响应率；
- [x] downbeat 响应率；
- [x] 无证据 note 比例；
- [x] 静音区违规 note；
- [x] fill burst 响应率；
- [x] 相同输入生成确定性。

完整实现为 `chart-alignment-v1`：`tools/benchmark_chart_alignment.py` 对每个 fixture 只执行一次默认音频分析和结构/resolution 构建，再用固定 `technical` / `auto` / 不启用特殊音符的规则配置分别生成 Easy 3、Normal 5、Hard 7 和 Oni 10，共 68 张谱面。每张谱面连续生成两次并比较完整 `ChartBar`，因此确定性检查同时覆盖落点、咚咔配色和气球计数等结构化输出，而不只是 note 时间。

`evaluation/chart_alignment.py` 将普通 note 和长音起点按对应 `BarFeature` 的实际起止时间与输出 resolution 映射回音频秒数。默认使用 50 ms 判断 note/onset、强 onset 和 fill onset 对齐，使用 70 ms 判断 downbeat 与理论 beat 证据；无证据 note 指既不接近真实 onset、也不接近理论 beat 的 note。静音区采用左闭右开范围，避免区间终点与首个合法事件重复计数。机器可读基线写入 `tests/fixtures/audio/chart_alignment_baseline.json`，逐 course 和逐谱面对照写入 `chart_alignment_baseline.md`。

当前基线共评测 3,358 个 note：note/onset precision 0.626563、onset recall 0.752504、强 onset response 0.923077、downbeat response 0.934211、fill onset response 0.656250、无证据 note 比例 0.059857、静音区违规 65 个，68 张谱面的重复生成完全一致。按难度观察，Easy 的 note/onset precision 为 0.941667，Oni 为 0.495811；高难度为保持可玩负载会加入更多由 beat、activity 或结构支持而非真实瞬态直接支持的落点。该结果是修改前基线，不直接作为质量门槛。

## 6.5 真实歌曲的角色

真实歌曲不建立强制人工标注。仅用于：

- 修改前后 A/B 试听；
- Web 播放检查；
- 观察 `analysis.json` 和 QualityReport；
- 发现合成 fixture 未覆盖的问题；
- 必要时新增能够程序化重现的 fixture。

不得把“听起来更好”直接升级为 CI 门槛。

## 6.6 退出条件

- [x] fixture 与 ground truth 可以由同一脚本确定性重建；
- [x] benchmark 不访问网络；
- [x] onset、beat、downbeat、resolution 和谱面对齐都有修改前 baseline；
- [x] 同一环境重复运行结果稳定；
- [x] 不需要真实歌曲标题、路径、原始音频或人工标签进入仓库。

阶段验收入口为 `python tools/verify_phase_zero.py`。`phase-zero-acceptance-v1` 会在两个独立临时目录中完整重建 fixture，将 17 个 WAV、17 个事件 JSON 和 ground truth schema 共 35 个产物逐字节与仓库版本比较；随后在屏蔽 `socket.connect` / `connect_ex` 的上下文中分别执行两次 `audio-alignment-v2` 和 `chart-alignment-v1`，确认 benchmark 不依赖网络且同一环境输出完全一致。

验收还会检查两个已提交 baseline 的 schema、版本、ground-truth schema、17-fixture/68-chart 覆盖范围和默认关闭 BeatNet 的离线路径，并递归扫描持久化 JSON，拒绝标题、艺术家、API key、base URL、输入/输出绝对路径、URL 和未知音频引用。机器可读结果写入 `tests/fixtures/audio/phase_zero_acceptance.json`，人类可读结果写入 `phase_zero_acceptance.md`。

2026-07-15 的正式验收结果为 **PASS**：35 个产物在两次独立重建中均无差异；两类 benchmark 各运行两次且结果稳定，并与当前提交 baseline 一致；531 个持久化字符串中没有发现真实歌曲身份、绝对路径、URL、服务凭证或未知音频引用。Phase 0 的五项退出条件全部满足，可以进入 Phase 1。

## Phase 0 任务划分列表

- [x] 6.1 盘点并复用现有 fixture
- [x] 6.2 建立 ground truth schema，并让 fixture 同步生成事件 JSON
- [x] 6.4a 建立基础 onset、beat 与 downbeat benchmark
- [x] 6.3 扩展切分、弱起、频带攻击、fill、3/4 和 6/8 fixture
- [x] 6.4b 完善全部 fixture 的自动 benchmark 与基线报告
- [x] 6.5 建立谱面对齐 baseline
- [x] 6.6 执行 Phase 0 阶段验收

---

# Phase 1：建立统一节奏显著性

**优先级：P1**
**预计工作量：L**
**依赖：Phase 0**

## 7. 阶段目标

把分散的 onset、activity、spectral、beat 和结构证据融合成稳定的 `RhythmicSalience`，让规则生成、AI prompt 和质量报告使用同一套可解释信号。

### Phase 1 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 7.1 定义 RhythmicSalience 数据模型 | 已完成 | 已定义 `RhythmicSaliencePoint`、`BarRhythmicSalience` 及 `rhythmic-salience-v1` 版本常量；字段具有稳定默认值和 0–1/非负边界，暂不改变现有 `analysis.json` schema。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖版本、序列化、默认值隔离和边界校验。 |
| 7.2 统一 canonical 时间轴 | 已完成 | `build_canonical_rhythmic_evidence()` 现在把 onset/strength、activity、beat/downbeat 和稀疏 spectral attack/flux 合并到完整 canonical grid；兼容旧 `BarFeature` 字段与显式 `GridFeature`，重复频谱证据取最大值，越界证据稳定忽略。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖 48 格 4/4、36 格 6/8、旧字段回退、显式证据优先、重复/越界处理和活动值门控。 |
| 7.3 定义 hit salience | 已完成 | `build_hit_salience()` / `build_bar_hit_salience()` 已融合 onset strength、局部 spectral 峰值、activity 门控的 beat/downbeat 骨架和 structure role 修正；强瞬态优先，activity 不会单独制造 hit，重复派生证据按最大值合并，数字静音与首尾静音强制归零。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖 off-beat onset 优先级、beat 骨架、activity-only 留白、邻近 spectral 合并、连续 onset 保留、结构修正、静音归零和确定性。 |
| 7.4 定义 accent salience | 已完成 | `build_accent_salience()` / `build_bar_accent_salience()` 已在既有 hit 候选上融合 downbeat、onset 局部峰值、低频攻击、旧 accent hint、歌曲/段落/乐句起点、正向 energy delta 与 peak/cadence/fill 等结构修正；accent 不会独立制造 hit，也不直接决定大音符。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖 downbeat、onset 峰值、低频攻击、非低频 spectral 留白、section start、energy rise、cadence 修正、activity/structure-only 留白及 activity strength 不误判 onset。 |
| 7.5 定义 don/ka 软倾向 | 已完成 | `build_don_ka_salience()` / `build_bar_don_ka_salience()` 已在既有 hit/accent 点上统一生成 don/ka preference：低频主导与 downbeat 弱偏咚，高频主导、percussive brightness 与反拍弱偏咔；中频主导或频带混合保持中性，所有倾向设上限并对连续强单色提示做确定性软化，最终配色仍由 style 和生成器决定。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖低/高频主导、中频/混合中性、downbeat/offbeat、6/8 复拍子反拍、brightness 门控和连续单色平衡。 |
| 7.6 定义置信度和 reason | 已完成 | hit salience 现使用 onset、spectral 和 beat drive 的绝对门控，避免微弱噪声仅因归一化后非零而形成候选；逐点置信度融合瞬态强度、全曲 75th percentile 参考、局部峰值差距、activity 支持、多证据一致性与节拍骨架可靠度，小节置信度再汇总有效事件数量、瞬态占比和活动覆盖。低于可用阈值或只有 beat 骨架时保留稀疏诊断但输出稳定 fallback reason，首尾静音和普通静音也使用固定 reason code。 | `pytest tests/test_rhythmic_salience.py -v` 覆盖绝对门控、多证据置信度提升、低置信瞬态、beat-only fallback、静音 reason 传播和确定性。 |
| 7.7 阶段退出条件 | 已完成 | 新增 `phase-one-acceptance-v1` 离线验收：对全部 17 个合成 fixture 在禁用 socket 连接时独立运行两次完整音频分析、结构分析与 hit/accent/don-ka salience，验证 canonical grid、确定性、首尾静音、onset 峰值对齐、confidence/reason 契约和稀疏输出。Phase 1 内部阶段共享同一 salience 流水线；fallback、AI prompt 与质量报告接入仍按 Phase 2–4 明确后移，不伪装为本阶段已完成。 | 2026-07-15：`python tools/verify_phase_one.py`，PASS；17 fixtures，onset peak precision 0.998555、recall 0.988555、F1 0.993530，最大误差 52.44 ms，2 次离线运行完全一致，10 个首尾静音小节 0 违规，1028/6240 稀疏点；`pytest tests/test_phase_one_acceptance.py -v`，7 passed。 |

## 7.1 建议数据模型

示意结构：

```python
class RhythmicSaliencePoint(BaseModel):
    grid: int
    hit: float = 0.0
    accent: float = 0.0
    don_preference: float = 0.0
    ka_preference: float = 0.0
    sustained_activity: float = 0.0
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class BarRhythmicSalience(BaseModel):
    points: list[RhythmicSaliencePoint] = Field(default_factory=list)
    active_ratio: float = 0.0
    onset_evidence_count: int = 0
    confidence: float = 0.0
    fallback_reason: str | None = None
```

最终字段应保持紧凑，`reasons` 可以在持久化诊断中使用短代码，在 AI payload 中不重复发送长字符串。

## 7.2 Hit salience 证据

建议融合：

- mix onset strength；
- HPSS percussive onset；
- spectral flux；
- low/mid/high band onset；
- beat/downbeat；
- activity gate；
- structure role；
- 可选 drum/bass/accompaniment onset。

原则：

- 强瞬态优先；
- beat 是无瞬态时的骨架，不应压过明显 off-beat onset；
- 持续 activity 不能单独制造大量离散 hit；
- 数字静音和首尾静音强制归零；
- 多个相近 frame 应合并为一个局部峰值；
- 同一证据不能通过多个派生字段被重复累计。

## 7.3 Accent salience 证据

建议使用：

- downbeat；
- onset strength 局部峰值；
- 低频攻击；
- drum onset；
- section/phrase 起点；
- peak/cadence/fill；
- 前后窗口能量差。

accent 只表示“值得强调”，不直接决定使用 `3` 或 `4` 大音符。最终仍受 course、level、style 和大音符上限约束。

## 7.4 咚咔软倾向

- 低频攻击、bass onset、downbeat：弱偏向咚；
- 高频攻击、percussive brightness、反拍：弱偏向咔；
- 中频或证据混合：保持中性；
- style 模板继续控制整体配色习惯；
- 不允许按频率硬编码所有 note 颜色；
- 必须避免长时间单色串。

## 7.5 归一化和置信度

避免只做全曲 min-max 后把微弱噪声放大。建议同时使用：

- 绝对活动门控；
- 全曲 percentile；
- 小节局部峰值；
- 多证据一致性；
- 静音与低持续比例；
- 有效事件数量；
- 与次佳局部峰值的差距。

置信度用于决定是否使用 salience，不作为谱面难度分数。

## 7.6 持久化边界

`analysis.json` 可以保存：

- salience feature version；
- 每小节汇总；
- 稀疏非零格点；
- 置信度；
- 稳定 fallback reason。

不保存：

- 重复的完整 dense frame；
- 每种中间算法的全部数组；
- 无法被消费者使用的调试 tensor；
- 绝对模型路径。

## 7.7 退出条件

- 所有 salience 点来自 canonical grid；
- 同一输入结果确定；
- 静音区全零；
- 已知 onset fixture 上 hit salience 峰值位置正确；
- Phase 1 内 hit、accent、don/ka 阶段复用同一 salience 流水线，不重复实现 onset/spectral 融合；
- fallback generator、AI prompt 和质量报告的消费迁移分别留给 Phase 2–4，不作为 Phase 1 已接入能力；
- Phase 1 不增加 AI payload 字段，后续接入时仍须保持紧凑。

阶段验收入口为 `python tools/verify_phase_one.py`。`phase-one-acceptance-v1` 在屏蔽 `socket.connect` / `connect_ex` 的上下文中对全部 17 个合成 fixture 独立执行两次 librosa/onset-grid、canonical `BarFeature`、`structure-v1` 和完整 hit/accent/don-ka salience 流水线；验收要求两次结构化结果完全一致，所有点均位于对应 canonical grid 且无重复或乱序，首尾静音小节没有 salience 点，confidence 与稳定 fallback reason 契约无冲突。

已知 fixture onset 使用 70 ms 容差与 `hit >= 0.45` 的 salience 峰值做一对一匹配，阻断门槛为 precision 不低于 0.98、recall 不低于 0.95；稀疏点数量不得超过 canonical dense grid 的 50%。机器可读结果写入 `tests/fixtures/audio/phase_one_acceptance.json`，人类可读结果写入 `phase_one_acceptance.md`。当前验收结果为 PASS：691/699 onset 匹配、precision 0.998555、recall 0.988555、最大误差 52.44 ms，10 个首尾静音小节 0 违规，稀疏点比例 0.164744。

## Phase 1 任务划分列表

- [x] 7.1 定义 `RhythmicSalience` 数据模型与版本
- [x] 7.2 统一 onset、activity、spectral、beat 的 canonical 时间轴
- [x] 7.3 定义并实现 hit salience
- [x] 7.4 定义并实现 accent salience
- [x] 7.5 定义并实现 don/ka 软倾向
- [x] 7.6 增加绝对门控、置信度和稳定 reason
- [x] 7.7 执行 Phase 1 阶段验收

---

# Phase 2：让普通 note 跟随节奏显著性

**优先级：P1**
**预计工作量：M–L**
**依赖：Phase 1**

## 8. 阶段目标

改进 fallback 与 AI 约束，使普通 note 优先落在可靠 salience 位置，同时保留难度梯度、可玩性和音乐留白。

### Phase 2 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 8.1 建立 salience 候选排序 | 已完成 | 新增 `features/salience_candidates.py`，按可靠强瞬态、可靠普通瞬态、持续 beat/downbeat 骨架、结构 highlight、弱证据建立稳定优先级，并按当前 `ResolutionPlan` 精确过滤不可表达、越界和同格重复点。 | 2026-07-15：`pytest tests/test_salience_candidates.py tests/test_rhythmic_salience.py -q`，40 passed；`pytest`，541 passed、1 deselected；`ruff check .`；`python tools/verify_phase_one.py`，PASS。 |
| 8.2 改进普通 note 选择 | 已完成 | fallback 现在为全曲一次构建逐小节 salience 候选，普通 note 在旧 style/spectral/instrument 补点前优先消费可靠强瞬态、普通瞬态、持续 beat/downbeat 骨架和结构 highlight；弱证据仍交给后续旧评分补足目标负荷。 | 2026-07-15：`pytest tests/test_fallback_generator.py -q`，38 passed；`pytest`，543 passed、1 deselected；`ruff check .`；临时 `chart-alignment-v1` benchmark 为 68/68 确定性，strong onset recall 由 0.923077 提升到 0.973077，downbeat recall 由 0.934211 提升到 0.982456，unsupported-note ratio 保持 0.059857，四难度 note 数保持 360 < 608 < 1077 < 1313；aggregate note/onset precision 从 0.626563 轻微变化为 0.626266，留待 8.3 的弱证据补点约束继续优化。 |
| 8.3 控制无证据补点 | 已完成 | fallback 将可靠 salience 之后的候选拆成有 onset/beat/activity/spectral/instrument 支持的弱证据与无证据格点，按 course 设置总弱补点和无证据子预算，连续弱 note 最多 2 个；sparse/breakdown 禁止无证据补点，纯活跃无格点证据时只保留最小 course 骨架。 | 2026-07-15：`pytest tests/test_fallback_generator.py tests/test_salience_candidates.py -q`，55 passed；`pytest`，550 passed、1 deselected；`ruff check .`；临时 `chart-alignment-v1` 为 68/68 确定性，unsupported-note ratio 从 0.059857 降至 0.045524，note/onset precision 从 0.626266 升至 0.638240，strong/downbeat recall 保持 0.973077/0.982456。 |
| 8.4 保留 course/density 约束 | 已完成 | 新增 `tests/test_phase_two_constraints.py`，以 4/4、3/4、6/8 的 16/24/48 与 12/18/36、四 course 和五档 density 组成完整矩阵，逐项验证输出长度、course/density 单调性、真实时间 NPS、legacy/实际 occupancy、跨 resolution 等价 hit 数，以及 silent/rest/sparse 硬上限。 | 2026-07-15：约束矩阵 6 passed，覆盖 360 个生成配置和 1080 个小节结果；`pytest`，556 passed、1 deselected；`ruff check .`；临时 `chart-alignment-v1` 指标与 8.3 完全一致，68/68 确定性。 |
| 8.5 接入 AI compact payload | 已完成 | AI payload 升级为 `tja-ai-chartgen-compact-v5`，逐小节发送仅含目标 resolution 可表达点的稀疏 salience，使用 0–1000 整数编码 hit/accent/don/ka/activity/confidence 和候选等级；成功 AI 结果生成 `ai-salience-validation-v1` 首轮 report-only 对齐诊断并持久化到 AI output/attempt sidecar。 | 2026-07-15：`pytest tests/test_ai_client.py tests/test_salience_candidates.py tests/test_rhythmic_salience.py tests/test_event_encoder.py -q`，107 passed；`pytest`，558 passed、1 deselected；`ruff check .`；`git diff --check`。 |
| 8.6 阶段退出条件 | 已完成 | 新增 `phase-two-acceptance-v1` 离线验收，禁网双跑 17 个 fixture/68 张规则谱面，对比 Phase 2 前 baseline，并执行 360 配置/1080 小节约束矩阵和规则/AI 共享 salience 指标契约检查。 | 2026-07-15：`python tools/verify_phase_two.py`，PASS；note/onset precision 0.626563→0.638240，unsupported ratio 0.059857→0.045524，strong/downbeat recall 提升到 0.973077/0.982456，静音违规 65→60，68/68 确定性，约束矩阵 0 违规。 |

## 8.1 候选选择原则

普通 note 候选优先级：

1. 可靠强 onset；
2. 可靠普通 onset；
3. downbeat/beat 与持续节奏骨架；
4. structure highlight；
5. style 需要的连接点；
6. 无显著证据时的最小可玩骨架。

候选仍受以下限制：

- 当前 output resolution 可表达；
- 最小事件间隔；
- course speed cap；
- occupancy cap；
- density hint；
- silent/rest/sparse 上限；
- 长音与普通 note 冲突；
- 同格点事件冲突。

当前 8.1 已提供独立候选层：`build_salience_candidate_bars()` 复用完整 hit/accent/don-ka salience 流水线，并逐小节读取 `ResolutionPlan`；`rank_bar_salience_candidates()` 只保留无需量化即可由目标 resolution 表达的 canonical 点，以证据等级、综合分、置信度、hit/accent 和 grid 建立确定性顺序，同格重复点只保留排序最优项。

当前 8.2 已将该候选层接入规则普通 note：`generate_fallback_chart_bars()` 对整组小节只构建一次 salience，`_select_hit_grids()` 先按候选顺序消费除 `weak-evidence` 外的可靠点，达到原有 `target_hits` 后立即停止；候选不足时才沿用原有 onset/beat/style/spectral/instrument 评分补点。因此 density hint、course speed cap、occupancy、silent/rest/sparse、特殊音符路径和最终 hit 数目标保持原边界，且不可表达的强 onset 不会作为可靠候选抢占可表达 salience。弱证据比例、连续弱点和最小骨架限制仍属于 8.3。

## 8.2 无证据补点

高难度允许在持续音乐活动中补充可玩骨架，但必须限制：

- 没有 beat、onset、activity 或结构支持的 note 比例；
- 连续多个无证据 note；
- breakdown/rest 中的补点；
- 仅为达到目标 NPS 而铺满弱格点；
- 高 resolution 导致的额外 note 数。

补点应优先连接已有节奏，而不是创建与音频无关的新周期。

当前 8.3 已把可靠候选之后的选择拆成两层。第一层只接收低置信 salience 或仍有 onset/accent/beat/downbeat、activity、spectral attack/flux、instrument onset 支持的格点，总弱补点比例按 Easy/Normal/Hard/Oni 限制为目标 hit 的 45%/50%/55%/60%；sparse 最多 2 个，breakdown 最多 2 个。第二层才允许无上述证据的 style 骨架，并进一步限制为目标 hit 的 10%/12%/15%/20%；sparse 和 breakdown 为 0，整小节没有任何格点证据但仍活跃时最多保留 1/2/2/3 个 course 最小骨架。所有非可靠补点共享最多 2 个连续输出格的限制，并对可靠节奏一至两个输出步长内的连接点加分；预算基于原 `target_hits` 而非 output resolution，因此升到 24/48 或 18/36 格不会自动增加弱 note。最终 benchmark 总 note 从 3358 降至 3295，四难度仍保持 360 < 605 < 1052 < 1278，静音区违规从 64 降至 60。

当前 8.4 通过独立约束矩阵锁定上述生成边界。负荷矩阵覆盖 3 种拍号、每种 3 档 resolution、4 个 course 和 5 档 density，共 180 个活跃谱面配置：每个结果必须满足 course speed cap、按 legacy resolution 计算的绝对 occupancy cap 和按实际输出长度计算的 occupancy ratio；同 course 内 density 不递减，同 density 内 Easy/Normal/Hard/Oni 不递减，同一输入在对应 resolution 家族中的普通 hit 数完全一致。静音矩阵再覆盖相同 180 个配置和每组 5 类小节，要求首尾静音与中段 rest 全零、活跃小节非空、sparse 不超过 4 个普通 hit。矩阵共验证 360 个生成配置、1080 个小节结果，没有发现需要调整生成器的约束回归。

## 8.3 AI 输入

AI payload 建议发送：

- 每小节目标 resolution；
- 稀疏 salience point；
- hit/accent/don/ka/confidence 的 0–1000 整数值；
- density hint；
- structure role；
- phrase progress；
- silent/rest 标记。

不再要求 AI 根据具体乐器名称决定 note 时间。

当前 8.5 已将 AI 输入升级为 `tja-ai-chartgen-compact-v5`。`bar_salience` 与 bars 按位置对齐，每个小节包含 bar confidence、稳定 fallback reason 和稀疏 point；point 使用 legend 定义的 `[grid,hit,accent,don_preference,ka_preference,sustained_activity,confidence,kind]` 行，其中连续值统一编码为 0–1000 整数。salience 在发送前复用与 fallback 相同的完整 hit/accent/don-ka 流水线和候选排序，并按当前逐小节 `ResolutionPlan` 删除不可精确表达的点，因此 AI 不会看到无法合法返回的细分 salience。prompt 明确要求先消费 strong-transient、transient、rhythmic-skeleton、structure-highlight，再考虑 weak-evidence 或短连接点；instrument 与原始 audio channel 降为乐句、motif、段落和邻近支持上下文，不再直接决定 note tick。

## 8.4 AI 校验

在 AI sanitize/quality gate 中增加：

- 可靠 salience 覆盖；
- 无证据 note 比例；
- 静音违规；
- 强 onset 长时间无响应；
- note 与目标 resolution 可表示性。

首轮 report-only，确认指标与人工试听一致后再进入 repair。

当前 8.5 新增 `ai-salience-validation-v1` 首轮 report-only 诊断。每次成功解析并通过现有硬门控的 AI 结果都会记录普通 note 数、目标 resolution 可表达 note 数、可靠候选及 strong-transient 覆盖、最长连续强瞬态漏响应、无任何 salience 候选支持的 note 数/比例、首尾静音 note 数，以及最多 16 个无支持、漏可靠和漏强瞬态示例。该结果同时写入返回的 AI output、成功 attempt 和需要持久化的 `ai_attempts*.json`，但 `report_only=true`，不会消耗内容修复次数或改变当前 AI 接受/回退语义；非法 canonical tick、目标 resolution 不可表达和首尾静音违规仍由既有硬校验阻断。

## 8.5 退出条件

- 合成 fixture 上 note/onset 对齐率高于修改前 baseline；
- 无证据 note 比例下降；
- Easy/Normal/Hard/Oni 负荷仍单调；
- silent/rest/sparse 不退化；
- 16/24/48 和 12/18/36 的等价节奏保持一致；
- 规则与 AI 结果使用同一对齐指标。

当前 8.6 已新增 `evaluation/phase_two_acceptance.py` 与 `tools/verify_phase_two.py`，形成 `phase-two-acceptance-v1` 持久化验收。工具在禁用 socket 连接时连续运行两次完整 `chart-alignment-v1`，要求结果逐值一致且 68/68 谱面确定；以仓库保留的 Phase 2 前 benchmark 为对照，锁定 note/onset precision 必须提高、unsupported-note ratio 必须下降、strong onset/downbeat recall 不得下降、静音违规不得增加，并检查 Easy/Normal/Hard/Oni 总 note 数单调。验收还独立执行与 8.4 相同边界的 360 配置/1080 小节矩阵，覆盖 NPS、occupancy、density/course 单调、16/24/48 与 12/18/36 等价性，以及 silent/rest/sparse；最后将规则谱面送入 AI 实际使用的 `ai-salience-validation-v1`，确认两类消费者共享同一组 report-only 对齐字段。结果写入 `tests/fixtures/audio/phase_two_acceptance.json` 和 `.md`。

本次验收 PASS：note/onset precision 从 0.626563 提升到 0.638240，unsupported-note ratio 从 0.059857 降到 0.045524，strong onset recall 从 0.923077 提升到 0.973077，downbeat recall 从 0.934211 提升到 0.982456，静音违规从 65 降到 60；四难度 note 数为 360 < 605 < 1052 < 1278，约束矩阵 0 违规。AI salience 诊断继续保持 `report_only=true`，进入 repair 的阈值仍留给后续人工试听校准。

## Phase 2 任务划分列表

- [x] 8.1 建立 salience 候选排序与可表达性过滤
- [x] 8.2 让规则普通 note 优先跟随可靠 salience
- [x] 8.3 限制无证据补点和连续弱证据铺点
- [x] 8.4 全程验证 course、density、NPS、occupancy 与静音约束
- [x] 8.5 将紧凑 salience 输入接入 AI prompt 与校验
- [x] 8.6 执行 Phase 2 阶段验收

---

# Phase 3：改进重音、咚咔和 fill 响应

**优先级：P2**
**预计工作量：M**
**依赖：Phase 1、Phase 2**

## 9. 阶段目标

在 note 时间对齐稳定后，提高强拍、频带攻击、cadence 和 fill burst 的谱面表达，不依赖具体乐器类别。

### Phase 3 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 9.1 重音候选升级 | 部分完成 | 当前已有 beat/downbeat、accent 和 peak 证据；尚未统一到 accent salience。 | 现有 accent coverage 测试可复用。 |
| 9.2 咚咔软映射 | 部分完成 | 已有低频咚、高频咔弱偏好；尚未基于统一 salience 校准。 | 现有配色与单色串指标可复用。 |
| 9.3 Fill burst 检测 | 部分完成 | 当前 structure-v1 已有 fill candidate 和 instrument fill support；尚未建立纯节奏 burst 指标。 | 现有 fill fixture 与特殊音符测试可复用。 |
| 9.4 特殊音符响应 | 部分完成 | 当前特殊音符只在活跃 fill candidate 中生成；尚未使用统一 burst salience。 | 现有特殊音符回归可复用。 |
| 9.5 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 9.1 重音

- downbeat 是稳定候选，但不要求每个 downbeat 都使用大音符；
- 强 onset 与 downbeat 一致时提高 accent；
- off-beat 强 onset 可以成为音乐重音；
- peak 不通过堆积大音符表达；
- Easy/Normal 限制重音数量；
- 连续大音符和不可玩交替必须受控。

## 9.2 咚咔

- 频带证据只提供偏好分，不直接输出 note 字符；
- style、左右手可玩性和重复控制继续参与最终配色；
- 高置信低频攻击可提高咚概率；
- 高置信高频/percussive 攻击可提高咔概率；
- 证据不足时沿用 style motif；
- 避免因为全曲 brightness 偏高而生成过多咔。

## 9.3 Fill burst

纯节奏 fill 证据可以包括：

- 小节后半段 onset 密度突然增加；
- spectral flux 局部爆发；
- percussive ratio 上升；
- 前后稳定段形成对比；
- 位于高置信 phrase end/cadence；
- burst 后进入新 section、peak 或 drop。

低 fill score 的 phrase end 不应生成机械 fill。

## 9.4 特殊音符

滚奏和气球继续要求：

- `--special-notes` 显式启用；
- 活跃、非静音；
- fill candidate；
- 足够持续时间；
- 与普通 note 不冲突；
- TJA preflight 合法。

新增 salience 只改善候选位置和持续范围，不改变合法性边界。

## 9.5 退出条件

- 强 onset、downbeat 和 cadence 的响应率提高；
- ka ratio 和最长单色串不退化；
- fill burst fixture 能命中合理后半小节；
- 无 burst 的 phrase end 不机械生成 fill；
- 特殊音符数量和持续时间保持可解释。

## Phase 3 任务划分列表

- [ ] 9.1 将 downbeat、强 onset 和结构变化统一为重音候选
- [ ] 9.2 使用频带与 percussive 证据校准 don/ka 软倾向
- [ ] 9.3 建立纯节奏 fill burst 检测
- [ ] 9.4 让滚奏和气球响应可靠 burst salience
- [ ] 9.5 执行 Phase 3 阶段验收

---

# Phase 4：BeatNet、librosa 与 onset-grid 仲裁

**优先级：P1**
**预计工作量：M–L**
**依赖：Phase 0**

## 10. 阶段目标

让 BeatNet 成为可拒绝、可部分采用的候选分析器。重点使用其 downbeat/meter 能力，同时避免在稳定 4/4 音乐上覆盖更可靠的 librosa/onset-grid 结果。

### Phase 4 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 10.1 建立统一节拍候选 | 部分完成 | 已有 librosa、onset-grid 和 BeatNet 结果，尚未统一建模。 | 现有 audio analyze 测试可复用。 |
| 10.2 定义候选证据 | 部分完成 | onset-grid 已记录支持率、onset 数、覆盖率和次佳候选；尚未覆盖 BeatNet meter/downbeat。 | 现有置信度拒绝测试可复用。 |
| 10.3 实现候选仲裁 | 部分完成 | 已有低置信 onset-grid 拒绝和 BeatNet 异常 fallback；尚未实现候选间择优。 | 尚未执行综合 benchmark。 |
| 10.4 支持部分采用 | 未开始 | 尚未实现保留 BPM 但采用 BeatNet meter/downbeat。 | 尚未执行。 |
| 10.5 变速诊断 | 未开始 | 尚未记录固定 BPM 拟合误差和疑似 rubato。 | 尚未执行。 |
| 10.6 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 10.1 候选模型

统一表达：

```python
class TempoMeterCandidate(BaseModel):
    source: str
    bpm: float
    offset: float
    time_signature: str
    beat_times: list[float]
    downbeat_times: list[float]
    onset_support: float
    time_coverage: float
    interval_stability: float
    confidence: float
    accepted: bool = False
    reason: str | None = None
```

候选至少包括：

- librosa；
- librosa + onset-grid；
- BeatNet；
- BeatNet + onset-grid。

## 10.2 证据

- onset 对 beat grid 的支持；
- 有效 onset 数；
- 时间覆盖率；
- 候选与次佳候选差距；
- beat interval 稳定性；
- BeatNet beat number 完整性；
- meter 在全曲的稳定性；
- downbeat 附近低频/percussive/accent 支持；
- 3/4、4/4、6/8 小节长度合理性；
- 半速/倍速候选的解释力；
- 可选 drum/bass onset 只能作为附加证据。

## 10.3 决策规则

- BeatNet 输出结构合法不等于接受；
- BeatNet 与 librosa BPM 一致但 downbeat 更可靠时，可以只采用 downbeat/meter；
- BeatNet onset 支持明显更低时保留 librosa/onset-grid；
- 两个候选差距不足时保守保留基础结果并记录 ambiguity；
- 手动 BPM/OFFSET/拍号覆盖始终优先；
- 覆盖前的自动诊断继续持久化；
- 6/8 继续使用项目统一的四分音符 BPM 语义。

## 10.4 固定 BPM 限制

当前项目最终重建规则 beat grid。首轮不实现 tempo map，但应记录：

- BeatNet beat interval 变异；
- 固定 BPM 拟合误差；
- 是否疑似变速、rubato 或现场演奏；
- 因项目限制采用的保守结果。

只有真实需求明确后，才独立设计 `#BPMCHANGE` 和非均匀小节时间轴。

## 10.5 默认开关

在自动 benchmark 证明以下条件前，`--use-beatnet` 继续默认关闭：

- 简单 4/4 不系统性退化；
- 3/4、6/8 或弱鼓点样本明显改善；
- downbeat 误差下降；
- 半速/倍速错误不增加；
- 失败继续稳定回退。

## 10.6 退出条件

- 仲裁优于固定采用任一单候选；
- BeatNet 可以被拒绝或只贡献 meter/downbeat；
- `analysis.json` 能解释最终决策；
- 缺少 BeatNet 依赖不影响默认分析；
- fixture benchmark 覆盖 3/4、4/4、6/8、弱起和半速/倍速模式。

## Phase 4 任务划分列表

- [ ] 10.1 将 librosa、onset-grid 和 BeatNet 统一为节拍候选模型
- [ ] 10.2 定义 onset、downbeat、meter、稳定性和速度别名证据
- [ ] 10.3 实现候选评分、拒绝、择优和 ambiguity 决策
- [ ] 10.4 支持保留原 BPM、仅采用 BeatNet meter/downbeat
- [ ] 10.5 增加固定 BPM 拟合误差与疑似变速诊断
- [ ] 10.6 执行 Phase 4 阶段验收

---

# Phase 5：可选 htdemucs 声部节奏增强

**优先级：P2**
**预计工作量：M**
**依赖：Phase 1**

## 11. 阶段目标

保留 `htdemucs` 对节奏有价值的粗粒度声部角色，移除近期主线对 AST 分类成功的依赖。

### Phase 5 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 11.1 复用 stem activity/onset | 已完成 | instrument-v1 已提供 vocals/drums/bass/other activity 与 onset。 | 现有 instrument 测试和真实模型 smoke 已覆盖。 |
| 11.2 定义 stem-role 轻量模式 | 未开始 | 尚未允许只准备/加载 separator 而不要求 AST。 | 尚未执行。 |
| 11.3 接入 RhythmicSalience | 未开始 | 尚未通过统一 salience 融合 stem onset。 | 尚未执行。 |
| 11.4 保持 partial/fallback | 部分完成 | 当前 AST 失败时已保留 Demucs 证据；尚未形成无分类器的正式 complete 语义。 | 现有 partial 测试可复用。 |
| 11.5 控制模型与性能成本 | 部分完成 | 已有设备选择、segment、overlap 和 CUDA OOM 重试。 | 尚未测量 stem-role 轻量 profile。 |
| 11.6 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 11.1 声部职责

### vocals

- vocal activity；
- vocal presence ratio；
- vocal onset；
- 人声进入和退出；
- phrase/cadence 的弱证据。

不做歌词、音素或逐音节 note 映射。

### drums

- drum activity；
- drum onset；
- percussive burst；
- fill salience；
- accent 的较强软证据。

不做完整鼓件转录。

### bass

- bass activity；
- bass onset；
- downbeat/beat 对齐；
- 低频咚倾向。

不做音高或 bass line 转录。

### other

- accompaniment activity；
- accompaniment onset；
- texture/section 变化；
- 不再要求判断具体乐器。

## 11.2 模型准备拆分

建议允许准备轻量 profile：

```text
models/stem-role-v1/demucs/
models/stem-role-v1/model_manifest.json
```

或者在统一 manifest 中将 classifier 设为可选组件。

要求：

- 唯一联网入口仍是模型准备命令；
- 生成阶段只使用本地 repo；
- 固定 revision 和文件 hash；
- 模型代码与权重许可证分开记录；
- 旧 `models/instrument-v1/` 继续可识别；
- 不要求用户为了 stem onset 下载 AST。

## 11.3 状态语义

建议将粗粒度声部能力定义为独立 feature version，例如：

```text
stem-role-v1
```

避免继续使用 `instrument-v1 complete` 表示“Demucs + AST 均成功”。

新状态：

- `unavailable`：未启用；
- `complete`：Demucs 与 stem 特征成功；
- `fallback`：没有 stem 证据，基础 salience 继续；
- `partial`：仅在确实存在部分 stem 或后处理失败时使用。

旧 `instrument-v1` JSON 继续兼容读取。

## 11.4 融合原则

- drum onset 权重高于通用 mix onset，但仍需活动门控；
- bass onset 只弱增强 beat/downbeat 和 don preference；
- vocal onset 只弱支持 phrase/cadence，不逐音节铺 note；
- accompaniment onset 作为 motif/highlight 证据；
- Demucs artifact 不得通过归一化变成强 salience；
- stem 与 mix/spectral 一致时提高置信度；
- stem 与基础证据冲突时保守降权。

## 11.5 退出条件

- 用户可以只准备 `htdemucs` 声部模型，不下载 AST；
- stem-role 模式完全离线；
- 模型失败时默认 salience 结果不变；
- drum/bass/vocal/accompaniment 证据能改善相应 fixture；
- AI payload 不再依赖具体乐器 taxonomy；
- 远程 Web 仍需管理员显式允许重型分析。

## Phase 5 任务划分列表

- [x] 11.1 复用现有 vocals、drums、bass、other activity/onset
- [ ] 11.2 定义只需要 `htdemucs` 的 stem-role 轻量模式
- [ ] 11.3 将 stem activity/onset 接入 `RhythmicSalience`
- [ ] 11.4 调整 complete、partial、fallback 与旧 instrument-v1 兼容语义
- [ ] 11.5 验证模型准备、离线加载、设备和性能成本
- [ ] 11.6 执行 Phase 5 阶段验收

---

# Phase 6：建立谱面对齐质量报告与门控

**优先级：P1**
**预计工作量：M–L**
**依赖：Phase 1–Phase 3**

## 12. 阶段目标

用可解释指标衡量 note 是否响应音频节奏。先作为 report-only 指标，经过 fixture 和试听校准后再选择少量稳定指标进入 AI repair。

### Phase 6 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 12.1 note/onset 对齐指标 | 未开始 | 尚未实施统一指标。 | 尚未执行。 |
| 12.2 强 onset 与 downbeat 响应 | 部分完成 | 已有 accent、drum 和 bass 对齐指标；尚未统一到 salience。 | 现有 QualityReport 测试可复用。 |
| 12.3 无证据 note 指标 | 未开始 | 尚未实施。 | 尚未执行。 |
| 12.4 fill burst 对齐 | 部分完成 | 已有 fill candidate precision 和 instrument fill support；尚未加入纯节奏 burst。 | 现有 fill 测试可复用。 |
| 12.5 report-only 校准 | 未开始 | 尚未实施新指标对比。 | 尚未执行。 |
| 12.6 AI repair 门槛评审 | 未开始 | 尚未决定任何新阻断阈值。 | 尚未执行。 |
| 12.7 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 12.1 建议指标

### `note_onset_alignment`

普通 note `1`–`4` 中，位于可靠 salience/onset 容差范围内的比例。

### `strong_onset_response`

强 onset 中被附近普通 note 或合理特殊音符响应的比例。

### `downbeat_response`

可靠 downbeat 中被普通 note 响应的比例。该指标不要求所有 downbeat 都命中，阈值应按 course 和 density 校准。

### `unsupported_note_rate`

普通 note 中缺少以下任一证据的比例：

- onset/salience；
- beat/downbeat；
- 持续 activity；
- structure highlight；
- 合理的节奏连接。

### `fill_burst_alignment`

实际 fill 或特殊音符与可靠 burst/fill candidate 重叠的比例。

### `silent_range_violation`

已知静音范围和首尾静音小节中的违规 note 数。

### `rhythmic_quantization_error`

被选 note 对应音频 salience 位置量化到输出 resolution 后的平均误差。

### `salience_coverage_by_density`

按 silent/rest/sparse/normal/dense/fill 分组统计谱面对可靠 salience 的响应，避免高难度与低难度使用同一目标。

## 12.2 指标边界

- 只统计普通 note 时明确排除 `5`、`7`、`8`；
- 长音使用起点、终点和持续范围的独立语义；
- 不要求每个 onset 都变成 note；
- 不以对齐率单独衡量趣味性；
- 不鼓励为了提高覆盖率而过度铺点；
- silent/rest 优先于覆盖率；
- 高 resolution 不应自动提高 note 数；
- 相同节奏在不同 resolution 下应得到相近指标。

## 12.3 进入 AI repair 的条件

新指标只有同时满足以下条件才可成为门槛：

1. 合成 fixture 上语义稳定；
2. 修改前后趋势符合预期；
3. 不惩罚合理留白；
4. 不鼓励机械逐 onset 映射；
5. 不因 course、density 或 resolution 产生系统偏差；
6. 多首真实歌曲试听与指标方向一致；
7. 阈值有宽松区间和明确错误信息。

首轮优先考虑：

- 静音违规；
- 极端无证据 note；
- 强 onset 长时间完全无响应。

其他指标继续 report-only。

## 12.4 退出条件

- 规则与 AI 成品使用相同指标；
- fixture ground truth 可自动计算所有核心指标；
- 指标不随 16/24/48 或 12/18/36 的等价编码明显变化；
- 至少完成一次修改前后 A/B 报告；
- 只有经过校准的少量指标进入 AI repair；
- QualityReport 不引入难以解释的统一总分。

## Phase 6 任务划分列表

- [ ] 12.1 实现 `note_onset_alignment`
- [ ] 12.2 实现强 onset 与 downbeat 响应指标
- [ ] 12.3 实现 `unsupported_note_rate` 与静音违规指标
- [ ] 12.4 实现 fill burst 对齐和节奏量化误差指标
- [ ] 12.5 使用全部 fixture 校准 report-only 指标
- [ ] 12.6 评审并选择少量指标进入 AI repair
- [ ] 12.7 执行 Phase 6 阶段验收

---

# Phase 7：消费者收敛与旧乐器分类降级

**优先级：P2**
**预计工作量：M–L**
**依赖：Phase 1、Phase 5、Phase 6**

## 13. 阶段目标

让 structure、fallback、AI prompt 和质量报告围绕 rhythm/salience 与粗粒度声部工作，把具体乐器 taxonomy 降为兼容字段或远期实验输入。

### Phase 7 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 13.1 structure 消费 salience | 未开始 | 尚未实施统一消费。 | 尚未执行。 |
| 13.2 fallback 消费 salience | 未开始 | 尚未实施统一消费。 | 尚未执行。 |
| 13.3 AI payload 移除具体乐器依赖 | 未开始 | 当前 compact payload 仍含 bar instruments。 | 尚未执行。 |
| 13.4 QualityReport 收敛 | 未开始 | 尚未以 rhythm alignment 指标替代具体乐器主线指标。 | 尚未执行。 |
| 13.5 旧 instrument-v1 兼容 | 部分完成 | 当前 schema 已有兼容默认值；尚未定义降级后的长期读取策略。 | 现有旧 JSON 测试可复用。 |
| 13.6 UI 与 notice 调整 | 未开始 | 尚未区分 stem-role 与具体乐器分类。 | 尚未执行。 |
| 13.7 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 13.1 Structure

结构层优先使用：

- energy/onset/activity；
- rhythm/activity profile；
- spectral novelty；
- salience 趋势；
- 可选 vocals/drums/bass/other 活动；
- phrase 前后窗口变化。

具体乐器名称不能成为 section 或 transition role 的必要条件。

## 13.2 Fallback

fallback 使用：

- hit salience 选择普通落点；
- accent salience 选择重音候选；
- don/ka preference 影响配色；
- sustained activity 控制连接和留白；
- burst salience 支持 fill；
- course、level、style、density 继续决定最终负荷和可玩性。

## 13.3 AI compact payload

建议新增：

- `rhythmic_salience_feature_version`；
- salience legend；
- 每小节 salience 汇总；
- 投影到目标 resolution 的稀疏 salience points；
- 可选 stem-role 汇总。

逐步移除 AI 对以下字段的依赖：

- guitar；
- piano/keyboard；
- strings；
- brass；
- woodwind；
- synth；
- organ。

旧 sidecar 继续可读取，但新 prompt 不要求这些字段存在。

## 13.4 QualityReport

具体乐器相关指标可以：

- 保留读取兼容；
- 当 `instrument-v1` 存在时继续生成诊断；
- 不进入近期质量门槛；
- 不作为默认模型升级的完成条件。

近期核心指标转为：

- note/onset alignment；
- strong onset response；
- unsupported note rate；
- downbeat response；
- fill burst alignment；
- rhythmic quantization error；
- silent range violation。

## 13.5 UI 与 Notice

用户界面应明确区分：

- 基础节奏分析：默认启用，无重型模型；
- BeatNet：可选节拍/downbeat/meter 增强；
- 声部节奏增强：可选 `htdemucs`，增加耗时；
- 旧具体乐器分类：兼容或实验功能，不再作为推荐主线。

如果最终停止准备 AST，应提供清晰迁移说明，而不是静默改变现有命令。

## 13.6 退出条件

- 未准备任何重型模型时，完整节奏优先路径可运行；
- 启用 `htdemucs` 时只增加声部活动/onset 证据；
- 具体乐器字段缺失不会降低核心谱面生成能力；
- AI payload 比 instrument-v1 路线更紧凑；
- 旧 analysis/config/sidecar 继续兼容读取；
- CLI/Web notice 与实际启用能力一致。

## Phase 7 任务划分列表

- [ ] 13.5 先锁定旧 instrument-v1、analysis、config 和 sidecar 兼容行为
- [ ] 13.1 让 structure 消费统一 salience 与可选 stem-role
- [ ] 13.2 让 fallback 使用 hit、accent、don/ka 和 burst salience
- [ ] 13.3 让 AI payload 使用紧凑 salience，并移除具体乐器依赖
- [ ] 13.4 将 QualityReport 主线收敛到节奏对齐指标
- [ ] 13.6 调整 CLI/Web 开关、进度、notice 和文档
- [ ] 13.7 执行 Phase 7 阶段验收

---

## 14. 数据版本与兼容策略

### 横向工作开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 14.1 定义 salience feature version | 未开始 | 尚未实施。 | 尚未执行兼容验证。 |
| 14.2 定义 stem-role feature version | 未开始 | 尚未实施。 | 尚未执行兼容验证。 |
| 14.3 升级 analysis schema | 未开始 | 尚未实施。 | 尚未执行旧 JSON 读取验证。 |
| 14.4 保持 GenerationConfig 兼容 | 部分完成 | 当前配置已有 BeatNet 和 instrument 开关；尚未定义 stem-role profile。 | 现有 generate-from-config 测试可复用。 |
| 14.5 保持 AI sidecar 兼容 | 部分完成 | 当前 sidecar 有版本化 compact payload；尚未加入 salience 版本。 | 现有 AI payload 测试可复用。 |
| 14.6 更新 CLI/Web 文档 | 未开始 | 尚未按新路线调整用户描述。 | 尚未执行用户流程验证。 |

### 14.1 Feature version

建议新增独立版本：

```text
rhythmic-salience-v1
stem-role-v1
```

不要把新语义继续写入 `instrument-v1`，因为它当前明确代表 Demucs + AST 具体乐器分析。

### 14.2 Analysis schema

如果持久化以下字段，应提高 `analysis_schema_version`：

- salience feature version/status/reason；
- 每小节 salience 汇总；
- 稀疏 salience points；
- beat 候选与仲裁诊断；
- stem-role version/status/reason；
- 旧 instrument-v1 是否存在。

旧 JSON 缺少新字段时使用默认值读取。

### 14.3 GenerationConfig

近期建议保留：

- `use_beatnet`；
- `use_instrument_analysis` 的兼容读取；
- instrument device/model dir 的兼容读取。

如果新增用户开关，优先使用更准确名称：

```text
--use-stem-analysis
```

是否保留旧 `--use-instrument-analysis` 作为别名，应在独立设计中决定并提供弃用周期。

内部 salience 权重、阈值和窗口不全部暴露为 CLI 参数，避免配置不可维护。

---

## 15. 模型准备、离线与供应链边界

### 15.1 唯一联网入口

模型准备命令继续是唯一允许下载权重的入口。生成、重生成和 Web 局部操作不得联网获取模型。

### 15.2 近期模型范围

近期只需要维护：

- 可选 BeatNet 现有依赖；
- 可选 `htdemucs` stem-role 权重。

AST 权重：

- 旧模型目录继续兼容；
- 不再作为新路线的必要组件；
- 是否停止默认准备应通过独立迁移设计决定；
- 不立即删除用户已经准备的模型。

### 15.3 Manifest

stem-role manifest 至少记录：

- schema version；
- feature version；
- separator model ID；
- fixed revision；
- 文件 hash；
- 代码许可证；
- 权重许可证；
- 采样率；
- segment/overlap 关键参数；
- 准备命令版本。

### 15.4 禁止行为

- 生成阶段隐式下载；
- 使用浮动 `main` revision；
- 根据用户任意 URL 下载 checkpoint；
- 把模型权重提交进 Git；
- 绕过 `.gitignore`；
- 在公开 notice 中暴露绝对模型路径；
- 因 AST 不存在而阻止 stem-role 分析；
- 为远期具体乐器研究预先下载无实际使用的模型。

---

## 16. 错误模型与降级

建议稳定 reason：

- `rhythmic-salience-fallback:<reason>`
- `tempo-candidate-ambiguous`
- `beatnet-low-confidence`
- `beatnet-meter-conflict`
- `suspected-variable-tempo`
- `stem-analysis-disabled`
- `missing-model:htdemucs`
- `invalid-model-manifest`
- `invalid-model-checksum`
- `device-unavailable:<device>`
- `cuda-out-of-memory`
- `stem-analysis-error:<Type>`
- `invalid-stem-output`

原则：

- salience 层失败时仍可使用现有 BarFeature/fallback；
- BeatNet 失败时保留 librosa/onset-grid；
- stem-role 失败时保留基础 salience；
- 旧 AST 失败不影响新节奏主线；
- notice 只记录稳定 reason，不记录 traceback、绝对路径或模型内部响应。

---

## 17. 测试矩阵

### 17.1 默认 CI

默认 CI 不下载重型权重、不要求 GPU，覆盖：

- fixture 与 ground truth 同步重建；
- onset、beat、downbeat benchmark；
- salience 纯函数；
- 静音门控；
- 频带咚咔倾向；
- note 候选排序；
- 无证据 note 指标；
- fill burst；
- BeatNet 仲裁纯函数；
- fake Demucs stem-role；
- structure/fallback/AI/QualityReport 消费；
- CLI/Web notice；
- 旧 analysis/config/sidecar 兼容。

### 17.2 真实模型 smoke

保留非默认 marker：

```bash
pytest -m instrument_model -v
```

后续可以改名或增加：

```bash
pytest -m stem_model -v
```

迁移期应避免同一测试被两个 marker 重复执行。

真实 smoke 验证：

- 本地模型目录；
- 完全离线加载；
- CPU 推理；
- 可用时 CUDA 推理；
- stem 名称、shape、长度和有限值；
- 数字静音；
- 短 OGG；
- Demucs 失败后的基础 salience fallback。

### 17.3 平台矩阵

| 平台 | Python | 默认 CI | 真实 stem smoke |
| --- | --- | --- | --- |
| Ubuntu | 3.11 | 完整 | 推荐 |
| Ubuntu | 3.13 | 完整 | 可选 |
| Windows | 3.11 | 完整 | 至少本地验证 |
| Windows | 3.13 | 完整 | 依赖支持后启用 |

### 17.4 回归原则

- 不对完整 `analysis.json` 做字节级快照；
- fixture ground truth 必须确定性；
- 浮点值使用明确容差；
- 相同节奏不同 resolution 的指标应接近；
- 模型 smoke 与普通 CI 分离；
- 性能数据记录但不使用易波动墙钟时间作为普通 CI 硬门槛；
- 自动指标不替代最终试听和游玩。

---

## 18. 性能预算

### 默认节奏分析

- 不加载 PyTorch、Demucs 或 Transformers；
- 不增加模型下载；
- salience 融合主要使用已有 envelope；
- 避免重复 HPSS、onset 或 spectral 计算；
- `analysis.json` 只保存稀疏 salience。

### BeatNet 增强

- 继续显式启用；
- 失败快速回退；
- 仲裁计算相对神经网络推理应保持轻量；
- 不为了比较同时运行多个 BeatNet 预训练模型。

### Stem-role 增强

- 继续显式启用；
- 支持 CPU，但明确耗时；
- CUDA OOM 只有限重试；
- 不隐式切换到不可预测耗时的 CPU 全曲重跑；
- `--max-bars` 只分析需要的前缀加边界上下文；
- 不加载 AST，降低额外显存和分类耗时。

---

## 19. 用户界面与文档

实现时同步更新：

- `README.md`
- `README_EN.md`
- `CLAUDE.md`
- `docs/quality-evaluation.md`
- `docs/development-roadmap.md`

文档应说明：

- 默认节奏分析不需要大型模型；
- BeatNet 主要用于 downbeat/meter 增强，不保证总是更好；
- stem-role 分析只提供粗粒度 vocals/drums/bass/other；
- 具体乐器识别不再是推荐主线；
- 不需要用户制作人工标注数据集；
- 自动 benchmark 使用程序生成 ground truth；
- 音频特征是谱面软证据，不是音乐转录；
- 所有模型在准备后离线运行。

---

## 20. 远期可选研究：具体乐器识别

具体乐器识别整体移出近期里程碑。以下方向不进入当前实施顺序：

- AST 替换；
- EfficientAT/DyMN；
- OpenMIC 微调；
- PaSST；
- BEATs；
- MERT；
- CLAP；
- SCNet-large separator 替换；
- BS-RoFormer 人声增强；
- 吉他、钢琴、弦乐、铜管、木管和合成器 taxonomy 扩展。

只有出现以下信号后，才新建设计文档重新评估：

1. 节奏对齐指标已经稳定，主要剩余问题明确来自声源混叠。
2. AI 或规则谱面需要根据配器设计可验证的 motif 差异。
3. 用户明确需要具体乐器可视化或编辑能力。
4. 有许可明确、可离线部署且无需自建大规模标注的模型。
5. 具体乐器结果能在不增加大量 prompt 的情况下改善最终谱面。
6. 能用公开 benchmark 和少量试听验证收益，不要求维护者自行标注数据集。

远期研究不能阻塞默认节奏主线。

---

## 21. 分阶段交付顺序

```text
Phase 0  无人工标注的自动对齐基线
   ↓
Phase 1  统一 RhythmicSalience
   ↓
Phase 2  普通 note 对齐
   ↓
Phase 3  重音、咚咔和 fill
   ↓
Phase 6  对齐 QualityReport 与门控
   ↓
Phase 7  消费者收敛与旧分类降级

Phase 0
   ↓
Phase 4  BeatNet/librosa/onset-grid 仲裁

Phase 1
   ↓
Phase 5  可选 htdemucs stem-role 增强
   ↓
Phase 7
```

Phase 4 和 Phase 5 可以在 Phase 2–3 期间独立推进，但核心 salience 和谱面生成不得依赖 BeatNet 或 Demucs 必须成功。

---

## 22. 推荐里程碑

### 里程碑状态

| 里程碑 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| Milestone A：自动对齐基线 | 未开始 | 尚未完成 Phase 0。 | 尚未执行里程碑验收。 |
| Milestone B：节奏显著性 v1 | 未开始 | 尚未完成 Phase 1。 | 尚未执行里程碑验收。 |
| Milestone C：规则谱面对齐 | 未开始 | 尚未完成 Phase 2–3。 | 尚未执行里程碑验收。 |
| Milestone D：节拍仲裁 v2 | 未开始 | 尚未完成 Phase 4。 | 尚未执行里程碑验收。 |
| Milestone E：轻量声部增强 | 未开始 | 尚未完成 Phase 5。 | 尚未执行里程碑验收。 |
| Milestone F：质量报告与消费者收敛 | 未开始 | 尚未完成 Phase 6–7。 | 尚未执行里程碑验收。 |

### Milestone A：自动对齐基线

完成标志：

- fixture ground truth 可确定性重建；
- onset、beat、downbeat、resolution 和谱面对齐有 baseline；
- 不需要人工标注真实歌曲。

### Milestone B：节奏显著性 v1

完成标志：

- 统一 hit/accent/don/ka/activity/confidence；
- 稀疏持久化；
- 静音正确；
- 消费者不重复融合相同证据。

### Milestone C：规则谱面对齐

完成标志：

- note/onset 对齐提高；
- 无证据 note 减少；
- 重音、咚咔和 fill 响应改善；
- 四难度和可玩性不退化。

### Milestone D：节拍仲裁 v2

完成标志：

- BeatNet 可接受、拒绝或部分采用；
- 简单 4/4 不系统性退化；
- downbeat/meter 有可解释改善；
- 默认仍可不启用 BeatNet。

### Milestone E：轻量声部增强

完成标志：

- 只准备 `htdemucs` 即可运行；
- 不要求 AST；
- stem onset 接入 salience；
- 失败保持基础路径。

### Milestone F：质量报告与消费者收敛

完成标志：

- 规则和 AI 使用同一对齐指标；
- 新指标经过 fixture 和试听校准；
- 具体乐器字段不再是核心依赖；
- 旧 JSON、配置和 sidecar 兼容。

---

## 23. 每阶段通用完成清单

- [ ] 建立修改前 baseline。
- [ ] 编写失败测试并确认失败原因。
- [ ] 实现最小可验证变更。
- [ ] 运行定向单元与集成测试。
- [ ] 重建 fixture 和 ground truth。
- [ ] 运行自动对齐 benchmark。
- [ ] 比较修改前后 QualityReport。
- [ ] 运行 `ruff check .`。
- [ ] 运行 `pytest`。
- [ ] 涉及模型时运行适用的真实 smoke。
- [ ] 对若干真实歌曲执行 A/B 试听和 Web 播放检查，但不要求人工标注。
- [ ] 检查生成阶段没有隐式网络访问。
- [ ] 检查模型 revision、hash 和许可证记录。
- [ ] 检查输出不包含绝对路径、用户音频内容或秘密。
- [ ] 数据流、命令、依赖或持久化格式变化时更新 `CLAUDE.md`。
- [ ] 用户可见行为变化时更新 README 和 Web 提示。
- [ ] 更新本路线图对应状态、完成详情和验证记录。
- [ ] 使用中文 Git 提交信息描述目标或结果。

---

## 24. 最终决策矩阵

| 子系统 | 当前方案 | 近期方向 | 默认建议 |
| --- | --- | --- | --- |
| 基础节奏 | librosa + onset-grid | 建立统一 salience | 默认启用 |
| 频谱节奏 | `spectral-v1` | 作为 salience 核心证据 | 默认启用 |
| 结构 | `structure-v1` | 消费 salience 趋势 | 默认启用 |
| 分辨率 | `ResolutionPlan` | 与 salience 量化误差联动 | 默认启用 |
| BeatNet | 合法输出后合并 | 多候选证据仲裁 | 继续可选 |
| 声部分析 | `htdemucs + AST` | `htdemucs` stem-role，无 AST 依赖 | 可选增强 |
| 具体乐器分类 | AST taxonomy | 降为兼容/远期研究 | 不作为近期主线 |
| EfficientAT/DyMN/OpenMIC | 未集成 | 暂缓 | 不实施 |
| SCNet/BS-RoFormer | 未集成 | 暂缓 | 不实施 |
| 人工真实歌曲标注 | 无 | 不建立 | 不需要 |
| 自动评测 | 部分 fixture/QualityReport | 程序 ground truth + alignment benchmark | 最高优先级 |

---

## 25. 总结

新的推荐路线是：

```text
默认：
librosa + onset-grid + spectral-v1 + structure-v1 + ResolutionPlan
                       ↓
              RhythmicSalience
                       ↓
              note / accent / don-ka / fill

可选：
BeatNet → 只在仲裁胜出时改善 beat/downbeat/meter
htdemucs → 只增强 vocals/drums/bass/other activity/onset

暂缓：
AST、EfficientAT/DyMN、OpenMIC、具体乐器 taxonomy、SCNet、BS-RoFormer
```

这条路线直接服务于“让 note 节奏对上音频”，避免维护者自行标注数据，也避免在基础节奏问题尚未解决时投入高成本、低确定性的具体乐器识别。

近期最高优先级依次是：

1. 自动 fixture ground truth 和对齐 baseline；
2. 统一 `RhythmicSalience`；
3. 普通 note 与强 onset 对齐；
4. 重音、咚咔和 fill 响应；
5. BeatNet 候选仲裁；
6. 可选 `htdemucs` stem-role；
7. QualityReport 与 AI/规则消费者收敛。

只有节奏主线已经稳定，并且具体乐器分类能证明对最终谱面有额外、可验证的收益时，才重新启动乐器识别研究。

---

## 26. 参考资料

- [BeatNet 官方仓库](https://github.com/mjhydri/beatnet)
- [librosa beat tracking 文档](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
- [Demucs 官方仓库](https://github.com/facebookresearch/demucs)
- [MIREX Audio Beat Tracking](https://www.music-ir.org/mirex/wiki/Audio_Beat_Tracking)
- [MIREX Audio Downbeat Estimation](https://www.music-ir.org/mirex/wiki/Audio_Downbeat_Estimation)
