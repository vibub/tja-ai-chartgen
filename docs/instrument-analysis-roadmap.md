# 人声、乐器与节拍分析升级路线图

日期：2026-07-15

## 1. 文档目的

本文定义 `tja-ai-chartgen` 在现有 `instrument-v1` 基础上升级人声、乐器与节拍分析能力的实施路线。路线图采用“先建立证据，再逐层替换”的方式，避免同时更换 source separation、乐器分类、时间平滑和节拍分析后无法判断质量变化来源。

核心方向如下：

1. 短期保留 Demucs `htdemucs` 四轨分离，先解决当前 AST AudioSet 分类与粗时间窗口造成的主要瓶颈。
2. 引入针对音乐乐器识别的 EfficientAT/DyMN + OpenMIC 路线，并保留可替换分类器边界。
3. 将固定 4 秒窗口、2 秒 hop 升级为更细时间粒度，并通过平滑、滞回和最短持续时间约束避免标签抖动。
4. 区分 mix、vocals、drums、bass、other 的分析职责，控制重复推理成本。
5. 将 BeatNet 从“合法输出即作为新基线”升级为与 librosa/onset-grid 进行证据仲裁的候选分析器。
6. 只有在项目真实歌曲与最终谱面指标证明收益后，才考虑以 SCNet-large 替换 `htdemucs`。
7. BS-RoFormer 仅作为后期可选的人声增强层，不进入首轮默认实现。

本文是专题路线图，不直接改变当前运行时行为。每个阶段都应拆成独立设计、实现计划和提交。

### 状态维护规范

所有可拆分开发目标都使用状态表跟踪，开发过程中直接更新对应行，不另建重复清单。

状态字段统一使用：

- `未开始`：尚未进入设计或实现。
- `设计中`：正在确认边界、数据模型或验收标准。
- `开发中`：已经开始编写测试或实现。
- `部分完成`：已有可复用基础或部分子项完成，但尚未满足该目标的全部验收条件。
- `验证中`：实现已完成，正在运行 benchmark、回归或人工试听。
- `已完成`：实现、验证和必要文档均已完成。
- `已阻塞`：存在依赖、许可证、模型、平台或评测数据阻塞。
- `已取消`：评测证明收益不足或不再符合项目目标。

“完成详情”应记录实际完成内容、关键决策和与原计划的差异；“验证记录”应记录执行过的测试、benchmark、平台和结果。不得只把状态改为 `已完成` 而保留“尚未实施”。

---

## 2. 当前基线

### 2.1 当前 source separation

当前 `instrument-v1` 使用 Demucs `htdemucs` 输出：

- `vocals`
- `drums`
- `bass`
- `other`

每个 stem 提取：

- 与基础音频分析 hop 对齐的 RMS 活动；
- 相对 stem 能量占比；
- onset strength；
- canonical grid 稀疏 onset 证据。

现有优势：

- 已有稳定的本地模型准备与 manifest 校验；
- 生成阶段完全离线；
- 支持 CPU、CUDA、MPS；
- CUDA OOM 有限重试；
- 模型失败时保留基础音频流水线；
- 已接入 structure、AI prompt、fallback 与 QualityReport。

### 2.2 当前乐器分类

当前分类器为：

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

当前分类过程：

1. 将 mix 与 `other` stem 重采样到 16 kHz mono。
2. 使用 4 秒窗口、2 秒 hop。
3. 分别运行 AST AudioSet 多标签分类。
4. 将 AudioSet 标签归并到固定 taxonomy：
   - `guitar`
   - `piano_keyboard`
   - `strings`
   - `brass`
   - `woodwind`
   - `synth`
   - `organ`
   - `other_instrument`
5. 按 mix/other 权重与活动门控映射到小节。

### 2.3 当前 BeatNet 集成

BeatNet 当前采用：

```python
model=1
mode="offline"
inference_model="DBN"
```

处理流程为：

```text
librosa beat_track
→ onset-grid BPM/OFFSET 估计
→ 可选 BeatNet beat/downbeat/meter
→ 使用 BeatNet 平均拍间隔得到 BPM
→ onset-grid 再尝试校正
→ 重建固定 BPM 的规则 beat grid
```

现有实现能处理导入、推理、输出解析和合并异常，但只要 BeatNet 返回结构合法的结果，就会把 BeatNet 作为新的基础候选；当前没有 BeatNet 与原始 librosa 结果的统一比较分数。

---

## 3. 已确认问题

### 3.1 AST 与项目目标不完全匹配

AST 当前 checkpoint 面向通用 AudioSet，而不是专门的音乐多乐器识别。由此产生：

- 非音乐环境音标签占用模型能力；
- 相近乐器容易混淆；
- AudioSet 标签到项目 taxonomy 的手工映射损失信息；
- 模型的通用 mAP 不等于 OpenMIC 乐器识别质量；
- 更换更干净的 separator 不一定能解决分类错误。

### 3.2 时间窗口过粗

4 秒窗口、2 秒 hop 会模糊：

- 乐器进入和退出；
- 短 solo；
- 吉他、钢琴或管乐短句；
- build-up 中逐步加入的声部；
- cadence 前后的配器变化；
- 小节级 fill 和伴奏 highlight。

### 3.3 `other` stem 承担过多语义

`other` 可能同时包含：

- 吉他；
- 钢琴；
- 弦乐；
- 管乐；
- 合成器；
- 非主唱人声残留；
- Demucs 分离泄漏和 artifact。

仅改善 `other` 的分类器仍不能完全解决多乐器共存与时间定位问题。

### 3.4 BeatNet 的优势没有被完整利用

BeatNet 可以输出逐拍 beat/downbeat，但当前流水线最终重建固定 BPM 网格。因此在以下音乐上，即使 BeatNet 逐拍跟踪正确，其优势也会被压缩：

- rubato；
- 真人演奏速度漂移；
- 渐快或渐慢；
- 现场录音；
- 包含 BPM change 的歌曲。

### 3.5 缺少模型级与谱面级的统一对照

当前缺少能够回答以下问题的稳定 benchmark：

- 新分类器是否真的比 AST 更准？
- 时间窗口变细是否提高切换定位，还是只增加噪声？
- SCNet 是否比 `htdemucs` 更有利于谱面生成？
- BeatNet 是否提高 downbeat/meter，还是在简单 4/4 上引入错误？
- 模型指标的提升是否转化为 structure、fallback 和最终谱面质量提升？

---

## 4. 目标与非目标

### 4.1 目标

1. 提高主要乐器存在性识别的准确性和稳定性。
2. 将乐器进入、退出和切换定位到接近小节级，而不是 2–4 秒粗粒度。
3. 保持人声、鼓、贝斯、伴奏 onset 的 canonical grid 证据。
4. 让结构分析更可靠地利用人声进出、鼓/贝斯增强和主导乐器切换。
5. 让规则谱面和 AI prompt 获得更可信、但仍属于软证据的音乐语义。
6. 提高 BeatNet 对 downbeat 和 meter 的收益，同时拒绝格式合法但证据较弱的输出。
7. 保持所有模型显式准备、固定 revision、离线加载和可审计 manifest。
8. 保持未启用 instrument analysis 时的依赖、启动速度和生成结果不变。
9. 保持模型失败时的 partial/fallback 行为。
10. 通过真实歌曲、自动指标和人工试听共同决定模型是否升级为默认实现。

### 4.2 非目标

本路线图不计划实现：

- 歌手身份识别；
- 歌词、音素或语言识别；
- 完整 MIDI 或多乐器音符转录；
- 精确乐器型号识别；
- 将分离 stem 作为公开下载产物；
- 生成阶段自动联网下载模型；
- 让分类标签直接突破静音、NPS、density、occupancy、resolution 或 TJA preflight 硬约束；
- 以单一 SDR、mAP 或统一加权总分决定最终模型；
- 在第一阶段支持任意第三方 checkpoint；
- 在没有许可证记录的情况下发布或自动下载模型权重。

---

## 5. 总体架构方向

目标数据流：

```text
输入音频
  ↓
librosa onset/RMS + spectral-v1
  ↓
librosa / onset-grid / BeatNet 候选与证据仲裁
  ↓
htdemucs（首轮保留）
  ├─ vocals activity/onset
  ├─ drums activity/onset
  ├─ bass activity/onset
  └─ other activity/onset
  ↓
可替换 InstrumentClassifier
  ├─ mix 乐器存在性
  └─ other 细粒度伴奏乐器
  ↓
时间平滑、滞回、最短持续时间与置信度校准
  ↓
canonical InstrumentBarFeature / InstrumentGridFeature
  ↓
structure-v1 / AI compact payload / fallback / QualityReport
```

首轮实现不要求对五个音频源都执行完整分类：

- `mix`：判断整体配器和防止 separator 漏判；
- `other`：识别吉他、钢琴、弦乐、管乐、合成器等；
- `vocals`：主要使用 activity/onset，不重复做大型通用分类；
- `drums`：主要使用 activity/onset，可后续增加专用鼓事件分析；
- `bass`：主要使用 activity/onset 和 downbeat 对齐，不重复做大型分类。

---

# Phase 0：建立可重复基线

**优先级：P1**  
**预计工作量：M–L**  
**依赖：无**

## 6. 阶段目标

在更换模型之前建立当前 `htdemucs + AST + BeatNet` 的稳定基线。任何后续模型只有同时通过模型层、结构层和谱面层比较，才允许进入下一阶段。

### Phase 0 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 6.1 测试语料分层 | 部分完成 | 已有可确定性重建的节拍、结构、静音和 resolution fixture；尚缺人声/乐器组合 fixture 与本地真实歌曲分层集。 | 现有真实音频流水线测试可复用，尚未执行本路线图专项 benchmark。 |
| 6.2 标注范围 | 未开始 | 尚未建立人声、乐器进入/退出及 downbeat 的匿名标注协议。 | 尚未执行。 |
| 6.3 基线指标 | 部分完成 | 已有 instrument、structure 与谱面 report-only 指标；尚缺模型级 F1/mAP、时间边界误差和分析器胜出率。 | 现有 QualityReport 测试可复用，尚未形成统一 baseline。 |
| 6.4 Benchmark 产物 | 未开始 | 尚未实施。 | 尚未执行。 |
| 6.5 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 6.1 测试语料分层

### A. 合成与可提交 fixture

继续使用现有确定性生成音频，并新增许可明确的合成组合：

- 人声样式持续 tone + 间歇段；
- 鼓组主拍、反拍和短 fill；
- 贝斯主拍与切分；
- 钢琴/吉他式短促 harmonic transient；
- 稳定段 → 加入鼓 → 加入人声 → 全编制 peak → drop；
- 3/4、4/4、6/8；
- 弱起、前置静音和非第 0 秒 downbeat。

合成 fixture 用于验证时间轴、门控、平滑和 schema，不宣称能代表真实模型准确率。

### B. 本地真实歌曲评测集

建立不进入仓库的本地评测集，至少覆盖：

- 稳定 4/4 电子音乐；
- 流行歌曲；
- 摇滚；
- 弱鼓点木吉他或钢琴；
- 人声密集与纯器乐；
- 3/4；
- 6/8；
- 切分音明显；
- 速度有轻微漂移的真人演奏。

每首歌曲保存匿名 hash ID，不在仓库和报告中保存标题、艺人、绝对路径或原始音频。

## 6.2 标注范围

人工标注采用小节级和事件级混合方式：

- 每小节主要活跃 stem；
- 每小节存在的 OpenMIC 乐器类别；
- 乐器进入、退出和显著切换时间；
- downbeat 时间；
- BPM、拍号和半速/倍速语义；
- 允许不确定标签，不强制标注听不清的乐器。

不要求完整逐音符转录。

## 6.3 基线指标

### 模型层

- 每类 precision、recall、F1；
- macro-F1；
- micro-F1；
- mAP；
- 乐器进入/退出边界误差；
- 标签每分钟切换次数；
- 同一稳定段内的预测抖动率；
- stem 泄漏导致的误报比例。

### Beat/meter 层

- BPM 绝对误差与相对误差；
- 半速/倍速错误率；
- offset/downbeat 绝对误差；
- beat F-measure；
- downbeat F-measure；
- 3/4、4/4、6/8 拍号准确率；
- librosa、onset-grid、BeatNet 各自胜出比例。

### 项目语义层

- structure boundary 命中率；
- build-up/peak/drop/breakdown/cadence 方向一致性；
- `instrument_transition_response`；
- `instrument_confident_bar_ratio`；
- `drum_onset_hit_coverage`；
- `bass_downbeat_alignment`；
- `vocal_phrase_response`；
- `instrument_fill_support`。

### 性能层

分别记录 CPU 和 CUDA：

- 模型准备后磁盘体积；
- 峰值内存/显存；
- 60 秒音频推理耗时；
- 5 分钟音频推理耗时；
- Windows 与 Ubuntu 是否可运行；
- Python 3.11 与 3.13 是否可运行。

## 6.4 产物

- 新增可重复 benchmark 工具，不把真实音频纳入仓库。
- 输出匿名 JSON 与 Markdown 汇总。
- 记录运行环境、模型 revision、模型 hash 和参数。
- baseline 报告必须区分 mock/合成测试与真实歌曲结果。

## 6.5 退出条件

- 当前 AST、Demucs 和 BeatNet 均有可重复基线。
- 至少能定位当前误差主要来自 separator、classifier、窗口还是节拍仲裁。
- benchmark 不需要网络，不写入用户音频内容或绝对路径。
- 结果能在同一机器重复运行并获得稳定数量级。

---

# Phase 1：抽象可替换分类器边界

**优先级：P1**  
**预计工作量：M**  
**依赖：Phase 0**

## 7. 阶段目标

在不改变现有输出语义的前提下，将 AST 专用实现收敛为可替换分类器接口，为 EfficientAT/DyMN 验证提供边界。

### Phase 1 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 7.1 可替换分类器模块边界 | 未开始 | 尚未实施。 | 尚未执行。 |
| 7.2 Manifest 升级 | 未开始 | 尚未实施。 | 尚未执行。 |
| 7.3 数据模型兼容 | 未开始 | 尚未实施。 | 尚未执行。 |
| 7.4 分类器边界测试 | 未开始 | 尚未实施。 | 尚未执行。 |
| 7.5 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 7.1 建议模块边界

新增内部分类器抽象，职责包括：

- 声明模型 ID、revision、许可证和输入采样率；
- 声明窗口长度、hop 与 batch size；
- 完全离线加载；
- 输入批量 mono waveform；
- 输出稳定 taxonomy 分数；
- 校验 shape、有限值和标签顺序；
- 释放 GPU 资源；
- 暴露可测试的模型元数据。

业务层不得依赖某个模型的具体 feature extractor 或 logits 标签顺序。

## 7.2 Manifest 升级

模型 manifest 应从“固定 Demucs + AST”升级为“separator + classifier 组件”：

```text
schema_version
feature_version
separator:
  backend
  model_id
  revision
  license
  files
classifier:
  backend
  model_id
  revision
  license
  taxonomy_version
  input_sample_rate
  window_seconds
  hop_seconds
  files
```

要求：

- 显式旧 manifest 继续可识别；
- 未知 schema version 明确拒绝；
- 文件 hash 继续可选完整校验；
- 模型代码许可证与权重许可证分开记录；
- 准备命令成功后再原子发布；
- 生成阶段不自动迁移或下载。

## 7.3 数据模型兼容

建议引入 `instrument-v2`，而不是静默改变 `instrument-v1` 语义。

`InstrumentAnalysisRaw` 增加或调整：

- separator backend/model/revision；
- classifier backend/model/revision；
- taxonomy version；
- window/hop 配置；
- 可选置信度校准版本；
- smoothing policy version。

旧 `instrument-v1` JSON 继续读取；新代码不得把旧 AST 分数解释成经过 OpenMIC 校准的分数。

## 7.4 测试

- fake AST backend 与现有结果等价；
- fake EfficientAT/DyMN backend 输出相同稳定协议；
- 标签顺序变化不影响 taxonomy；
- 模型缺失、hash 错误、未知 backend 稳定 fallback；
- 离线环境不触发网络；
- 旧 manifest 与旧 analysis JSON 兼容；
- partial 状态继续保留 Demucs stem 证据。

## 7.5 退出条件

- AST 通过新接口运行时，现有测试和真实 baseline 不产生非预期变化。
- 分类器后端可通过配置或 manifest 选择，但用户生成命令暂不暴露任意 checkpoint。
- 业务消费者只读取稳定 Pydantic 模型。

---

# Phase 2：验证 EfficientAT/DyMN + OpenMIC

**优先级：P1**  
**预计工作量：L–XL**  
**依赖：Phase 0、Phase 1**

## 8. 阶段目标

使用专门的音乐多乐器数据和稳定 taxonomy 替换通用 AST AudioSet 分类。首选评估 EfficientAT/DyMN 系列，OpenMIC-2018 作为主要 taxonomy 与微调基准。

### Phase 2 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 8.1 模型候选筛选 | 未开始 | 尚未实施。 | 尚未执行。 |
| 8.2 Taxonomy 策略 | 未开始 | 尚未实施。 | 尚未执行。 |
| 8.3 训练与权重流程 | 未开始 | 尚未实施。 | 尚未执行。 |
| 8.4 模型接受门槛 | 未开始 | 尚未实施。 | 尚未执行。 |
| 8.5 分类器降级策略 | 未开始 | 尚未实施。 | 尚未执行。 |
| 8.6 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 8.1 候选顺序

1. DyMN-L：质量优先候选。
2. DyMN-S 或同系列较小模型：CPU/显存友好候选。
3. PaSST：若 EfficientAT/DyMN 的可部署 checkpoint、许可证或 Windows 兼容性不满足要求，则作为保守备选。
4. BEATs：仅在计划维护项目专用微调权重时进入后续研究，不作为首轮即插即用模型。
5. MERT：因官方常用权重存在非商业许可证限制，不作为默认发布候选。

## 8.2 Taxonomy 策略

内部保留 OpenMIC 的标准标签层，再确定性映射到项目 taxonomy。

建议保留：

- OpenMIC 原始类别分数；
- 项目稳定类别分数；
- 映射版本；
- 未知/未映射类别不进入 AI prompt。

项目 taxonomy 首轮继续兼容现有字段，不无依据增加大量类别。只有在真实歌曲中证明能稳定识别且能改善谱面时，才增加新字段。

## 8.3 训练与权重要求

如果没有满足要求的现成 checkpoint，应建立可重复微调流程：

- 固定数据集版本和 split；
- 固定随机种子；
- 记录数据预处理；
- 保存最佳 checkpoint 的评测指标；
- 输出模型卡；
- 记录代码、预训练权重、微调数据与最终权重许可证；
- 不将训练数据或大模型权重提交到 Git；
- 准备命令只下载已经审核、固定 revision 的最终推理产物。

## 8.4 模型接受门槛

新分类器必须同时满足：

- OpenMIC macro-F1 和 mAP 明显高于当前 AST baseline；
- 真实歌曲的主要乐器误报率下降；
- 稳定段标签抖动不高于 AST；
- 乐器进入/退出边界不因分类延迟明显恶化；
- CPU 推理仍可完成，不要求必须有 CUDA；
- Windows Python 3.11 至少通过真实模型 smoke；
- 权重允许项目预期使用场景；
- 完全离线加载；
- 模型体积和耗时在文档中明确告知用户。

仅提高通用 AudioSet mAP，不足以替换 AST。

## 8.5 降级策略

- classifier 失败但 separator 成功：`partial`，继续使用 stem activity/onset；
- classifier 权重缺失：明确 notice，不回退联网下载；
- EfficientAT/DyMN 不可用时，不在同一次生成中隐式加载 AST 双模型；
- 需要 AST 回退时，由模型目录或显式兼容配置选择，而不是运行时静默切换。

## 8.6 退出条件

- 选定一个默认 OpenMIC 分类后端和一个资源友好配置。
- 模型卡、许可证、revision、hash、输入参数和 benchmark 完整。
- 新模型在真实歌曲和谱面消费者指标上优于 AST，或至少在显著降低计算成本的同时保持质量。

---

# Phase 3：提高时间分辨率并抑制标签抖动

**优先级：P1**  
**预计工作量：M–L**  
**依赖：Phase 2**

## 9. 阶段目标

将乐器分类从粗粒度 clip tagging 升级为适合小节结构的时间序列，同时避免短窗口带来的不稳定预测。

### Phase 3 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 9.1 窗口与 hop 参数对照 | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.2 时间平滑与 hysteresis | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.3 置信度校准 | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.4 小节映射 | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.5 性能控制 | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.6 时间序列测试 | 未开始 | 尚未实施。 | 尚未执行。 |
| 9.7 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 9.1 初始参数

首轮 benchmark 使用：

```text
window_seconds = 2.0
hop_seconds = 0.5
```

同时保留以下对照：

- 4.0 / 2.0：现有 baseline；
- 3.0 / 1.0：中间成本候选；
- 2.0 / 0.5：推荐高时间分辨率候选。

最终参数由真实歌曲 benchmark 决定，不仅根据单模型验证集决定。

## 9.2 时间平滑

建议按以下顺序处理：

1. 将窗口分数映射到统一时间轴。
2. 对分数做短范围加权平滑。
3. 使用进入阈值与退出阈值不同的 hysteresis。
4. 删除低于最短持续时间的孤立标签。
5. 合并间隔很短的同类片段。
6. 在数字静音和边缘静音区域强制清零。
7. 再映射到小节和 phrase。

进入阈值应高于退出阈值，避免在边界附近来回切换。

## 9.3 置信度校准

不得直接假定 sigmoid 分数跨模型、跨类别可比较。至少评估：

- 每类独立阈值；
- temperature scaling 或等价校准；
- 第一、第二候选差距；
- mix 与 other 一致性；
- 对应 stem 的实际活动；
- 标签持续时间。

`dominant_instrument` 只有在绝对分数、活动门控和候选差距均满足要求时设置。

## 9.4 小节映射

窗口映射到小节时继续使用时间重叠权重，但应同时保存：

- 平均置信度；
- 峰值置信度；
- presence ratio；
- 进入/退出事件；
- 高置信持续时间。

结构层优先消费稳定段和变化事件，不直接消费单个短窗口尖峰。

## 9.5 性能控制

- 批量推理；
- 避免为 mix 与 other 重复执行可共享的预处理；
- 限制 batch 的峰值内存；
- CUDA OOM 时只降低 batch size 并有限重试；
- 不在 OOM 后自动切换到不可预测耗时的 CPU 全曲重跑；
- `--max-bars` 只分析所需前缀加一个窗口上下文。

## 9.6 测试

- 2 秒窗口和 0.5 秒 hop 的边界数量；
- 最后不足窗口时的确定性补零；
- 平滑前后时间轴长度不变；
- hysteresis 进入/退出行为；
- 孤立尖峰被移除；
- 稳定标签不被过度平滑；
- 乐器真实切换不被延迟超过约一个 hop；
- 首尾静音强制归零；
- 短 `--max-bars` 不分析整首歌曲。

## 9.7 退出条件

- 乐器进入/退出边界误差优于 4 秒/2 秒 baseline。
- 稳定段抖动率不高于 baseline。
- 计算成本增幅有明确记录并处于可接受范围。
- 新时间参数进入 manifest 和 `analysis.json`，结果可复现。

---

# Phase 4：分 stem 职责与证据融合

**优先级：P2**  
**预计工作量：M**  
**依赖：Phase 2、Phase 3**

## 10. 阶段目标

使用不同 stem 的可靠信息，但避免对所有 stem 重复运行大型分类器。

### Phase 4 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 10.1 定义 mix 与各 stem 职责 | 部分完成 | 当前 mix/other 执行 AST，vocals/drums/bass/other 已提供 activity/onset；尚未形成 instrument-v2 的正式职责协议。 | 现有 instrument 单元测试覆盖基础 stem 证据。 |
| 10.2 动态证据融合 | 部分完成 | 当前已有固定 mix/other 权重和活动门控；尚未实现按活动、一致性和 artifact 动态调整。 | 已有固定融合测试，尚未执行动态融合验证。 |
| 10.3 结构与生成消费规则 | 未开始 | 尚未实施。 | 尚未执行。 |
| 10.4 AI payload 控制 | 未开始 | 尚未实施。 | 尚未执行。 |
| 10.5 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 10.1 推荐职责

### mix

- 整体乐器存在性；
- 防止 Demucs 将乐器错误分到其他 stem 后完全漏判；
- 作为 other 分类的弱补充。

### other

- 吉他、钢琴/键盘、弦乐、铜管、木管、合成器、风琴等主要分类输入；
- 结合 mix 识别分离泄漏和多乐器共存。

### vocals

- 人声 activity；
- 人声 presence ratio；
- vocal onset；
- 人声进入、退出与乐句收束；
- 首轮不运行完整 OpenMIC 分类器。

### drums

- drum activity；
- drum onset；
- fill burst；
- 与 beat/downbeat 的关系；
- 首轮不进行鼓件级转录。

### bass

- bass activity；
- bass onset；
- 与 beat/downbeat 的对齐；
- 首轮不进行音高或 bass note 转录。

## 10.2 融合规则

融合不得继续固定依赖单一 `0.65 × other + 0.35 × mix` 常量。建议根据证据动态调整：

- other 活动高且 mix/other 一致：提高 other 权重；
- other 活动低但 mix 分类稳定：保留 mix 弱证据；
- other 存在明显 artifact 或标签快速抖动：降低置信度；
- vocals/drums/bass stem 活动与 mix 标签冲突：标记不确定，不强行设置 dominant instrument；
- 数字静音或总活动不足：全部清零。

动态融合必须通过纯函数实现，并有参数化测试。

## 10.3 结构与生成消费规则

- 人声进入/退出只能增强已有 phrase boundary，不独立创建边界；
- drum/bass 同时增强可支持 build-up/peak；
- 人声退出且鼓/贝斯减弱可支持 breakdown；
- 人声收束后的 drum burst 可支持 cadence/fill；
- dominant instrument 切换只有在前后均高置信且持续足够长时才支持 section boundary；
- 所有 instrument 证据继续服从 density、NPS、occupancy、resolution 和静音硬约束。

## 10.4 AI payload 控制

提高时间分辨率后不得把所有 0.5 秒窗口原样发送给 AI。继续采用：

- 小节汇总；
- 进入/退出事件；
- 仅发送达到门槛的 dominant/secondary instrument；
- canonical grid 只发送 stem onset；
- 通过固定 legend 表达字段；
- 低置信度与平滑前原始分数不进入 prompt。

## 10.5 退出条件

- 新融合逻辑在真实歌曲中降低 mix-only 和 other-only 的典型误判。
- AI payload 大小没有因窗口变细线性膨胀。
- 规则生成不会把乐器标签机械映射为固定音符。

---

# Phase 5：BeatNet、librosa 与 onset-grid 证据仲裁

**优先级：P1**  
**预计工作量：M–L**  
**依赖：Phase 0**

## 11. 阶段目标

将 BeatNet 改为可比较候选，而不是结构合法后直接采用。重点提升 downbeat 和 meter，同时避免简单 4/4 音乐被错误 BeatNet 结果拖累。

### Phase 5 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 11.1 建立节拍候选集合 | 部分完成 | 已有 librosa、onset-grid 与 BeatNet 结果；尚未统一建模为可比较候选集合。 | 现有音频分析测试覆盖各分析器基础路径。 |
| 11.2 定义仲裁证据 | 部分完成 | onset-grid 已记录支持率、有效 onset、覆盖率和次佳候选；尚未加入统一 BeatNet/downbeat/meter 比较。 | 现有 onset-grid 置信度测试已覆盖部分证据。 |
| 11.3 实现决策规则 | 部分完成 | 已有低置信 onset-grid 拒绝和 BeatNet 异常 fallback；尚未实现候选间择优、部分采用和 ambiguity 决策。 | 已验证异常输出不阻断主流程，尚未执行仲裁 benchmark。 |
| 11.4 固定 BPM 与变速诊断 | 未开始 | 尚未实施。 | 尚未执行。 |
| 11.5 BeatNet 模型选择评测 | 未开始 | 尚未实施。 | 尚未执行。 |
| 11.6 节拍仲裁指标 | 未开始 | 尚未实施。 | 尚未执行。 |
| 11.7 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 11.1 候选集合

统一生成以下候选：

1. librosa beat tracker 原始结果；
2. librosa + onset-grid 结果；
3. BeatNet 原始 beat/downbeat/meter 结果；
4. BeatNet 基线 + onset-grid 校正结果。

每个候选必须保留：

- BPM；
- offset/downbeat；
- time signature；
- beat times；
- 来源；
- 证据分数；
- 拒绝原因；
- 是否存在半速/倍速别名。

## 11.2 仲裁证据

候选评分至少考虑：

- onset 对候选 beat grid 的支持率；
- downbeat 附近的低频、鼓组和 accent 支持；
- 有效 onset 数；
- 时间覆盖率；
- 与次佳候选的差距；
- beat interval 稳定性；
- BeatNet beat number 的完整性；
- meter 在全曲的稳定性；
- 3/4、4/4、6/8 下的小节长度合理性；
- 半速/倍速候选在整首上的解释力；
- 若 instrument analysis 已启用，drum/bass onset 只能作为附加证据，不能成为必需依赖。

## 11.3 决策规则

- BeatNet 输出结构合法不等于自动接受。
- 如果 BeatNet 与 librosa/onset-grid 基本一致，优先保留 downbeat/meter 更完整的 BeatNet 结果。
- 如果 BeatNet 的 onset 支持明显较低，保留 librosa/onset-grid。
- 如果两者差距不足，保守保留原基础结果并记录 ambiguity。
- 如果 BeatNet 只改善 meter/downbeat，可保留原 BPM，仅采用可靠 meter/downbeat。
- 如果 BeatNet 判断 6/8，继续使用项目统一的四分音符 BPM 语义。
- 手动 BPM/OFFSET/拍号覆盖始终优先，但覆盖前诊断继续持久化。

## 11.4 固定 BPM 与变速边界

首轮仍保持单一 BPM 的项目约束，但应明确记录：

- BeatNet beat interval 变异程度；
- 固定 BPM 拟合误差；
- 是否疑似变速或 rubato；
- 当前因不支持 tempo map 而采用的保守结果。

如果真实评测证明变速歌曲占比和收益足够，再独立设计 tempo map / `#BPMCHANGE` 生成，不在本路线图中顺带实现。

## 11.5 BeatNet 模型选择

当前固定 `model=1`。应在本地真实评测集中比较 BeatNet 提供的不同预训练模型，但不按歌曲类型自动猜测并动态切换，除非：

- 模型选择规则可解释；
- 不需要同时运行全部模型；
- 对不同曲风有稳定收益；
- 不增加不可接受的推理耗时。

首轮优先改进仲裁，再考虑切换 BeatNet 预训练模型。

## 11.6 指标

- Beat/downbeat F-measure；
- offset 误差；
- meter accuracy；
- 半速/倍速错误率；
- BeatNet 被接受、部分采用和拒绝的比例；
- 仲裁后相对最佳单候选的 regret；
- 简单 4/4 上的退化率；
- 3/4、6/8 和弱鼓点样本上的改善率。

## 11.7 退出条件

- 仲裁结果在评测集上不低于最佳单一默认候选的稳定水平。
- BeatNet 仍保持可选，不因安装或推理失败影响基础结果。
- `analysis.json` 能解释最终采用、部分采用或拒绝 BeatNet 的原因。
- 在有足够真实数据前，不默认启用 `--use-beatnet`。

---

# Phase 6：评估 SCNet-large 替换 `htdemucs`

**优先级：P2**  
**预计工作量：L**  
**依赖：Phase 0–Phase 4**

## 12. 阶段目标

在分类器、窗口和平滑已经稳定后，单独评估 separator 是否仍是主要瓶颈。

### Phase 6 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 12.1 建立 separator 对照 | 未开始 | 尚未实施。 | 尚未执行。 |
| 12.2 Separator 项目指标 | 未开始 | 尚未实施。 | 尚未执行。 |
| 12.3 工程与平台指标 | 未开始 | 尚未实施。 | 尚未执行。 |
| 12.4 替换门槛评审 | 未开始 | 尚未实施。 | 尚未执行。 |
| 12.5 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 12.1 对照模型

```text
htdemucs
SCNet-large
```

两者使用相同：

- 输入音频；
- 分析时长；
- classifier；
- 窗口与平滑参数；
- canonical 映射；
- structure 与谱面生成参数。

## 12.2 Separator 指标

除 SDR 类指标外，重点记录项目实际需要的指标：

- vocals activity 误检/漏检；
- drum onset precision/recall；
- bass onset precision/recall；
- other stem 中人声和鼓残留导致的分类误报；
- stem 活动时间轴稳定性；
- structure boundary 变化；
- fill candidate 变化；
- 最终 `drum_onset_hit_coverage`；
- 最终 `bass_downbeat_alignment`；
- 最终 `vocal_phrase_response`。

## 12.3 工程指标

- Windows 安装和推理；
- Python 3.11/3.13；
- CPU 推理速度；
- CUDA 峰值显存；
- 模型体积；
- 短音频和长音频稳定性；
- chunk/overlap 边界 artifact；
- OOM 有限重试；
- 完全离线加载；
- 代码和 checkpoint 许可证。

## 12.4 替换门槛

SCNet-large 只有在以下条件同时满足时才替换默认 separator：

1. 至少一个核心 stem 任务显著改善，且其他 stem 没有明显退化。
2. 改善能够传递到结构或谱面指标，而不仅是主观听感更干净。
3. Windows 和 CPU 路径可用。
4. 推理耗时和模型体积可接受。
5. checkpoint 来源、revision、hash 和许可证完整。
6. 现有 partial/fallback、manifest 和模型准备边界可以保持。

否则继续使用 `htdemucs`，并将 SCNet 保留为实验后端。

## 12.5 退出条件

形成明确结论之一：

- SCNet-large 成为默认 separator；
- SCNet-large 仅作为质量优先可选项；
- SCNet-large 未证明对谱面生成有净收益，暂不集成。

不得以“论文 SDR 更高”作为默认替换的唯一依据。

---

# Phase 7：可选 BS-RoFormer 人声增强

**优先级：P3**  
**预计工作量：L–XL**  
**依赖：Phase 6 或真实评测证明 vocals 仍是主要瓶颈**

## 13. 阶段目标

只在人声分离质量明确限制 phrase、section 或 vocal response 时，引入专门的 BS-RoFormer/Mel-Band RoFormer vocals 模型。

### Phase 7 开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 13.1 明确 BS-RoFormer 定位 | 未开始 | 尚未实施。 | 尚未执行。 |
| 13.2 风险与成本评估 | 未开始 | 尚未实施。 | 尚未执行。 |
| 13.3 接受门槛评审 | 未开始 | 尚未实施。 | 尚未执行。 |
| 13.4 阶段退出条件 | 未开始 | 尚未达成。 | 尚未执行阶段验收。 |

## 13.1 推荐定位

BS-RoFormer 不直接替换四轨 separator，而是可选增强：

```text
BS-RoFormer → 高质量 vocals / instrumental
htdemucs 或 SCNet → drums / bass / other
```

高质量 vocals 用于：

- vocal activity；
- vocal onset；
- 人声进入/退出；
- 人声乐句收束；
- 防止人声残留污染 other 分类。

## 13.2 风险

- 社区 checkpoint 质量差异大；
- 代码许可证与权重许可证可能不同；
- 很多模型只输出 vocals/instrumental；
- 显存和推理耗时较高；
- 不同音乐风格可能需要不同 checkpoint；
- 同时运行两个 separator 会显著增加成本。

## 13.3 接受门槛

- checkpoint 许可证明确；
- 固定 revision/hash；
- 不需要运行时联网；
- vocal boundary 和 phrase 指标显著改善；
- other 分类误报下降；
- 总推理成本有明确的用户提示；
- 默认仍可关闭；
- 不把社区模型列表开放成任意远程下载入口。

## 13.4 退出条件

BS-RoFormer 只能成为：

- 明确标注“质量优先、耗时较高”的可选增强；或
- 因成本、许可证或实际收益不足而不集成。

首轮不将其设为默认。

---

## 14. 数据版本与兼容策略

### 横向工作开发状态

| 小目标 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| 14.1 Feature version 升级 | 未开始 | 尚未实施。 | 尚未执行兼容验证。 |
| 14.2 Analysis schema 升级 | 未开始 | 尚未实施。 | 尚未执行兼容验证。 |
| 14.3 GenerationConfig 边界 | 未开始 | 尚未实施。 | 尚未执行配置复跑验证。 |
| 15.1 保持唯一联网入口 | 部分完成 | 当前 `prepare-instrument-models` 已是 instrument-v1 唯一下载入口；尚未覆盖新 classifier/separator。 | 已有离线加载验证，尚未验证新模型。 |
| 15.2 完善模型元数据记录 | 部分完成 | 当前 manifest 已记录模型、revision、许可证和文件 hash；尚未拆分代码、权重与数据许可证。 | 已有 manifest/hash 测试，尚未验证 v2 schema。 |
| 15.3 落实供应链禁止规则 | 部分完成 | 当前生成阶段使用本地 Demucs repo 与 `local_files_only=True`；尚未覆盖新增模型来源。 | 已验证 instrument-v1 离线加载，尚未执行新后端网络检查。 |
| 15.4 保持模型原子发布 | 部分完成 | 当前模型准备已使用 staging、校验和目录替换；尚未适配 manifest v2 与多后端。 | 已有原子替换和失败清理测试基础。 |
| 16 扩展错误模型与降级 | 部分完成 | 已有 complete/partial/fallback 和稳定 reason；尚未增加新 classifier、仲裁与供应链错误类型。 | 现有故障注入测试可复用。 |
| 17.1 默认 CI 覆盖 | 部分完成 | 当前默认 CI 已使用 fake separator/classifier 覆盖 instrument-v1；尚未覆盖 v2 接口、平滑和仲裁。 | 现有完整测试通过记录可作为基线。 |
| 17.2 真实模型 smoke | 部分完成 | 已有 `instrument_model` marker 和真实 Demucs/AST smoke；尚未覆盖新模型。 | 需要准备新权重后执行。 |
| 17.3 平台矩阵 | 部分完成 | 当前 CI 覆盖 Ubuntu Python 3.11/3.13 和 Windows Python 3.13 基础流水线；真实重型模型平台覆盖不足。 | 现有 CI 可作为基础，尚未完成路线图目标。 |
| 17.4 回归规则 | 部分完成 | 已有 schema 语义断言、浮点容差和独立真实模型 marker；尚未加入模型 revision benchmark。 | 现有测试策略已验证基础行为。 |
| 18 Balanced 性能预算 | 未开始 | 尚未为新分类器与细窗口建立相对性能预算。 | 尚未执行性能测量。 |
| 18 Quality 性能预算 | 未开始 | 尚未建立质量优先 profile。 | 尚未执行性能测量。 |
| 19 用户界面与文档更新 | 部分完成 | 当前 CLI/Web 已支持 instrument 开关、设备、进度和 notice；尚未加入 v2 profile、模型信息和新成本提示。 | 现有 CLI/Web 测试可作为基线。 |

### 14.1 Feature version

建议将升级后的分类、平滑和融合语义定义为 `instrument-v2`。

不能仅保留 `instrument-v1` 名称然后改变：

- taxonomy 来源；
- 分数含义；
- 窗口长度；
- 平滑策略；
- dominant instrument 门槛。

### 14.2 Analysis schema

如果新增以下持久化字段，应提高 `analysis_schema_version`：

- classifier backend/revision；
- taxonomy version；
- calibration version；
- smoothing policy；
- instrument enter/exit events；
- beat candidate/arbitration 诊断；
- separator backend。

旧 JSON 继续使用默认值读取，不进行破坏性迁移。

### 14.3 GenerationConfig

只有用户可选择的稳定行为才进入 `GenerationConfig`。内部实验参数不得全部暴露为 CLI/Web 配置。

建议最终只暴露：

- 是否启用 instrument analysis；
- device；
- 经审核的 analysis profile，例如 `balanced` 或 `quality`；
- BeatNet 开关。

窗口、hop、阈值和 backend 在稳定前只属于模型 manifest 或内部实验配置。

---

## 15. 模型准备、离线与供应链边界

### 15.1 唯一联网入口

继续保持：

```bash
tja-ai-chartgen prepare-instrument-models
```

为唯一允许下载正式运行权重的入口。

### 15.2 必须记录

- 模型 ID；
- 固定 revision；
- 文件 SHA-256；
- 模型代码许可证；
- 预训练权重许可证；
- 微调数据许可证；
- 最终 checkpoint 许可证；
- 输入采样率；
- taxonomy；
- 模型版本；
- 准备命令版本。

### 15.3 禁止行为

- 生成阶段隐式访问 Hugging Face、GitHub 或其他模型源；
- 根据用户输入的任意 URL 下载 checkpoint；
- 使用浮动 `main` revision；
- 把权重提交进 Git；
- 使用 `git add -f` 绕过 `models/` ignore；
- 只记录代码许可证而忽略权重许可证；
- 在 notice 或公开 Web 状态中暴露模型绝对路径。

### 15.4 原子发布

模型准备继续使用：

```text
staging 下载
→ 文件 hash
→ 离线子进程加载验证
→ manifest 写入
→ 原子替换正式目录
```

任一步失败都不得破坏已有可用模型目录。

---

## 16. 错误模型与降级

建议稳定 reason 至少覆盖：

- `missing-model:separator`
- `missing-model:classifier`
- `invalid-model-manifest`
- `unsupported-model-manifest-version`
- `invalid-model-checksum`
- `classifier-load-error:<Type>`
- `classifier-output-error:<Type>`
- `classifier-taxonomy-mismatch`
- `classifier-calibration-mismatch`
- `separator-load-error:<Type>`
- `separator-output-error:<Type>`
- `device-unavailable:<device>`
- `cuda-out-of-memory`
- `beatnet-low-confidence`
- `beatnet-meter-conflict`
- `tempo-candidate-ambiguous`

状态语义继续保持：

- `unavailable`：本次未请求；
- `complete`：separator 和 classifier 均成功；
- `partial`：至少 stem activity/onset 可用；
- `fallback`：instrument 证据完全不可用，但基础生成继续。

BeatNet 继续使用独立状态，不与 instrument analysis 状态混合。

---

## 17. 测试矩阵

### 17.1 默认 CI

默认 CI 不下载大型权重、不要求 GPU，覆盖：

- fake separator；
- fake classifier；
- manifest；
- taxonomy；
- 窗口与平滑；
- 置信度门槛；
- canonical 映射；
- structure 消费；
- AI compact payload；
- fallback；
- BeatNet 候选仲裁纯函数；
- CLI/Web notice；
- 旧 JSON 兼容。

### 17.2 真实模型 smoke

继续使用非默认 marker：

```bash
pytest -m instrument_model -v
```

真实 smoke 至少验证：

- 模型目录完整；
- 完全离线加载；
- CPU 推理；
- 可用时 CUDA 推理；
- 输出 shape、有限值和 taxonomy；
- 短音频；
- 数字静音；
- 单次真实 OGG 分析。

### 17.3 平台矩阵

最低目标：

| 平台 | Python | 默认 CI | 真实模型 smoke |
| --- | --- | --- | --- |
| Ubuntu | 3.11 | 完整 | 推荐 |
| Ubuntu | 3.13 | 完整 | 可选 |
| Windows | 3.11 | 完整 | 必须至少本地验证 |
| Windows | 3.13 | 完整 | 依赖支持后启用 |

### 17.4 回归原则

- 不对完整 `analysis.json` 做脆弱字节快照；
- 锁定 schema、关键字段和语义；
- 浮点值使用合理容差；
- 模型输出 benchmark 与普通 CI 分开；
- 模型质量变化必须记录 revision，不允许浮动权重导致无解释回归。

---

## 18. 性能预算

首轮建议使用相对预算，基于当前 `htdemucs + AST` baseline：

### Balanced profile

- 总耗时不超过 baseline 的约 1.5 倍；
- GPU 峰值显存不超过 baseline 的约 1.5 倍；
- CPU 必须可运行；
- 默认只分析 mix 与 other 分类；
- 不启用 BS-RoFormer。

### Quality profile

- 允许更细窗口和更大 classifier；
- 总耗时可达到 baseline 的约 2–3 倍；
- 必须在 UI/CLI 提示耗时和显存成本；
- 可选使用更高质量 separator；
- BS-RoFormer 仍需单独显式启用，不自动包含。

最终预算应以真实测量替代估计，不将易波动的墙钟时间作为普通 CI 硬门槛。

---

## 19. 用户界面与文档

如果实现影响用户行为，应同步更新：

- `README.md`
- `README_EN.md`
- `CLAUDE.md`
- `docs/quality-evaluation.md`
- `docs/development-roadmap.md`

文档必须说明：

- 当前使用的 separator 和 classifier；
- 模型是否为通用 AudioSet 或 OpenMIC 专用；
- 运行阶段完全离线；
- 模型准备命令；
- 权重目录和磁盘体积；
- CPU/CUDA 预期耗时；
- partial/fallback 语义；
- BeatNet 并非总是优于 librosa；
- 乐器标签是概率性编谱证据，不是精确鉴定或转录；
- 启用 instrument analysis 会增加 AI prompt 信息量，但 payload 继续保持紧凑。

---

## 20. 分阶段交付顺序

```text
Phase 0  当前模型与真实歌曲基线
   ↓
Phase 1  可替换分类器接口与 manifest v2
   ↓
Phase 2  EfficientAT/DyMN + OpenMIC 验证
   ↓
Phase 3  2 秒窗口、0.5 秒 hop、平滑与校准
   ↓
Phase 4  分 stem 职责与动态证据融合

Phase 0
   ↓
Phase 5  BeatNet/librosa/onset-grid 仲裁

Phase 2–4 稳定
   ↓
Phase 6  SCNet-large 对照实验
   ↓
Phase 7  可选 BS-RoFormer 人声增强
```

Phase 5 可以在 Phase 1–4 期间独立推进，但不得依赖 instrument analysis 必须开启。

---

## 21. 推荐里程碑

### 里程碑状态

| 里程碑 | 状态 | 完成详情 | 验证记录 |
| --- | --- | --- | --- |
| Milestone A：可比较 | 未开始 | 尚未完成 Phase 0。 | 尚未执行里程碑验收。 |
| Milestone B：分类器可替换 | 未开始 | 尚未完成 Phase 1。 | 尚未执行里程碑验收。 |
| Milestone C：乐器识别 v2 | 未开始 | 尚未完成 Phase 2–4。 | 尚未执行里程碑验收。 |
| Milestone D：节拍仲裁 v2 | 未开始 | 尚未完成 Phase 5。 | 尚未执行里程碑验收。 |
| Milestone E：separator 决策 | 未开始 | 尚未完成 Phase 6。 | 尚未执行里程碑验收。 |
| Milestone F：高质量人声可选增强 | 未开始 | 尚未完成 Phase 7。 | 尚未执行里程碑验收。 |

### Milestone A：可比较

包含 Phase 0。

完成标志：

- 能对当前 AST、BeatNet 和 `htdemucs` 生成匿名 benchmark；
- 能区分模型质量、时间定位与最终谱面指标；
- 后续替换有稳定 baseline。

### Milestone B：分类器可替换

包含 Phase 1。

完成标志：

- AST 通过统一 classifier 接口运行；
- manifest 可以描述不同 classifier；
- 旧模型和旧 JSON 兼容。

### Milestone C：乐器识别 v2

包含 Phase 2、Phase 3、Phase 4。

完成标志：

- 默认候选切换到 OpenMIC 定向分类器；
- 小节级进入/退出定位改善；
- 标签抖动受控；
- AI payload 不显著膨胀；
- `instrument-v2` 可追踪且可回退。

### Milestone D：节拍仲裁 v2

包含 Phase 5。

完成标志：

- BeatNet 与 librosa/onset-grid 通过统一证据比较；
- BeatNet 可只贡献 meter/downbeat；
- 简单 4/4 不因启用 BeatNet 出现系统性退化；
- 仍保持可选开关。

### Milestone E：separator 决策

包含 Phase 6。

完成标志：

- 对 SCNet-large 是否替换 `htdemucs` 有基于项目指标的明确结论；
- 不以论文或听感单独决定。

### Milestone F：高质量人声可选增强

包含 Phase 7。

完成标志：

- 只有在真实需求成立时提供 BS-RoFormer；
- 许可证、成本和收益均可解释；
- 不影响默认 balanced 路径。

---

## 22. 每阶段通用完成清单

- [ ] 先建立修改前 baseline。
- [ ] 编写失败测试并确认失败原因。
- [ ] 实现最小可验证变更。
- [ ] 运行定向单元与集成测试。
- [ ] 运行 `ruff check .`。
- [ ] 运行 `pytest`。
- [ ] 运行适用的真实模型 smoke。
- [ ] 对同一真实歌曲集执行修改前后 benchmark。
- [ ] 比较模型指标、结构指标和最终谱面指标。
- [ ] 执行人工试听和 Web 播放检查。
- [ ] 检查生成阶段没有网络访问。
- [ ] 检查模型 revision、hash 和许可证记录完整。
- [ ] 检查输出文件不包含模型绝对路径、用户音频内容或秘密。
- [ ] 数据流、命令、依赖或持久化格式变化时更新 `CLAUDE.md`。
- [ ] 用户可见行为变化时更新 README 和 Web 提示。
- [ ] 使用中文 Git 提交信息描述目标或结果。

---

## 23. 最终决策矩阵

| 子系统 | 当前方案 | 首选升级 | 何时升级 | 默认建议 |
| --- | --- | --- | --- | --- |
| 四轨分离 | `htdemucs` | SCNet-large | 项目指标证明净收益后 | 暂时保留 `htdemucs` |
| 人声增强 | `htdemucs vocals` | BS-RoFormer | vocals 仍是明确瓶颈且成本可接受 | 不默认启用 |
| 乐器分类 | AST AudioSet | EfficientAT/DyMN + OpenMIC | 通过真实歌曲、许可证和离线门槛 | 最高优先级 |
| 时间窗口 | 4 秒 / 2 秒 | 2 秒 / 0.5 秒 + 平滑 | 边界误差下降且抖动受控 | 优先实施 |
| 分 stem 分类 | mix + other | 职责化 mix/other + stem activity | 不重复大型推理且收益明确 | 优先实施 |
| 节拍分析 | librosa/onset-grid + 可选 BeatNet | 多候选证据仲裁 | 简单 4/4 不退化且弱鼓/downbeat 改善 | BeatNet 暂不默认开启 |
| 变速支持 | 固定 BPM | 独立 tempo map 设计 | 真实需求和评测证明必要 | 本路线图不实现 |

---

## 24. 总结

本路线图的首要结论是：当前最值得优先投入的部分不是立即替换 `htdemucs`，而是解决 AST 的任务不匹配、4 秒窗口的时间模糊和缺少稳定平滑的问题。

推荐执行顺序为：

1. 建立当前 `htdemucs + AST + BeatNet` 的真实基线。
2. 抽象分类器边界和模型 manifest。
3. 采用 EfficientAT/DyMN + OpenMIC 路线验证乐器识别。
4. 引入 2 秒窗口、0.5 秒 hop、置信度校准、平滑和 hysteresis。
5. 明确 mix、other 与各 stem 的职责，控制推理成本。
6. 将 BeatNet 改为可拒绝、可部分采用的候选分析器，继续保持默认关闭。
7. 在以上环节稳定后再评估 SCNet-large。
8. 只有在人声质量仍是主要瓶颈时才增加 BS-RoFormer 可选增强。

所有模型替换都必须证明其改善能传递到结构分析、谱面生成和人工试听，而不仅是通用 benchmark 分数更高或分离音频主观上更干净。

---

## 25. 参考资料

- [BeatNet 官方仓库](https://github.com/mjhydri/beatnet)
- [EfficientAT 官方仓库](https://github.com/fschmid56/EfficientAT)
- [PaSST 官方仓库](https://github.com/kkoutini/PaSST)
- [BEATs 官方实现](https://github.com/microsoft/unilm/tree/master/beats)
- [OpenMIC-2018 数据集](https://zenodo.org/records/1492445)
- [MIREX 2026 Audio Instrument Recognition](https://music-ir.org/mirex/wiki/2026%3AAudio_Instrument_Recognition)
- [SCNet 官方仓库](https://github.com/starrytong/SCNet)
- [BS-RoFormer 实现](https://github.com/lucidrains/BS-RoFormer)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [MERT-v1-95M 模型卡](https://huggingface.co/m-a-p/MERT-v1-95M)
