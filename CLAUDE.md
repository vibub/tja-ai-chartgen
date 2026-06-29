# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`tja-ai-chartgen` 是一个持续迭代的太鼓达人 `.tja` 谱面生成项目。目标流程：

1. 输入音频文件。
2. 使用音频分析工具提取 BPM、拍号、节拍点、结构段落等音乐信息。
3. 将结构化分析结果交给 AI。
4. 生成可维护、可校验、可逐步改进的 `.tja` 谱面。

当前项目使用 Python 3.11+，以 Typer CLI 作为入口，目标是先完成可安装、可测试、可生成规则谱面草稿的 MVP。后续每次引入工程结构、运行命令、外部工具链、核心模块或重要约束时，都应同步更新本文件。

## 当前开发命令

```bash
pip install -e ".[dev]"
tja-ai-chartgen version
pytest
pytest tests/test_tja_writer.py -v
ruff check .
```

生成规则谱面草稿的目标命令：

```bash
tja-ai-chartgen generate path/to/song.mp3 --title "Song Title"
```

系统依赖：`ffmpeg`，用于把输入音频转换为 `.ogg`。

不要在本文件中保留已经失效的命令；命令变更时必须同步更新。

## 当前架构

项目围绕“可替换的流水线”组织，而不是把音频分析、AI 提示词、谱面输出耦合在单个脚本中：

- **CLI 编排层**：`src/tja_ai_chartgen/cli.py` 提供 `version` 和 `generate` 命令，负责串联转换、分析、特征、谱面生成、渲染、校验和报告写入。
- **音频层**：`audio/convert.py` 通过 `ffmpeg` 转 `.ogg`；`audio/analyze.py` 通过 `librosa` 提取 BPM、beat、onset、duration 和初步 offset。
- **特征层**：`features/bars.py` 将 raw analysis 映射为 4/4、每小节 16 格的 `BarFeature`；`features/sections.py` 根据位置和 energy 标记 intro / outro / chorus / verse / break。
- **数据模型层**：`tja/model.py` 定义 `SongAnalysis`、`BarFeature`、`ChartMetadata`、`ChartBar`、`TjaChart`，是 JSON 输出、AI 输入和 TJA writer 之间的稳定中间表示。
- **谱面生成层**：`rules/fallback_generator.py` 提供不依赖 AI 的规则生成器；`ai/prompts.py` 和 `ai/client.py` 提供 LiteLLM prompt、调用入口和 AI 输出清洗，AI 失败时 CLI 必须回退到规则生成器。
- **TJA 层**：`tja/writer.py` 只负责把 `TjaChart` 渲染为文本；`tja/validator.py` 做 MVP 级格式检查。
- **工具层**：`utils/paths.py` 集中处理 UTF-8 JSON 写入和 Pydantic 模型序列化。

`generate` 当前数据流：输入音频 → `output/<stem>.ogg` → `analysis.json` → AI 或 fallback `ChartBar` → `song.tja` → `report.txt`。

## 工程约束

- `.tja` 生成不要只依赖自由文本拼接；应尽早建立结构化谱面表示，再由导出器负责格式化。
- 音频分析结果、AI 输入、AI 原始输出、校验结果和最终 `.tja` 应尽量可追踪，便于比较不同版本生成质量。
- AI 生成逻辑应保留 prompt / 参数 / 模型版本等信息，以便复现实验结果。
- 对外部音频分析工具或 AI SDK 的调用应集中封装，避免散落在业务流程中。
- 新增核心模块时，同时考虑其输入、输出、错误模型和可测试边界。
- 如果引入临时脚本，应明确其生命周期；稳定后迁移到正式模块或从仓库删除。
- Git 提交信息使用中文，简洁描述本次变更的目标或结果。

## CLAUDE.md 维护要求

在修改项目代码时，如果出现以下变化，必须更新本文件：

- 新增或变更常用开发命令。
- 新增技术栈、框架、包管理器或测试框架。
- 新增核心目录、流水线阶段或架构边界。
- 改变音频分析、AI 生成、校验或 `.tja` 导出的数据流。
- 引入必须安装的本地工具、系统依赖或外部服务配置。
- 确立新的工程约定、文件命名规则或持久化格式。

更新时只记录对未来维护者有帮助的信息，避免写入临时计划、已经能从代码中直接看出的文件清单，或泛泛的编码建议。
