---
name: verify
description: 通过真实 CLI 生成流程验证音频分析和谱面输出的运行时行为。
---

# 项目运行时验证

1. 使用 `src/tja_ai_chartgen/assets/taiko_don_16bit_44100.wav` 作为小体积真实输入。
2. 为每个场景使用独立临时输出目录，运行：

```powershell
tja-ai-chartgen generate <audio.wav> --title "Verify" --output-dir <temp-dir> --max-bars 1
```

3. 验证 BeatNet 路径时，创建临时 `BeatNet/BeatNet.py` 包并通过 `PYTHONPATH` 注入；让 `process()` 返回目标合法或异常数据，再增加 `--use-beatnet`。
4. 捕获 CLI 输出，并读取生成的 `analysis.json`、`report.txt` 或 `.tja` 中与变更相关的字段。
5. 至少运行一个相邻异常探测；所有验证产物放在系统临时目录，不写入项目 `output/`。

注意：短打击音 WAV 的 librosa 基线 BPM 可能较高，验证降级时应与同一次环境下未启用 BeatNet 的基线结果比较，不要硬编码 BPM。
