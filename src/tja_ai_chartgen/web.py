import json
from pathlib import Path
from shutil import copy2
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import get_meter_spec, validate_time_signature
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars, validate_density
from tja_ai_chartgen.rules.styles import STYLE_LEVELS, validate_style
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, ChartMetadata, SongAnalysis, TjaChart
from tja_ai_chartgen.tja.writer import read_tja_text, render_tja, write_tja_text
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_WEB_OUTPUT_DIR = Path("output/web")
WEB_ASSET_DIR = Path(__file__).resolve().parent / "assets"
WEB_SOUND_FILES = {
    "taiko_don_16bit_44100.wav": "don",
    "taiko_ka_16bit_44100.wav": "ka",
}
ALLOWED_WEB_NOTES = set("01234578")
DEFAULT_WEB_BALLOON_COUNT = 8
WEB_COURSE_OPTIONS = (
    ("Easy", "简单（Easy）"),
    ("Normal", "普通（Normal）"),
    ("Hard", "困难（Hard）"),
    ("Oni", "魔王（Oni）"),
)
WEB_STYLE_OPTIONS = (
    ("technical", "技巧（technical）"),
    ("stamina", "体力（stamina）"),
    ("hybrid", "综合（hybrid）"),
    ("performance", "演出（performance）"),
)


_PAGE_CSS = """
:root {
  color-scheme: dark;
  --bg: #101114;
  --bg-soft: #16181d;
  --surface: rgba(246, 240, 226, 0.08);
  --surface-strong: rgba(246, 240, 226, 0.13);
  --ink: #f6f0e2;
  --muted: #ada89a;
  --line: rgba(246, 240, 226, 0.16);
  --accent: #d6a85f;
  --accent-soft: rgba(214, 168, 95, 0.18);
  --danger: #e08a74;
  --shadow: 0 24px 80px rgba(4, 6, 10, 0.42);
  --radius-lg: 30px;
  --radius-md: 18px;
  --radius-sm: 12px;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  min-height: 100dvh;
  margin: 0;
  color: var(--ink);
  font-family: "Outfit", "Aptos", "Segoe UI", "Noto Sans SC", sans-serif;
  line-height: 1.6;
  background:
    radial-gradient(circle at 14% -12%, rgba(214, 168, 95, 0.24), transparent 34rem),
    radial-gradient(circle at 88% 8%, rgba(119, 133, 117, 0.28), transparent 32rem),
    linear-gradient(135deg, #101114 0%, #17181d 48%, #12100e 100%);
  text-rendering: optimizeLegibility;
}

body::before {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  content: "";
  opacity: 0.16;
  background-image:
    linear-gradient(rgba(246, 240, 226, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(246, 240, 226, 0.04) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at top, black 0%, transparent 68%);
}

body::after {
  position: fixed;
  inset: 0;
  z-index: 10;
  pointer-events: none;
  content: "";
  opacity: 0.045;
  background-image: radial-gradient(rgba(255, 255, 255, 0.55) 0.5px, transparent 0.5px);
  background-size: 3px 3px;
}

.skip-link {
  position: fixed;
  top: 1rem;
  left: 1rem;
  z-index: 20;
  padding: 0.7rem 1rem;
  color: #16130e;
  background: var(--accent);
  border-radius: 999px;
  transform: translateY(-180%);
  transition: transform 180ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.shell {
  width: min(1180px, calc(100% - 2rem));
  margin: 0 auto;
  padding: 1.25rem 0 4rem;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: clamp(2rem, 6vw, 5rem);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--ink);
  font-weight: 700;
  letter-spacing: -0.03em;
  text-decoration: none;
}

.brand-mark {
  display: grid;
  width: 2.4rem;
  height: 2.4rem;
  place-items: center;
  color: #17130d;
  background: var(--accent);
  border-radius: 0.8rem;
  box-shadow: 0 14px 34px rgba(214, 168, 95, 0.2);
}

.nav-note {
  color: var(--muted);
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(22rem, 1.1fr);
  gap: clamp(1.2rem, 4vw, 3rem);
  align-items: start;
}

.hero-copy {
  position: sticky;
  top: 1.5rem;
  padding-top: 1.3rem;
}

.eyebrow,
.field-hint,
.meta-label,
th {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  max-width: 10ch;
  margin-bottom: 1.1rem;
  font-size: clamp(3.4rem, 10vw, 7.8rem);
  font-weight: 800;
  letter-spacing: -0.085em;
  line-height: 0.84;
  text-wrap: balance;
}

h2 {
  margin-bottom: 1rem;
  font-size: clamp(1.7rem, 3vw, 2.45rem);
  font-weight: 760;
  letter-spacing: -0.055em;
  line-height: 1;
}

.lede {
  max-width: 34rem;
  color: #d7d0c0;
  font-size: clamp(1.05rem, 1.7vw, 1.24rem);
  text-wrap: pretty;
}

.panel,
.metric-card,
.table-wrap,
.code-card,
.notice {
  background: linear-gradient(145deg, rgba(246, 240, 226, 0.11), rgba(246, 240, 226, 0.055));
  border: 1px solid var(--line);
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(22px);
}

.panel {
  position: relative;
  overflow: hidden;
  padding: clamp(1.1rem, 3vw, 2rem);
  border-radius: var(--radius-lg);
}

.panel::before {
  position: absolute;
  inset: 0 auto auto 10%;
  width: 65%;
  height: 1px;
  content: "";
  background: linear-gradient(90deg, transparent, rgba(214, 168, 95, 0.72), transparent);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.2rem;
}

.field {
  display: grid;
  gap: 0.45rem;
}

.field-wide {
  grid-column: 1 / -1;
}

.title-artist-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 0.75rem;
  align-items: end;
}

.swap-button {
  min-width: 3.15rem;
  padding: 0.8rem 0.9rem;
  color: var(--ink);
  background: rgba(246, 240, 226, 0.09);
  border: 1px solid rgba(246, 240, 226, 0.16);
  box-shadow: none;
}

.swap-button:hover {
  background: rgba(214, 168, 95, 0.18);
  box-shadow: none;
}

label {
  color: var(--ink);
  font-weight: 650;
}

input,
select,
button,
textarea {
  font: inherit;
}

input,
select,
textarea {
  width: 100%;
  min-height: 3.15rem;
  padding: 0.78rem 0.9rem;
  color: var(--ink);
  background: rgba(12, 13, 16, 0.62);
  border: 1px solid rgba(246, 240, 226, 0.18);
  border-radius: var(--radius-sm);
  outline: none;
  transition:
    border-color 180ms ease,
    box-shadow 180ms ease,
    background 180ms ease;
}

input[type="file"] {
  padding: 0.72rem;
}

input[type="checkbox"] {
  width: 1.1rem;
  min-height: auto;
  accent-color: var(--accent);
}

input:focus,
select:focus,
button:focus-visible,
a:focus-visible {
  outline: 3px solid rgba(214, 168, 95, 0.36);
  outline-offset: 3px;
}

input:focus,
select:focus,
textarea:focus {
  background: rgba(12, 13, 16, 0.82);
  border-color: rgba(214, 168, 95, 0.78);
  box-shadow: 0 0 0 4px rgba(214, 168, 95, 0.11);
}

.checkbox-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-height: 3.15rem;
  padding: 0.78rem 0.9rem;
  background: rgba(12, 13, 16, 0.44);
  border: 1px solid rgba(246, 240, 226, 0.12);
  border-radius: var(--radius-sm);
}

.advanced-panel {
  padding: 0.78rem 0.9rem;
  background: rgba(12, 13, 16, 0.32);
  border: 1px solid rgba(246, 240, 226, 0.12);
  border-radius: var(--radius-sm);
}

.advanced-panel summary {
  cursor: pointer;
  color: var(--ink);
  font-weight: 700;
}

.advanced-panel .form-grid {
  margin-top: 1rem;
}

.inline-debug-card {
  display: grid;
  gap: 0.85rem;
  padding: 1rem;
  margin-top: 1.2rem;
  background: rgba(5, 6, 8, 0.36);
  border: 1px solid rgba(214, 168, 95, 0.22);
  border-radius: var(--radius-md);
}

.inline-debug-card h2 {
  margin-bottom: 0;
  font-size: clamp(1.25rem, 2vw, 1.65rem);
}

.inline-debug-card p {
  max-width: 54rem;
  margin-bottom: 0;
  color: #d7d0c0;
}

button,
.button-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 3.2rem;
  padding: 0.8rem 1.15rem;
  color: #16130d;
  font-weight: 800;
  letter-spacing: -0.02em;
  text-decoration: none;
  cursor: pointer;
  background: linear-gradient(180deg, #e5bd76, #c99140);
  border: 0;
  border-radius: 999px;
  box-shadow: 0 16px 34px rgba(214, 168, 95, 0.22);
  transition:
    transform 180ms ease,
    box-shadow 180ms ease,
    filter 180ms ease;
}

button:hover,
.button-link:hover {
  filter: brightness(1.04);
  box-shadow: 0 20px 44px rgba(214, 168, 95, 0.3);
  transform: translateY(-1px);
}

button:active,
.button-link:active {
  transform: translateY(1px) scale(0.985);
}

form[data-loading="true"] button[type="submit"] {
  color: rgba(22, 19, 13, 0.72);
  cursor: progress;
}

form[data-loading="true"] button[type="submit"]::after {
  width: 0.85rem;
  height: 0.85rem;
  margin-left: 0.7rem;
  content: "";
  border: 2px solid rgba(22, 19, 13, 0.22);
  border-top-color: #16130d;
  border-radius: 999px;
  animation: spin 720ms linear infinite;
}

@keyframes spin {
  to { transform: rotate(1turn); }
}

.helper-strip,
.summary-head,
.result-path {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
}

.helper-strip {
  margin-top: 1rem;
  color: var(--muted);
  font-size: 0.92rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  min-height: 1.8rem;
  padding: 0.25rem 0.58rem;
  color: #f8e7bf;
  background: rgba(214, 168, 95, 0.11);
  border: 1px solid rgba(214, 168, 95, 0.24);
  border-radius: 0.6rem;
}

.steps {
  display: grid;
  gap: 0.85rem;
  max-width: 30rem;
  padding: 0;
  margin: 2rem 0 0;
  list-style: none;
}

.steps li {
  display: grid;
  grid-template-columns: 2.25rem 1fr;
  gap: 0.85rem;
  align-items: start;
  color: #d7d0c0;
}

.step-no {
  display: grid;
  width: 2.25rem;
  height: 2.25rem;
  place-items: center;
  color: var(--accent);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  background: rgba(214, 168, 95, 0.1);
  border-radius: 0.75rem;
}

.stack {
  display: grid;
  gap: 1.25rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 1.2rem 0;
}

.metric-card {
  padding: 1rem;
  border-radius: var(--radius-md);
}

.metric-value {
  display: block;
  margin-top: 0.35rem;
  font-size: clamp(1.35rem, 3vw, 2rem);
  font-weight: 790;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.045em;
}

code,
pre,
td:first-child,
.metric-value {
  font-family: "Cascadia Mono", "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

code {
  padding: 0.18rem 0.38rem;
  color: #f6d99c;
  word-break: break-all;
  background: rgba(12, 13, 16, 0.66);
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: 0.45rem;
}

.table-wrap {
  overflow: hidden;
  border-radius: var(--radius-md);
}

.table-scroll {
  overflow-x: auto;
}

table {
  width: 100%;
  min-width: 42rem;
  border-collapse: collapse;
}

th,
td {
  padding: 0.82rem 0.9rem;
  text-align: left;
  border-bottom: 1px solid rgba(246, 240, 226, 0.09);
}

tbody tr:hover {
  background: rgba(214, 168, 95, 0.06);
}

td {
  color: #e7dfcf;
  font-variant-numeric: tabular-nums;
}

pre {
  max-height: 36rem;
  padding: 1.1rem;
  margin: 0;
  overflow: auto;
  color: #f4e8d0;
  background: rgba(5, 6, 8, 0.82);
  border-radius: var(--radius-md);
}

.code-card {
  display: grid;
  gap: 0.8rem;
  padding: 1rem;
  border-radius: var(--radius-lg);
}

.audio-card audio {
  width: 100%;
  margin-top: 0.8rem;
  accent-color: var(--accent);
}

.play-preview {
  display: grid;
  gap: 1rem;
  padding: 0;
  overflow: hidden;
  background: #1f2020;
  border: 1px solid rgba(246, 240, 226, 0.16);
  border-radius: var(--radius-lg);
  box-shadow: 0 26px 90px rgba(2, 3, 4, 0.56);
}

.preview-topline,
.preview-controls,
.preview-side,
.timeline-head,
.timeline-row-label {
  font-variant-numeric: tabular-nums;
}

.preview-topline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  color: #d8d4c9;
  background: #2b2c2c;
  border-bottom: 1px solid rgba(246, 240, 226, 0.09);
}

.preview-tabs {
  display: inline-flex;
  gap: 0.35rem;
}

.preview-tab {
  padding: 0.42rem 0.72rem;
  color: #dedbd2;
  background: #3a3a39;
  border-radius: 0.35rem 0.35rem 0 0;
}

.preview-tab.is-active {
  color: #17130d;
  background: #d7cdb9;
}

.preview-status {
  color: var(--muted);
  font-size: 0.86rem;
}

.preview-stage-grid {
  display: grid;
  grid-template-columns: minmax(11rem, 0.25fr) minmax(24rem, 1fr) minmax(11rem, 0.25fr);
  gap: 0.8rem;
  padding: 0 0.8rem;
}

.preview-side {
  display: grid;
  gap: 0.75rem;
  align-content: start;
  color: #ddd8cc;
}

.inspector-group {
  overflow: hidden;
  background: #252626;
  border: 1px solid rgba(246, 240, 226, 0.09);
  border-radius: 0.55rem;
}

.inspector-title {
  padding: 0.45rem 0.65rem;
  font-weight: 760;
  background: #343534;
}

.inspector-line {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.48rem 0.65rem;
  color: #ccc7ba;
  border-top: 1px solid rgba(246, 240, 226, 0.07);
}

.inspector-line span:last-child {
  color: #f3ead8;
  text-align: right;
}

.preview-stage {
  display: grid;
  gap: 0.7rem;
  align-content: center;
  min-height: 25rem;
}

.taiko-lane {
  position: relative;
  height: clamp(10rem, 22vw, 15rem);
  overflow: hidden;
  background: #242626;
  border: 0.42rem solid #050505;
  box-shadow: inset 0 -3.1rem 0 #8b8b88, inset 0 -3.55rem 0 #050505;
}

.taiko-lane::before {
  position: absolute;
  top: 0;
  bottom: 3.55rem;
  left: 8.5rem;
  width: 2px;
  content: "";
  background: #bbb7aa;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.28);
}

.hit-ring {
  position: absolute;
  top: calc((100% - 3.55rem) / 2);
  left: 8.5rem;
  z-index: 3;
  width: 6.6rem;
  height: 6.6rem;
  border: 0.48rem solid #f1eadc;
  border-radius: 999px;
  transform: translate(-50%, -50%);
  box-shadow: 0 0.35rem 0 rgba(0, 0, 0, 0.45);
}

.hit-ring::after {
  position: absolute;
  inset: 0.78rem;
  content: "";
  background: #332f28;
  border-radius: inherit;
}

.lane-notes {
  position: absolute;
  z-index: 4;
  inset: 0 0 3.55rem;
  contain: layout paint;
}

.drum-note {
  position: absolute;
  top: 50%;
  left: 0;
  z-index: 4;
  width: 4.55rem;
  height: 4.55rem;
  pointer-events: none;
  opacity: 0;
  border: 0.38rem solid #f1eadc;
  border-radius: 999px;
  transform: translate3d(-200vw, -50%, 0) translateX(-50%);
  box-shadow: 0 0.32rem 0 rgba(0, 0, 0, 0.45);
  will-change: transform, opacity;
}

.drum-note.big {
  width: 5.7rem;
  height: 5.7rem;
}

.drum-note.don,
.timeline-note.don { background: #fa4028; }
.drum-note.ka,
.timeline-note.ka { background: #46c1c4; }
.drum-note.roll,
.timeline-note.roll { background: #e6bb57; }
.drum-note.balloon,
.timeline-note.balloon { background: #9bd272; }
.drum-note.end,
.timeline-note.end { background: #d7d7d3; }

.drum-note.is-hit {
  filter: brightness(1.18);
  outline: 0.35rem solid rgba(246, 240, 226, 0.22);
}

.combo-readout {
  position: absolute;
  top: 1rem;
  right: 1.2rem;
  color: #f1eadc;
  font-size: clamp(1.7rem, 4vw, 3.4rem);
  font-weight: 850;
  letter-spacing: -0.06em;
  opacity: 0.9;
}

.preview-controls {
  display: grid;
  grid-template-columns: auto minmax(14rem, 1fr) auto;
  gap: 0.8rem;
  align-items: center;
  padding: 0 0.2rem;
}

.play-button {
  min-width: 7.5rem;
}

.time-readout {
  color: #d7d0c0;
  font-family: "Cascadia Mono", "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

.preview-seek {
  width: 100%;
  accent-color: var(--accent);
}

.preview-audio {
  display: none;
}

.timeline-panel {
  margin: 0 0.8rem 0.8rem;
  overflow: hidden;
  background: #1b1c1c;
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: var(--radius-md);
}

.timeline-head {
  display: grid;
  grid-template-columns: 10.5rem 1fr;
  min-height: 3.05rem;
  color: #d7d0c0;
  background: #242525;
  border-bottom: 1px solid rgba(246, 240, 226, 0.08);
}

.timeline-scale {
  position: relative;
  overflow: hidden;
  background-image: linear-gradient(90deg, rgba(246, 240, 226, 0.06) 1px, transparent 1px);
  background-size: 6.25% 100%;
}

.timeline-bar-mark {
  position: absolute;
  top: 0.35rem;
  bottom: 0;
  min-width: 5rem;
  color: #d7d0c0;
  font-size: 0.78rem;
  line-height: 1.15;
  pointer-events: none;
  transform: translateX(0.25rem);
}

.timeline-bar-mark::before {
  position: absolute;
  top: -0.35rem;
  bottom: -4.9rem;
  left: -0.25rem;
  width: 1px;
  content: "";
  background: rgba(246, 240, 226, 0.18);
}

.timeline-row {
  display: grid;
  grid-template-columns: 10.5rem 1fr;
  min-height: 5.7rem;
  border-bottom: 1px solid rgba(246, 240, 226, 0.07);
}

.timeline-row-label {
  display: grid;
  align-items: center;
  padding: 0 0.7rem;
  color: #d7d0c0;
  background: #202121;
  border-right: 1px solid rgba(246, 240, 226, 0.08);
}

.timeline-track {
  position: relative;
  overflow: hidden;
  background-image:
    linear-gradient(90deg, rgba(246, 240, 226, 0.08) 1px, transparent 1px),
    linear-gradient(180deg, rgba(246, 240, 226, 0.045) 1px, transparent 1px);
  background-size: 6.25% 100%, 100% 50%;
}

.timeline-note {
  position: absolute;
  top: 50%;
  width: 1.9rem;
  height: 1.9rem;
  opacity: 0;
  border: 0.22rem solid #f1eadc;
  border-radius: 999px;
  transform: translate(-50%, -50%);
}

.timeline-cursor {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  z-index: 4;
  width: 2px;
  pointer-events: none;
  background: #8ebc76;
}

.preview-empty {
  display: grid;
  min-height: 12rem;
  place-items: center;
  color: var(--muted);
}

.notice {
  padding: 1rem 1.1rem;
  border-radius: var(--radius-md);
}

.error {
  color: #ffe1d7;
  background: rgba(224, 138, 116, 0.12);
  border-color: rgba(224, 138, 116, 0.36);
}

.footer {
  margin-top: 4rem;
  color: var(--muted);
  font-size: 0.9rem;
}

@media (max-width: 860px) {
  .hero,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .hero-copy {
    position: static;
  }

  .form-grid,
  .preview-stage-grid,
  .preview-controls,
  .timeline-head,
  .timeline-row {
    grid-template-columns: 1fr;
  }

  .title-artist-grid {
    grid-template-columns: 1fr;
  }

  .swap-button {
    width: 100%;
  }

  .preview-side {
    display: none;
  }

  .timeline-row-label {
    min-height: 2rem;
    border-right: 0;
    border-bottom: 1px solid rgba(246, 240, 226, 0.08);
  }
}

@media (max-width: 560px) {
  .shell {
    width: min(100% - 1rem, 1180px);
    padding-top: 0.75rem;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .panel {
    border-radius: 22px;
  }
}
"""

_PAGE_SCRIPT = """
const NOTE_CLASSES = {
  '1': 'don',
  '2': 'ka',
  '3': 'don big',
  '4': 'ka big',
  '5': 'roll',
  '7': 'balloon',
  '8': 'end',
};

const NOTE_SOUND_URLS = {
  don: '/assets/taiko_don_16bit_44100.wav',
  ka: '/assets/taiko_ka_16bit_44100.wav',
};

function formatTime(seconds) {
  const value = Math.max(0, seconds || 0);
  const minutes = Math.floor(value / 60).toString().padStart(2, '0');
  const rest = (value % 60).toFixed(3).padStart(6, '0');
  return `${minutes}:${rest}`;
}

function noteClass(note) {
  return NOTE_CLASSES[note] || 'don';
}

function noteSoundKey(note) {
  if (note === '1' || note === '3') return 'don';
  if (note === '2' || note === '4') return 'ka';
  return null;
}

function parseSongFileName(filename) {
  const stem = (filename || '').replace(/\\.[^/.]+$/, '').trim();
  if (!stem) return { title: '', artist: '' };
  const separators = [' - ', ' – ', ' — ', '-', '–', '—'];
  for (const separator of separators) {
    const index = stem.indexOf(separator);
    if (index > 0 && index < stem.length - separator.length) {
      return {
        title: stem.slice(0, index).trim(),
        artist: stem.slice(index + separator.length).trim(),
      };
    }
  }
  return { title: stem, artist: '' };
}

function setupAnalyzeForm(form) {
  const audioInput = form.querySelector('[data-role="audio-input"]');
  const titleInput = form.querySelector('[data-role="title-input"]');
  const artistInput = form.querySelector('[data-role="artist-input"]');
  const swapButton = form.querySelector('[data-role="swap-title-artist"]');
  if (audioInput && titleInput && artistInput) {
    audioInput.addEventListener('change', () => {
      const file = audioInput.files && audioInput.files[0];
      if (!file) return;
      const parsed = parseSongFileName(file.name);
      titleInput.value = parsed.title;
      artistInput.value = parsed.artist;
    });
  }
  if (swapButton && titleInput && artistInput) {
    swapButton.addEventListener('click', () => {
      const title = titleInput.value;
      titleInput.value = artistInput.value;
      artistInput.value = title;
      titleInput.focus();
    });
  }
}

function setupGamePreview(root) {
  const dataElement = root.querySelector('script[type="application/json"]');
  if (!dataElement) return;
  const data = JSON.parse(dataElement.textContent || '{}');
  const audio = root.querySelector('audio');
  const playButton = root.querySelector('[data-role="play"]');
  const seek = root.querySelector('[data-role="seek"]');
  const readout = root.querySelector('[data-role="time"]');
  const lane = root.querySelector('[data-role="lane"]');
  const hitRing = root.querySelector('.hit-ring');
  const combo = root.querySelector('[data-role="combo"]');
  const timelineCursor = root.querySelector('[data-role="timeline-cursor"]');
  const timelineTrack = root.querySelector('[data-role="timeline-track"]');
  const timelineScale = root.querySelector('[data-role="timeline-scale"]');
  const start = data.startTime || 0;
  const end = data.endTime || Math.max(start + 1, ...data.notes.map((note) => note.time + 1));
  const notes = data.notes || [];
  const bars = data.bars || [];
  const pixelsPerSecond = 360;
  const hitEpsilonSeconds = 0.012;
  const noteHideAfterSeconds = 0.035;
  const soundLookAheadSeconds = 0.05;
  const soundPools = Object.fromEntries(
    Object.entries(NOTE_SOUND_URLS).map(([key, url]) => [
      key,
      Array.from({ length: 6 }, () => {
        const player = new Audio(url);
        player.preload = 'auto';
        player.volume = 0.85;
        return player;
      }),
    ]),
  );
  const soundPoolIndexes = { don: 0, ka: 0 };
  let animationFrame = null;
  let hitX = 0;
  let visibleAheadSeconds = 0;
  let activeIndexes = new Set();
  let nextSoundIndex = 0;
  let audioContext = null;
  let soundBuffers = {};
  let soundBuffersPromise = null;
  let scheduledSources = [];
  let scheduledFallbackTimers = [];

  seek.min = start.toString();
  seek.max = end.toString();
  seek.step = '0.001';
  seek.value = start.toString();
  audio.currentTime = start;

  window.addEventListener('resize', () => {
    measurePreview();
    update(audio.currentTime || start);
  });

  const noteNodes = notes.map((note) => {
    const node = document.createElement('span');
    node.className = `drum-note ${noteClass(note.type)}`;
    node.dataset.time = String(note.time);
    lane.appendChild(node);
    return { ...note, node };
  });

  const timelineNoteNodes = notes.map((note) => {
    const node = document.createElement('span');
    node.className = `timeline-note ${noteClass(note.type).replace(' big', '')}`;
    node.title = `第 ${note.bar} 小节 / 第 ${note.grid} 格`;
    timelineTrack.appendChild(node);
    return { ...note, node };
  });

  const timelineMarkNodes = bars.map((bar) => {
    const node = document.createElement('span');
    node.className = 'timeline-bar-mark';
    node.innerHTML = `${bar.index}<br>${formatTime(bar.time)}`;
    timelineScale.appendChild(node);
    return { ...bar, node };
  });

  function measurePreview() {
    const laneRect = lane.getBoundingClientRect();
    const hitRect = hitRing.getBoundingClientRect();
    hitX = hitRect.left + hitRect.width / 2 - laneRect.left;
    visibleAheadSeconds = Math.max(0.25, (lane.clientWidth + 120 - hitX) / pixelsPerSecond);
  }

  function setNoteHidden(index) {
    const note = noteNodes[index];
    if (!note) return;
    note.node.style.opacity = '0';
    note.node.style.transform = 'translate3d(-200vw, -50%, 0) translateX(-50%)';
    note.node.classList.remove('is-hit');
  }

  function firstNoteIndexAtOrAfter(time) {
    let low = 0;
    let high = noteNodes.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (noteNodes[middle].time < time) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function syncSoundCursor(currentTime) {
    nextSoundIndex = firstNoteIndexAtOrAfter(currentTime);
  }

  function getAudioContext() {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return null;
    if (!audioContext) audioContext = new Context();
    return audioContext;
  }

  function prepareSoundBuffers() {
    const context = getAudioContext();
    if (!context) return Promise.resolve(null);
    if (!soundBuffersPromise) {
      soundBuffersPromise = Promise.all(
        Object.entries(NOTE_SOUND_URLS).map(([key, url]) =>
          fetch(url)
            .then((response) => response.arrayBuffer())
            .then((buffer) => context.decodeAudioData(buffer))
            .then((decoded) => {
              soundBuffers[key] = decoded;
            })
            .catch(() => {}),
        ),
      ).then(() => soundBuffers);
    }
    return soundBuffersPromise;
  }

  function stopScheduledSounds() {
    scheduledSources.forEach((source) => {
      try {
        source.stop();
      } catch (_error) {}
    });
    scheduledSources = [];
    scheduledFallbackTimers.forEach((timer) => window.clearTimeout(timer));
    scheduledFallbackTimers = [];
  }

  function playNoteSound(noteType, delaySeconds = 0, targetTime = null) {
    const soundKey = noteSoundKey(noteType);
    if (!soundKey) return;
    const context = getAudioContext();
    const buffer = soundBuffers[soundKey];
    if (context && buffer) {
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = buffer;
      gain.gain.value = 0.85;
      source.connect(gain).connect(context.destination);
      scheduledSources.push(source);
      source.onended = () => {
        scheduledSources = scheduledSources.filter((item) => item !== source);
      };
      source.start(context.currentTime + Math.max(0, delaySeconds));
      return;
    }

    const playFallback = () => {
      if (targetTime !== null && (audio.paused || Math.abs(audio.currentTime - targetTime) > 0.12)) return;
      const pool = soundPools[soundKey];
      if (!pool || pool.length === 0) return;
      const player = pool[soundPoolIndexes[soundKey] % pool.length];
      soundPoolIndexes[soundKey] += 1;
      player.currentTime = 0;
      player.play().catch(() => {});
    };
    if (delaySeconds > 0) {
      const timer = window.setTimeout(() => {
        scheduledFallbackTimers = scheduledFallbackTimers.filter((item) => item !== timer);
        playFallback();
      }, delaySeconds * 1000);
      scheduledFallbackTimers.push(timer);
    } else {
      playFallback();
    }
  }

  function playDueSounds(currentTime) {
    if (audio.paused) return;
    const soundUntil = currentTime + soundLookAheadSeconds;
    while (nextSoundIndex < noteNodes.length && noteNodes[nextSoundIndex].time <= soundUntil) {
      const note = noteNodes[nextSoundIndex];
      playNoteSound(note.type, note.time - currentTime, note.time);
      nextSoundIndex += 1;
    }
  }

  function updateTimeline(currentTime) {
    const windowSeconds = 6.6;
    const windowStart = Math.max(start, Math.min(currentTime - 0.35, end - windowSeconds));
    const windowEnd = Math.min(end, windowStart + windowSeconds);
    const windowDuration = Math.max(0.001, windowEnd - windowStart);
    const cursorProgress = Math.min(1, Math.max(0, (currentTime - windowStart) / windowDuration));
    timelineCursor.style.left = `${cursorProgress * 100}%`;

    timelineMarkNodes.forEach((bar) => {
      const barTime = Number(bar.time);
      const inWindow = barTime >= windowStart && barTime <= windowEnd;
      bar.node.style.opacity = inWindow ? '1' : '0';
      if (inWindow) {
        bar.node.style.left = `${((barTime - windowStart) / windowDuration) * 100}%`;
      }
    });

    timelineNoteNodes.forEach((note) => {
      const noteTime = Number(note.time);
      const inWindow = noteTime >= windowStart && noteTime <= windowEnd;
      note.node.style.opacity = inWindow ? '1' : '0';
      if (inWindow) {
        note.node.style.left = `${((noteTime - windowStart) / windowDuration) * 100}%`;
      }
    });
  }

  function update(currentTime) {
    seek.value = currentTime.toString();
    readout.textContent = `${formatTime(currentTime)} / ${formatTime(end)}`;
    updateTimeline(currentTime);

    const nextActiveIndexes = new Set();
    let hitCount = 0;
    for (let index = 0; index < noteNodes.length; index += 1) {
      const note = noteNodes[index];
      const delta = note.time - currentTime;
      if (delta < -noteHideAfterSeconds) {
        hitCount += 1;
        continue;
      }
      if (delta > visibleAheadSeconds) {
        continue;
      }
      const x = hitX + delta * pixelsPerSecond;
      note.node.style.opacity = '1';
      note.node.style.transform = `translate3d(${x}px, -50%, 0) translateX(-50%)`;
      note.node.classList.toggle('is-hit', Math.abs(delta) < 0.055);
      nextActiveIndexes.add(index);
    }
    activeIndexes.forEach((index) => {
      if (!nextActiveIndexes.has(index)) setNoteHidden(index);
    });
    activeIndexes = nextActiveIndexes;
    combo.textContent = hitCount ? `${hitCount}` : '';
  }

  function tick() {
    if (audio.currentTime >= end) {
      audio.pause();
      audio.currentTime = end;
    }
    update(audio.currentTime);
    playDueSounds(audio.currentTime);
    animationFrame = requestAnimationFrame(tick);
  }

  function startLoop() {
    if (animationFrame === null) {
      animationFrame = requestAnimationFrame(tick);
    }
  }

  playButton.addEventListener('click', async () => {
    if (audio.paused) {
      if (audio.currentTime < start || audio.currentTime >= end) audio.currentTime = start;
      await audio.play();
    } else {
      audio.pause();
    }
  });

  audio.addEventListener('play', () => {
    playButton.textContent = '暂停';
    const context = getAudioContext();
    if (context) {
      prepareSoundBuffers();
      context.resume().catch(() => {});
    }
    syncSoundCursor(audio.currentTime);
    startLoop();
  });
  audio.addEventListener('pause', () => {
    playButton.textContent = '播放';
    stopScheduledSounds();
  });
  audio.addEventListener('loadedmetadata', () => {
    if (audio.currentTime < start) audio.currentTime = start;
    syncSoundCursor(audio.currentTime);
    measurePreview();
    update(audio.currentTime);
  });
  seek.addEventListener('input', () => {
    stopScheduledSounds();
    audio.currentTime = Number(seek.value);
    syncSoundCursor(audio.currentTime);
    update(audio.currentTime);
  });

  prepareSoundBuffers();
  measurePreview();
  syncSoundCursor(start);
  update(start);
  startLoop();
}

document.querySelectorAll('[data-analyze-form]').forEach(setupAnalyzeForm);
document.querySelectorAll('[data-game-preview]').forEach(setupGamePreview);

document.querySelectorAll('form').forEach((form) => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    form.dataset.loading = 'true';
    form.setAttribute('aria-busy', 'true');
    if (button && button.dataset.loadingText) {
      button.textContent = button.dataset.loadingText;
    }
  });
});
"""


def create_app(output_dir: Path = DEFAULT_WEB_OUTPUT_DIR) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="tja-ai-chartgen Web UI")
    app.state.output_dir = output_dir

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _page("tja-ai-chartgen", _analysis_form())

    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze(
        audio: Annotated[UploadFile, File()],
        title: Annotated[str, Form()],
        artist: Annotated[str, Form()] = "",
        max_bars: Annotated[int | None, Form()] = None,
        bpm: Annotated[float | None, Form()] = None,
        offset: Annotated[float | None, Form()] = None,
        time_signature: Annotated[str, Form()] = "",
        use_beatnet: Annotated[bool, Form()] = False,
    ) -> HTMLResponse:
        try:
            job_dir = _new_job_dir(app.state.output_dir)
            input_path = _save_upload(job_dir, audio)
            ogg_path = job_dir / f"{input_path.stem}.ogg"
            convert_to_ogg(input_path, ogg_path)

            raw = analyze_audio(ogg_path, use_beatnet=use_beatnet)
            raw = apply_analysis_overrides(raw, bpm=bpm, offset=offset)
            if time_signature:
                validate_time_signature(time_signature)
                raw = raw.model_copy(update={"time_signature": time_signature})
            bars = assign_sections(build_bar_features(raw, max_bars=max_bars))
            analysis = SongAnalysis(
                title=title,
                artist=artist or None,
                audio_file=str(input_path),
                ogg_file=str(ogg_path),
                bpm=raw.bpm,
                offset=raw.offset,
                time_signature=raw.time_signature,
                bars=bars,
            )
            write_json(job_dir / "analysis.json", analysis)
            chart_bars = generate_fallback_chart_bars(bars)
            chart = TjaChart(
                metadata=ChartMetadata(
                    title=analysis.title,
                    artist=analysis.artist,
                    wave=Path(analysis.ogg_file).name,
                    bpm=analysis.bpm,
                    offset=analysis.offset,
                ),
                bars=chart_bars,
            )
            tja_text = render_tja(chart)
            output_path = job_dir / "preview.tja"
            write_tja_text(output_path, tja_text)
            return HTMLResponse(
                _page(
                    "Game preview",
                    _result_panel(
                        job_id=job_dir.name,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=chart.metadata.course,
                        level=chart.metadata.level,
                    )
                    + _regenerate_form(job_dir.name, len(analysis.bars), chart.metadata.course),
                )
            )
        except Exception as error:  # noqa: BLE001 - Web boundary returns a readable error page.
            return HTMLResponse(
                _page("Analysis failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.get("/assets/{filename}")
    async def web_asset(filename: str) -> FileResponse:
        safe_filename = Path(filename).name
        if safe_filename not in WEB_SOUND_FILES:
            raise HTTPException(status_code=404, detail=f"Web asset not found: {filename}")
        path = WEB_ASSET_DIR / safe_filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Web asset not found: {filename}")
        return FileResponse(path)

    @app.post("/preview-tja", response_class=HTMLResponse)
    async def preview_tja(
        tja: Annotated[UploadFile, File()],
        audio: Annotated[UploadFile, File()],
    ) -> HTMLResponse:
        try:
            job_dir = _new_job_dir(app.state.output_dir)
            tja_path = _save_upload(job_dir, tja)
            ogg_path = _save_ogg_upload(job_dir, audio)
            tja_text = read_tja_text(tja_path)
            analysis, chart_bars, course, level = _parse_tja_preview(
                tja_text,
                audio_file=ogg_path,
                ogg_file=ogg_path,
            )
            write_json(job_dir / "analysis.json", analysis)
            return HTMLResponse(
                _page(
                    "Game preview",
                    _result_panel(
                        job_id=job_dir.name,
                        output_path=tja_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    )
                    + _regenerate_form(job_dir.name, len(analysis.bars), course),
                )
            )
        except (UnicodeDecodeError, ValueError) as error:
            return HTMLResponse(
                _page("TJA preview failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.get("/jobs/{job_id}/{filename}")
    async def job_file(job_id: str, filename: str) -> FileResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        path = job_dir / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Job file not found: {filename}")
        return FileResponse(path)

    @app.post("/regenerate", response_class=HTMLResponse)
    async def regenerate(
        job_id: Annotated[str, Form()],
        start_bar: Annotated[int, Form()],
        end_bar: Annotated[int, Form()],
        course: Annotated[str, Form()] = "Oni",
        level: Annotated[int, Form()] = 10,
        style: Annotated[str, Form()] = "technical",
        density: Annotated[str, Form()] = "auto",
        special_notes: Annotated[bool, Form()] = False,
        use_ai: Annotated[bool, Form()] = False,
        ai_model: Annotated[str, Form()] = "",
        ai_base_url: Annotated[str, Form()] = "",
        ai_api_key: Annotated[str, Form()] = "",
        ai_repair_retries: Annotated[int, Form()] = 2,
    ) -> HTMLResponse:
        try:
            validate_density(density)
            validate_style(style)
            if ai_repair_retries < 0:
                raise ValueError("AI repair retries must be greater than or equal to 0")
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            selected_bars = _select_bars(analysis, start_bar, end_bar)
            chart_bars = None
            ai_failure: str | None = None
            if use_ai:
                try:
                    chart_bars = _generate_ai_chart_bars_for_web(
                        job_dir=job_dir,
                        analysis=analysis,
                        selected_bars=selected_bars,
                        start_bar=start_bar,
                        end_bar=end_bar,
                        course=course,
                        level=level,
                        style=style,
                        density=density,
                        model=_optional_form_text(ai_model),
                        api_base=_optional_form_text(ai_base_url),
                        api_key=_optional_form_text(ai_api_key),
                        ai_repair_retries=ai_repair_retries,
                        special_notes=special_notes,
                    )
                except Exception as error:  # noqa: BLE001 - Web UI should still render a usable draft.
                    ai_failure = str(error)

            if chart_bars is None:
                chart_bars = generate_fallback_chart_bars(
                    selected_bars,
                    style=style,
                    density=density,
                    special_notes=special_notes,
                )
            chart = TjaChart(
                metadata=ChartMetadata(
                    title=analysis.title,
                    artist=analysis.artist,
                    wave=Path(analysis.ogg_file).name,
                    bpm=analysis.bpm,
                    offset=analysis.offset,
                    course=course,
                    level=level,
                ),
                bars=chart_bars,
            )
            tja_text = render_tja(chart)
            output_path = job_dir / f"regenerated_{start_bar}_{end_bar}.tja"
            write_tja_text(output_path, tja_text)
            body = "".join(
                [
                    _ai_generation_notice(ai_failure) if use_ai else "",
                    _result_panel(
                        job_id=job_id,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(job_id, len(analysis.bars), course),
                ]
            )
            return HTMLResponse(_page("Regenerated bars", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Regeneration failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.post("/export-chart", response_class=HTMLResponse)
    async def export_chart(
        job_id: Annotated[str, Form()],
        tja_filename: Annotated[str, Form()],
        output_dir: Annotated[str, Form()],
    ) -> HTMLResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            tja_path = _job_file_path(job_dir, tja_filename)
            tja_text = read_tja_text(tja_path)
            _export_chart_files(tja_path=tja_path, analysis=analysis, output_dir=Path(output_dir))
            _parsed_analysis, chart_bars, course, level = _parse_tja_preview(
                tja_text,
                audio_file=Path(analysis.audio_file),
                ogg_file=Path(analysis.ogg_file),
            )
            body = "".join(
                [
                    _export_success_notice(Path(output_dir), Path(analysis.ogg_file).stem),
                    _result_panel(
                        job_id=job_id,
                        output_path=tja_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(job_id, len(analysis.bars), course),
                ]
            )
            return HTMLResponse(_page("Exported chart", body))
        except (FileExistsError, FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Export failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.post("/save-chart", response_class=HTMLResponse)
    async def save_chart(request: Request) -> HTMLResponse:
        try:
            form = await request.form()
            job_id = str(form.get("job_id", ""))
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            course = str(form.get("course", "Oni"))
            level = int(str(form.get("level", "10")))
            chart_bars = _edited_chart_bars_from_form(form)
            chart = TjaChart(
                metadata=ChartMetadata(
                    title=analysis.title,
                    artist=analysis.artist,
                    wave=Path(analysis.ogg_file).name,
                    bpm=analysis.bpm,
                    offset=analysis.offset,
                    course=course,
                    level=level,
                ),
                bars=chart_bars,
            )
            tja_text = render_tja(chart)
            output_path = job_dir / _edited_output_filename(chart_bars)
            write_tja_text(output_path, tja_text)
            body = "".join(
                [
                    _result_panel(
                        job_id=job_id,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(job_id, len(analysis.bars), course),
                ]
            )
            return HTMLResponse(_page("Saved chart edits", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Save failed", _error_notice(str(error))),
                status_code=400,
            )

    return app


app = create_app()


def _new_job_dir(output_dir: Path) -> Path:
    job_dir = output_dir / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _job_dir(output_dir: Path, job_id: str) -> Path:
    if not job_id.isalnum():
        raise ValueError("Invalid job id")
    job_dir = output_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return job_dir


def _save_upload(job_dir: Path, audio: UploadFile) -> Path:
    filename = Path(audio.filename or "upload.audio").name
    path = job_dir / filename
    content = audio.file.read()
    if not content:
        raise ValueError("Uploaded audio is empty")
    path.write_bytes(content)
    return path


def _save_ogg_upload(job_dir: Path, audio: UploadFile) -> Path:
    filename = Path(audio.filename or "upload.ogg").name
    if Path(filename).suffix.lower() != ".ogg":
        raise ValueError("TJA preview audio must be an .ogg file")
    return _save_upload(job_dir, audio)


def _job_file_path(job_dir: Path, filename: str) -> Path:
    safe_filename = Path(filename).name
    if not safe_filename:
        raise ValueError("Output filename is required")
    path = job_dir / safe_filename
    if not path.is_file():
        raise FileNotFoundError(f"Job file not found: {filename}")
    return path


def _export_chart_files(*, tja_path: Path, analysis: SongAnalysis, output_dir: Path) -> tuple[Path, Path]:
    if not str(output_dir).strip():
        raise ValueError("Output directory is required")
    ogg_path = Path(analysis.ogg_file)
    if not ogg_path.is_file():
        raise FileNotFoundError(f"OGG file not found: {ogg_path}")
    if tja_path.suffix.lower() != ".tja":
        raise ValueError("Only .tja files can be exported")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = ogg_path.stem
    export_ogg_path = output_dir / f"{output_stem}.ogg"
    export_tja_path = output_dir / f"{output_stem}.tja"
    existing_paths = [path for path in (export_ogg_path, export_tja_path) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Export target already exists: {existing}")

    copy2(ogg_path, export_ogg_path)
    copy2(tja_path, export_tja_path)
    return export_ogg_path, export_tja_path


def _select_bars(analysis: SongAnalysis, start_bar: int, end_bar: int):
    if start_bar < 1:
        raise ValueError("start_bar must be greater than or equal to 1")
    if end_bar < start_bar:
        raise ValueError("end_bar must be greater than or equal to start_bar")
    selected = [bar for bar in analysis.bars if start_bar - 1 <= bar.index <= end_bar - 1]
    if not selected:
        raise ValueError("No bars selected")
    return selected


def _edited_chart_bars_from_form(form) -> list[ChartBar]:
    bar_count = int(str(form.get("bar_count", "0")))
    if bar_count < 1:
        raise ValueError("No editable bars submitted")

    chart_bars: list[ChartBar] = []
    for position in range(bar_count):
        notes = str(form.get(f"notes_{position}", ""))
        expected_length = int(str(form.get(f"grids_per_bar_{position}", len(notes))))
        if len(notes) != expected_length:
            raise ValueError(
                f"notes_{position} must contain exactly {expected_length} character(s), got {len(notes)}"
            )
        illegal_characters = sorted(set(notes) - ALLOWED_WEB_NOTES)
        if illegal_characters:
            raise ValueError(f"notes_{position} contains illegal character(s): {''.join(illegal_characters)}")

        time_signature = str(form.get(f"time_signature_{position}", "4/4"))
        validate_time_signature(time_signature)
        chart_bars.append(
            ChartBar(
                index=int(str(form.get(f"bar_index_{position}", position))),
                notes=notes,
                time_signature=time_signature,
                balloon_counts=_parse_balloon_counts(
                    str(form.get(f"balloon_counts_{position}", "")),
                    notes.count("7"),
                ),
            )
        )

    return chart_bars


def _parse_balloon_counts(raw_value: str, expected_count: int) -> list[int]:
    if expected_count == 0:
        return []
    if not raw_value.strip():
        return [DEFAULT_WEB_BALLOON_COUNT] * expected_count

    counts: list[int] = []
    for raw_count in raw_value.split(","):
        value = int(raw_count.strip())
        if value < 1:
            raise ValueError("balloon counts must be positive integers")
        counts.append(value)

    if len(counts) != expected_count:
        raise ValueError(f"balloon counts must contain exactly {expected_count} value(s)")
    return counts


def _edited_output_filename(chart_bars: list[ChartBar]) -> str:
    first_bar = chart_bars[0].index + 1
    last_bar = chart_bars[-1].index + 1
    return f"edited_{first_bar}_{last_bar}.tja"


def _parse_tja_preview(
    tja_text: str,
    *,
    audio_file: Path,
    ogg_file: Path,
) -> tuple[SongAnalysis, list[ChartBar], str, int]:
    metadata: dict[str, str] = {}
    chart_bars: list[ChartBar] = []
    in_chart = False
    time_signature = "4/4"

    for raw_line in tja_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        upper_line = line.upper()
        if upper_line == "#START":
            in_chart = True
            continue
        if upper_line == "#END":
            break
        if not in_chart:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().upper()] = value.strip()
            continue
        if upper_line.startswith("#MEASURE"):
            time_signature = _time_signature_from_measure(upper_line)
            continue
        if upper_line.startswith("#"):
            continue

        for raw_notes in line.split(","):
            notes = "".join(character for character in raw_notes.strip() if not character.isspace())
            if not notes:
                continue
            illegal_characters = sorted(set(notes) - ALLOWED_WEB_NOTES)
            if illegal_characters:
                raise ValueError(f"TJA contains unsupported note character(s): {''.join(illegal_characters)}")
            chart_bars.append(
                ChartBar(
                    index=len(chart_bars),
                    notes=notes,
                    time_signature=time_signature,
                    balloon_counts=[DEFAULT_WEB_BALLOON_COUNT] * notes.count("7"),
                )
            )

    if not chart_bars:
        raise ValueError("TJA does not contain playable note bars")

    bpm = float(metadata.get("BPM", "120"))
    tja_offset = float(metadata.get("OFFSET", "0"))
    offset = -tja_offset
    title = metadata.get("TITLE", audio_file.stem)
    artist = metadata.get("SUBTITLE", "").removeprefix("-- ") or None
    course = metadata.get("COURSE", "Oni")
    level = int(float(metadata.get("LEVEL", "10")))
    bars = _bar_features_from_chart(chart_bars, bpm=bpm, offset=offset)
    analysis = SongAnalysis(
        title=title,
        artist=artist,
        audio_file=str(audio_file),
        ogg_file=str(ogg_file),
        bpm=bpm,
        offset=offset,
        time_signature=chart_bars[0].time_signature,
        bars=bars,
    )
    return analysis, chart_bars, course, level


def _time_signature_from_measure(measure_line: str) -> str:
    parts = measure_line.split(maxsplit=1)
    if len(parts) < 2:
        return "4/4"
    ratio = parts[1].strip()
    if ratio == "1/1":
        return "4/4"
    if ratio == "3/4":
        return "3/4"
    return "4/4"


def _bar_features_from_chart(chart_bars: list[ChartBar], *, bpm: float, offset: float) -> list[BarFeature]:
    current_time = offset
    features: list[BarFeature] = []
    for chart_bar in chart_bars:
        meter = get_meter_spec(chart_bar.time_signature)
        bar_length = meter.beats_per_bar * 60.0 / bpm
        note_grids = [index for index, note in enumerate(chart_bar.notes) if note != "0"]
        features.append(
            BarFeature(
                index=chart_bar.index,
                start_time=round(current_time, 6),
                end_time=round(current_time + bar_length, 6),
                energy=round(min(1.0, len(note_grids) / max(1, len(chart_bar.notes))), 3),
                time_signature=chart_bar.time_signature,
                grids_per_bar=len(chart_bar.notes),
                onset_16=note_grids,
                accent_16=[grid for grid in note_grids if grid in meter.accent_grids],
                beat_grids=sorted(meter.accent_grids),
                downbeat_grid=0,
                section="tja",
            )
        )
        current_time += bar_length
    return features


def _generate_ai_chart_bars_for_web(
    *,
    job_dir: Path,
    analysis: SongAnalysis,
    selected_bars: list[BarFeature],
    start_bar: int,
    end_bar: int,
    course: str,
    level: int,
    style: str,
    density: str,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    ai_repair_retries: int,
    special_notes: bool,
) -> list[ChartBar]:
    from tja_ai_chartgen.ai.client import generate_chart_bars_with_ai, sanitize_ai_bars
    from tja_ai_chartgen.ai.prompts import build_chart_generation_payload

    selected_analysis = analysis.model_copy(update={"bars": selected_bars})
    ai_input_path = job_dir / f"ai_input_{start_bar}_{end_bar}.json"
    ai_output_path = job_dir / f"ai_output_{start_bar}_{end_bar}.json"
    write_json(
        ai_input_path,
        build_chart_generation_payload(
            selected_analysis,
            course,
            level,
            style,
            density,
            special_notes=special_notes,
        ),
    )
    try:
        ai_bars, ai_output = generate_chart_bars_with_ai(
            selected_analysis,
            course,
            level,
            style,
            density,
            model,
            api_base=api_base,
            api_key=api_key,
            max_repair_attempts=ai_repair_retries,
            special_notes=special_notes,
        )
    except Exception as error:
        _write_web_ai_failure(ai_output_path, error)
        raise

    write_json(ai_output_path, ai_output)
    sanitized = sanitize_ai_bars(
        ai_bars,
        expected_count=len(selected_bars),
        expected_bars=selected_bars,
    )
    return _reindex_chart_bars(sanitized, selected_bars)


def _write_web_ai_failure(path: Path, error: Exception) -> None:
    output = getattr(error, "output", None)
    payload = {"error": str(error)}
    if isinstance(output, dict):
        payload.update(output)
    write_json(path, payload)


def _reindex_chart_bars(chart_bars: list[ChartBar], selected_bars: list[BarFeature]) -> list[ChartBar]:
    reindexed: list[ChartBar] = []
    for index, chart_bar in enumerate(chart_bars):
        feature = selected_bars[index]
        reindexed.append(
            ChartBar(
                index=feature.index,
                notes=chart_bar.notes,
                time_signature=feature.time_signature,
                balloon_counts=chart_bar.balloon_counts,
            )
        )
    return reindexed


def _optional_form_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _analysis_form() -> str:
    return f"""
<section class="hero" aria-labelledby="page-title">
  <div class="hero-copy">
    <p class="eyebrow">本地谱面工作台</p>
    <h1 id="page-title">tja-ai-chartgen</h1>
    <p class="lede">
      上传音频，先做节拍与小节分析，再挑选片段生成可检查的 TJA 草稿。
      这个界面适合快速试谱、修正 BPM/OFFSET，以及比较不同风格模板。
    </p>
    <ol class="steps" aria-label="生成流程">
      <li><span class="step-no">01</span><span>导入音频并转换为项目使用的 OGG 文件。</span></li>
      <li><span class="step-no">02</span><span>提取 BPM、OFFSET、拍号、小节能量和段落标签。</span></li>
      <li><span class="step-no">03</span><span>选择小节范围，重新生成局部 TJA 片段。</span></li>
    </ol>
  </div>

  <section class="panel" aria-labelledby="upload-heading">
    <p class="eyebrow">分析音频</p>
    <h2 id="upload-heading">上传并分析</h2>
    <form action="/analyze" enctype="multipart/form-data" method="post" data-analyze-form>
      <div class="form-grid">
        <label class="field field-wide">
          音频文件
          <input name="audio" type="file" accept="audio/*,video/mp4,.mp4,.m4s" data-role="audio-input" required>
          <span class="field-hint">mp3、wav、flac、m4s 等 ffmpeg 可读取的音频。文件名格式建议为：歌名 - 歌手.mp4。</span>
        </label>
        <div class="title-artist-grid field-wide">
          <label class="field">
            歌名
            <input name="title" placeholder="自动从文件名识别" data-role="title-input" required>
          </label>
          <button class="swap-button" type="button" data-role="swap-title-artist" aria-label="交换歌名和歌手">⇄</button>
          <label class="field">
            歌手
            <input name="artist" placeholder="可选" data-role="artist-input">
          </label>
        </div>
        <label class="field">
          最大小节数
          <input name="max_bars" type="number" min="1" placeholder="16">
        </label>
        <label class="field">
          BPM 覆盖
          <input name="bpm" type="number" step="0.001" min="0" placeholder="220.588">
        </label>
        <label class="field">
          OFFSET 覆盖
          <input name="offset" type="number" step="0.001" placeholder="0.725">
        </label>
        <label class="field">
          拍号
          <select name="time_signature">
            <option value="">使用分析结果</option>
            <option value="4/4">4/4</option>
            <option value="3/4">3/4</option>
            <option value="6/8">6/8</option>
          </select>
        </label>
        <label class="checkbox-card field-wide">
          <input name="use_beatnet" type="checkbox" value="true" checked>
          <span>使用 BeatNet <span class="field-hint">尝试增强强拍、拍号和 offset。</span></span>
        </label>
      </div>
      <div class="helper-strip">
        <button type="submit" data-loading-text="分析中">上传并分析</button>
        <span>可用风格：{', '.join(STYLE_LEVELS)}</span>
      </div>
    </form>
    {_tja_preview_form()}
  </section>
</section>
"""


def _tja_preview_form() -> str:
    return """
<section class="inline-debug-card" aria-labelledby="tja-preview-heading">
  <p class="eyebrow">调试预览</p>
  <h2 id="tja-preview-heading">直接播放 TJA</h2>
  <p>已有 `.tja` 时可以直接上传调试。这里只接受已经准备好的 OGG 音频，不会再调用 ffmpeg 转换或额外输出音频文件。</p>
  <form action="/preview-tja" enctype="multipart/form-data" method="post">
    <div class="form-grid">
      <label class="field">
        TJA 文件
        <input name="tja" type="file" accept=".tja,text/plain" required>
      </label>
      <label class="field">
        OGG 音频
        <input name="audio" type="file" accept=".ogg,audio/ogg" required>
      </label>
    </div>
    <div class="helper-strip">
      <button type="submit" data-loading-text="载入中">打开 TJA 预览</button>
      <span>用于快速定位谱面播放、对齐和滚动问题。</span>
    </div>
  </form>
</section>
"""


def _analysis_summary(analysis: SongAnalysis, analysis_path: Path, job_id: str) -> str:
    rows = "".join(
        f"<tr><td>{bar.index + 1}</td><td>{bar.start_time:.3f}</td><td>{bar.end_time:.3f}</td>"
        f"<td>{bar.energy:.3f}</td><td>{_escape(bar.section)}</td><td>{bar.grids_per_bar}</td></tr>"
        for bar in analysis.bars
    )
    return f"""
<section class="stack" aria-labelledby="analysis-heading">
  <div class="summary-head">
    <span class="badge">任务 <code>{_escape(job_id)}</code></span>
    <span class="badge">分析 JSON <code>{_escape(str(analysis_path))}</code></span>
  </div>
  <section class="panel">
    <p class="eyebrow">分析预览</p>
    <h1 id="analysis-heading">分析预览</h1>
    <p class="lede">分析结果已保存。确认 BPM、OFFSET 和小节数量后，可以只重生成需要调整的片段。</p>
    <div class="summary-grid" aria-label="分析指标">
      <article class="metric-card">
        <span class="meta-label">BPM: {analysis.bpm}</span>
        <span class="metric-value">{analysis.bpm:.3f}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">OFFSET: {analysis.offset}</span>
        <span class="metric-value">{analysis.offset:.3f}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">拍号</span>
        <span class="metric-value">{_escape(analysis.time_signature)}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">小节数</span>
        <span class="metric-value">{len(analysis.bars)}</span>
      </article>
    </div>
  </section>

  {_audio_preview(analysis, job_id)}

  <section class="table-wrap" aria-labelledby="bars-heading">
    <div class="panel">
      <p class="eyebrow">小节映射</p>
      <h2 id="bars-heading">小节预览</h2>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr><th>小节</th><th>开始</th><th>结束</th><th>能量</th><th>段落</th><th>格数</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
</section>
"""


def _regenerate_form(job_id: str, bar_count: int, course: str = "Oni") -> str:
    end_bar = max(1, bar_count)
    return f"""
<section class="panel" aria-labelledby="regen-heading">
  <p class="eyebrow">谱面片段</p>
  <h2 id="regen-heading">重新生成选中小节</h2>
  <p class="lede">选择起止小节、难度和模板，只输出这段片段，方便你逐步修正谱面。</p>
  <form action="/regenerate" method="post">
    <input name="job_id" type="hidden" value="{_escape(job_id)}">
    <div class="form-grid">
      <label class="field">
        起始小节
        <input name="start_bar" type="number" min="1" value="1" required>
      </label>
      <label class="field">
        结束小节
        <input name="end_bar" type="number" min="1" value="{end_bar}" required>
      </label>
      <label class="field">
        难度类型
        <select name="course">{_course_option_tags(course)}</select>
      </label>
      <label class="field">
        难度等级
        <input name="level" type="number" min="1" max="10" value="10">
      </label>
      <label class="field">
        风格
        <select name="style">{_style_option_tags("technical")}</select>
      </label>
      <label class="field">
        密度
        <select name="density">{_option_tags(("auto", "low", "medium", "high", "max"), "auto")}</select>
      </label>
      <label class="checkbox-card field-wide">
        <input name="special_notes" type="checkbox" value="true" checked>
        <span>特殊音符 <span class="field-hint">允许简单滚奏和气球。</span></span>
      </label>
      <label class="checkbox-card field-wide">
        <input name="use_ai" type="checkbox" value="true" checked>
        <span>使用 AI 增强 <span class="field-hint">优先调用 LiteLLM / OpenAI 兼容接口，失败时自动回退规则生成。</span></span>
      </label>
      <details class="advanced-panel field-wide">
        <summary>AI 参数</summary>
        <div class="form-grid">
          <label class="field">
            AI 模型
            <input name="ai_model" placeholder="留空读取 MODEL">
          </label>
          <label class="field">
            AI Base URL
            <input name="ai_base_url" placeholder="留空读取 OPENAI_BASE_URL">
          </label>
          <label class="field">
            AI API Key
            <input name="ai_api_key" type="password" autocomplete="off" placeholder="留空读取 OPENAI_API_KEY">
          </label>
          <label class="field">
            修复重试
            <input name="ai_repair_retries" type="number" min="0" value="2">
          </label>
        </div>
      </details>
    </div>
    <div class="helper-strip">
      <button type="submit" data-loading-text="重新生成中">重新生成</button>
      <span>当前 job：<code>{_escape(job_id)}</code>。API Key 只用于本次请求，不写入输出文件。</span>
    </div>
  </form>
</section>
"""


def _audio_preview(analysis: SongAnalysis, job_id: str) -> str:
    ogg_name = Path(analysis.ogg_file).name
    return f"""
  <section class="panel audio-card" aria-labelledby="audio-heading">
    <p class="eyebrow">音频预览</p>
    <h2 id="audio-heading">音频预览</h2>
    <p class="lede">播放转换后的 OGG，配合下方小节起止时间检查 OFFSET 与局部节奏。</p>
    <audio controls preload="metadata" src="/jobs/{_escape(job_id)}/{_escape(ogg_name)}"></audio>
  </section>
"""


def _game_preview(
    job_id: str,
    analysis: SongAnalysis,
    chart_bars: list[ChartBar],
    course: str,
    level: int,
) -> str:
    payload = _preview_payload(analysis, chart_bars)
    ogg_name = Path(analysis.ogg_file).name
    duration = max(0.001, payload["endTime"] - payload["startTime"])
    return f"""
  <section class="play-preview" data-game-preview aria-labelledby="game-preview-heading">
    <script type="application/json">{_json_script(payload)}</script>
    <div class="preview-topline">
      <div class="preview-tabs" aria-label="预览标签">
        <span class="preview-tab is-active">游玩预览</span>
        <span class="preview-tab">{_escape(course)} x{level}</span>
      </div>
      <span class="preview-status">{analysis.bpm:.3f} BPM · {duration:.3f}s · 自动演奏预览</span>
    </div>
    <div class="preview-stage-grid">
      <aside class="preview-side" aria-label="谱面属性">
        <div class="inspector-group">
          <div class="inspector-title">谱面</div>
          <div class="inspector-line"><span>歌名</span><span>{_escape(analysis.title)}</span></div>
          <div class="inspector-line"><span>制作者</span><span>tja-ai-chartgen</span></div>
          <div class="inspector-line"><span>Offset</span><span>{analysis.offset:.3f}s</span></div>
        </div>
        <div class="inspector-group">
          <div class="inspector-title">难度</div>
          <div class="inspector-line"><span>类型</span><span>{_escape(course)}</span></div>
          <div class="inspector-line"><span>等级</span><span>x{level}</span></div>
        </div>
      </aside>
      <div class="preview-stage">
        <div class="taiko-lane" aria-label="太鼓自动演奏预览">
          <span class="hit-ring" aria-hidden="true"></span>
          <span class="combo-readout" data-role="combo" aria-hidden="true"></span>
          <div class="lane-notes" data-role="lane"></div>
        </div>
        <div class="preview-controls">
          <button class="play-button" type="button" data-role="play">播放</button>
          <input class="preview-seek" data-role="seek" type="range" aria-label="谱面播放进度">
          <span class="time-readout" data-role="time">00:00.000 / 00:00.000</span>
          <audio class="preview-audio" preload="metadata" src="/jobs/{_escape(job_id)}/{_escape(ogg_name)}"></audio>
        </div>
      </div>
      <aside class="preview-side" aria-label="当前选择">
        <div class="inspector-group">
          <div class="inspector-title">说明</div>
          <div class="inspector-line"><span>播放</span><span>自动演奏</span></div>
          <div class="inspector-line"><span>拖动</span><span>查看任意位置</span></div>
          <div class="inspector-line"><span>输出</span><span>已保存 .tja</span></div>
        </div>
      </aside>
    </div>
    <div class="timeline-panel" aria-label="谱面时间线">
      <div class="timeline-head">
        <div class="timeline-row-label">谱面时间线</div>
        <div class="timeline-scale" data-role="timeline-scale"></div>
      </div>
      <div class="timeline-row">
        <div class="timeline-row-label">音符</div>
        <div class="timeline-track" data-role="timeline-track">
          <span class="timeline-cursor" data-role="timeline-cursor"></span>
        </div>
      </div>
    </div>
  </section>
"""


def _preview_payload(analysis: SongAnalysis, chart_bars: list[ChartBar]) -> dict:
    features = {bar.index: bar for bar in analysis.bars}
    selected_features = [features[bar.index] for bar in chart_bars if bar.index in features]
    start_time = selected_features[0].start_time if selected_features else 0.0
    end_time = selected_features[-1].end_time if selected_features else start_time + 1.0
    notes: list[dict[str, float | int | str]] = []
    bars: list[dict[str, float | int | str]] = []

    for chart_bar in chart_bars:
        feature = features.get(chart_bar.index)
        if feature is None:
            continue
        bar_duration = max(0.001, feature.end_time - feature.start_time)
        step = bar_duration / max(1, len(chart_bar.notes))
        bars.append(
            {
                "index": chart_bar.index + 1,
                "time": feature.start_time,
                "endTime": feature.end_time,
                "timeSignature": chart_bar.time_signature,
            }
        )
        for grid_index, note in enumerate(chart_bar.notes):
            if note == "0":
                continue
            notes.append(
                {
                    "time": feature.start_time + grid_index * step,
                    "type": note,
                    "bar": chart_bar.index + 1,
                    "grid": grid_index + 1,
                }
            )

    return {
        "startTime": start_time,
        "endTime": end_time,
        "notes": notes,
        "bars": bars,
    }


def _json_script(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _timeline_marks(payload: dict) -> str:
    start_time = float(payload["startTime"])
    end_time = float(payload["endTime"])
    duration = max(0.001, end_time - start_time)
    return "".join(
        f'<span class="timeline-bar-mark" style="left: {((float(bar["time"]) - start_time) / duration) * 100:.4f}%">'
        f'{bar["index"]}<br>{float(bar["time"]):.3f}</span>'
        for bar in payload["bars"]
    )


def _timeline_notes(payload: dict) -> str:
    start_time = float(payload["startTime"])
    end_time = float(payload["endTime"])
    duration = max(0.001, end_time - start_time)
    return "".join(
        f'<span class="timeline-note { _timeline_note_class(str(note["type"])) }" '
        f'style="left: {((float(note["time"]) - start_time) / duration) * 100:.4f}%" '
        f'title="Bar {note["bar"]} / grid {note["grid"]}"></span>'
        for note in payload["notes"]
    )


def _timeline_note_class(note: str) -> str:
    if note in {"1", "3"}:
        return "don"
    if note in {"2", "4"}:
        return "ka"
    if note == "5":
        return "roll"
    if note == "7":
        return "balloon"
    return "end"


def _result_panel(
    *,
    job_id: str,
    output_path: Path,
    tja_text: str,
    analysis: SongAnalysis,
    chart_bars: list[ChartBar],
    course: str,
    level: int,
) -> str:
    return f"""
<section class="stack" aria-labelledby="result-heading">
  <section class="panel">
    <p class="eyebrow">预览已生成</p>
    <h1 id="result-heading">游玩预览</h1>
    <p class="result-path">谱面已生成：<code>{_escape(str(output_path))}</code></p>
    <p class="lede">生成完成后直接进入可视化预览。点击播放自动演奏，拖动进度条查看任意位置，不再把大段 TJA 数值直接丢给用户。</p>
  </section>
  {_export_form(job_id, output_path, analysis)}
  {_game_preview(job_id, analysis, chart_bars, course, level)}
</section>
"""


def _export_form(job_id: str, output_path: Path, analysis: SongAnalysis) -> str:
    ogg_stem = Path(analysis.ogg_file).stem
    return f"""
  <section class="panel" aria-labelledby="export-heading">
    <p class="eyebrow">保存结果</p>
    <h2 id="export-heading">保存 OGG 和 TJA</h2>
    <p class="lede">保存当前预览使用的 OGG 和 TJA；输出文件名会统一为 <code>{_escape(ogg_stem)}.ogg</code> 和 <code>{_escape(ogg_stem)}.tja</code>。</p>
    <form action="/export-chart" method="post">
      <input name="job_id" type="hidden" value="{_escape(job_id)}">
      <input name="tja_filename" type="hidden" value="{_escape(output_path.name)}">
      <div class="form-grid">
        <label class="field field-wide">
          保存目录
          <input name="output_dir" placeholder="例如 D:\\Taiko\\Songs\\{_escape(ogg_stem)}" required>
        </label>
      </div>
      <div class="helper-strip">
        <button type="submit" data-loading-text="保存中">保存 OGG 和 TJA</button>
        <span>如果目标目录中已有同名文件，本次保存会停止并提示。</span>
      </div>
    </form>
  </section>
"""


def _export_success_notice(output_dir: Path, output_stem: str) -> str:
    return (
        '<p class="notice">已保存：'
        f'<code>{_escape(str(output_dir / f"{output_stem}.ogg"))}</code> 和 '
        f'<code>{_escape(str(output_dir / f"{output_stem}.tja"))}</code></p>'
    )


def _ai_generation_notice(ai_failure: str | None) -> str:
    if ai_failure:
        message = f"AI 增强失败，已自动回退到规则生成：{ai_failure}"
        return f'<p class="notice error">{_escape(message)}</p>'
    return '<p class="notice">AI 增强已完成，本次谱面来自 AI 输出校验后的结果。</p>'


def _error_notice(message: str) -> str:
    return f"""
<section class="panel" aria-labelledby="error-heading">
  <p class="eyebrow">请求失败</p>
  <h1 id="error-heading">无法完成操作</h1>
  <p class="notice error">{_escape(message)}</p>
  <a class="button-link" href="/">返回上传页面</a>
</section>
"""


def _option_tags(options, selected: str) -> str:
    return "".join(
        f'<option value="{_escape(option)}"{_selected_attr(option, selected)}>{_escape(option)}</option>'
        for option in options
    )


def _course_option_tags(selected: str) -> str:
    options = list(WEB_COURSE_OPTIONS)
    if selected and selected not in {value for value, _label in options}:
        options.append((selected, selected))
    return "".join(
        f'<option value="{_escape(value)}"{_selected_attr(value, selected)}>{_escape(label)}</option>'
        for value, label in options
    )


def _style_option_tags(selected: str) -> str:
    options = list(WEB_STYLE_OPTIONS)
    if selected and selected not in {value for value, _label in options}:
        options.append((selected, selected))
    return "".join(
        f'<option value="{_escape(value)}"{_selected_attr(value, selected)}>{_escape(label)}</option>'
        for value, label in options
    )


def _selected_attr(option: str, selected: str) -> str:
    return " selected" if option == selected else ""


def _page(title: str, body: str) -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="tja-ai-chartgen 本地 Web UI，用于分析音频并生成 TJA 谱面草稿。">
  <meta property="og:title" content="{_escape(title)}">
  <meta property="og:description" content="本地音频分析和 TJA 谱面草稿工作台。">
  <meta name="theme-color" content="#101114">
  <title>{_escape(title)}</title>
  <style>{_PAGE_CSS}</style>
</head>
<body>
  <a class="skip-link" href="#main">跳到主要内容</a>
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="tja-ai-chartgen 首页">
        <span class="brand-mark" aria-hidden="true">太</span>
        <span>tja-ai-chartgen</span>
      </a>
      <span class="nav-note">本地 Web 界面 · 不上传云端</span>
    </header>
    <main id="main">
      {body}
    </main>
    <footer class="footer">
      <p>本地工具界面。输出文件保存在配置的 <code>output/web</code> job 目录中。</p>
    </footer>
  </div>
  <script>{_PAGE_SCRIPT}</script>
</body>
</html>
"""


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
