import asyncio
import json
import os
import re
from _thread import LockType
from pathlib import Path
from shutil import copy2, rmtree
from threading import Event, Lock, Thread
from time import sleep
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError

from tja_ai_chartgen.cancellation import GenerationCancelledError, raise_if_cancelled
from tja_ai_chartgen.features.meter import get_meter_spec, validate_time_signature
from tja_ai_chartgen.generation import (
    MAX_AI_TRANSPORT_RETRIES,
    GenerationNotice,
    build_analysis_notices,
    build_song_analysis,
    generate_chart_bars,
)
from tja_ai_chartgen.rules.fallback_generator import validate_density
from tja_ai_chartgen.rules.styles import validate_style
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, ChartMetadata, SongAnalysis, TjaChart
from tja_ai_chartgen.tja.writer import read_tja_text, render_tja, write_tja_text
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_WEB_OUTPUT_DIR = Path("output/web")
WEB_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
_WEB_UPLOAD_CHUNK_BYTES = 1024 * 1024
WEB_ASSET_DIR = Path(__file__).resolve().parent / "assets"
WEB_SOUND_FILES = {
    "taiko_don_16bit_44100.wav": "don",
    "taiko_ka_16bit_44100.wav": "ka",
}
ALLOWED_WEB_NOTES = set("01234578")
DEFAULT_WEB_BALLOON_COUNT = 8
DEFAULT_AI_REQUEST_TIMEOUT = 300.0
DEFAULT_AI_TRANSPORT_RETRIES = 1
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


class UploadTooLargeError(ValueError):
    pass


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

.metadata-retry-card {
  display: grid;
  gap: 0.85rem;
  padding: 1rem;
  background: rgba(5, 6, 8, 0.32);
  border: 1px solid rgba(224, 138, 116, 0.32);
  border-radius: var(--radius-md);
}

.metadata-retry-card h2,
.metadata-retry-card p {
  margin-bottom: 0;
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

.danger-button {
  color: #fff4ef;
  background: linear-gradient(180deg, #b84b38, #843326);
  box-shadow: 0 16px 34px rgba(184, 75, 56, 0.2);
}

.danger-button:disabled {
  cursor: wait;
  filter: saturate(0.55);
  opacity: 0.72;
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

.conflict-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin-top: 1rem;
}

.conflict-actions form {
  margin: 0;
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

.progress-shell {
  display: grid;
  grid-template-columns: minmax(0, 0.82fr) minmax(22rem, 1fr);
  gap: clamp(1rem, 4vw, 2.5rem);
  align-items: stretch;
}

.progress-visual {
  position: relative;
  min-height: 28rem;
  overflow: hidden;
  background:
    radial-gradient(circle at 30% 28%, rgba(250, 64, 40, 0.18), transparent 13rem),
    radial-gradient(circle at 78% 66%, rgba(70, 193, 196, 0.2), transparent 14rem),
    linear-gradient(145deg, rgba(246, 240, 226, 0.1), rgba(246, 240, 226, 0.045));
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
}

.progress-drum {
  position: absolute;
  top: 50%;
  left: 50%;
  display: grid;
  width: min(72vw, 19rem);
  aspect-ratio: 1;
  place-items: center;
  background: #2e2b26;
  border: 1.15rem solid #f1eadc;
  border-radius: 999px;
  transform: translate(-50%, -50%);
  box-shadow: 0 1.1rem 0 rgba(0, 0, 0, 0.42), inset 0 0 0 1rem #463f35;
}

.progress-drum::before,
.progress-drum::after {
  position: absolute;
  width: 4.8rem;
  height: 4.8rem;
  content: "";
  background: #fa4028;
  border: 0.38rem solid #f1eadc;
  border-radius: 999px;
  box-shadow: 0 0.35rem 0 rgba(0, 0, 0, 0.38);
  animation: orbit-note 2600ms linear infinite;
}

.progress-drum::after {
  background: #46c1c4;
  animation-delay: -1300ms;
}

.progress-drum-core {
  z-index: 1;
  color: #f1eadc;
  font-size: clamp(4rem, 9vw, 7rem);
  font-weight: 860;
  letter-spacing: -0.12em;
}

.progress-beatline {
  position: absolute;
  right: 9%;
  bottom: 11%;
  left: 9%;
  height: 0.8rem;
  overflow: hidden;
  background: rgba(246, 240, 226, 0.09);
  border: 1px solid rgba(246, 240, 226, 0.14);
  border-radius: 999px;
}

.progress-beatline span {
  display: block;
  width: 42%;
  height: 100%;
  background: linear-gradient(90deg, transparent, #d6a85f, transparent);
  border-radius: inherit;
  animation: scan-beat 1500ms ease-in-out infinite;
}

.progress-panel {
  display: grid;
  gap: 1rem;
  align-content: start;
}

.progress-meter {
  overflow: hidden;
  height: 0.82rem;
  background: rgba(12, 13, 16, 0.62);
  border: 1px solid rgba(246, 240, 226, 0.14);
  border-radius: 999px;
}

.progress-meter-fill {
  display: block;
  width: 0%;
  height: 100%;
  background: linear-gradient(90deg, #fa4028, #d6a85f 55%, #46c1c4);
  border-radius: inherit;
  transition: width 420ms ease;
}

.progress-stage-list {
  display: grid;
  gap: 0.75rem;
  padding: 0;
  margin: 0;
  list-style: none;
}

.progress-step {
  display: grid;
  grid-template-columns: 2.3rem 1fr;
  gap: 0.8rem;
  padding: 0.85rem;
  color: #d7d0c0;
  background: rgba(12, 13, 16, 0.34);
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: var(--radius-sm);
}

.progress-step-marker {
  display: grid;
  width: 2.3rem;
  height: 2.3rem;
  place-items: center;
  color: var(--muted);
  font-weight: 820;
  font-variant-numeric: tabular-nums;
  background: rgba(246, 240, 226, 0.08);
  border-radius: 999px;
}

.progress-step-title {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  align-items: center;
  justify-content: space-between;
  color: var(--ink);
  font-weight: 760;
}

.progress-step-copy {
  margin: 0.15rem 0 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.progress-step-state {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.1em;
}

.progress-step[data-state="running"] {
  border-color: rgba(214, 168, 95, 0.46);
  box-shadow: 0 0 0 4px rgba(214, 168, 95, 0.08);
}

.progress-step[data-state="running"] .progress-step-marker {
  color: #16130d;
  background: var(--accent);
  animation: pulse-step 900ms ease-in-out infinite alternate;
}

.progress-step[data-state="done"] .progress-step-marker {
  color: #15200f;
  background: #8ebc76;
}

.progress-step[data-state="error"] {
  border-color: rgba(224, 138, 116, 0.48);
}

.progress-step[data-state="error"] .progress-step-marker {
  color: #2a100a;
  background: var(--danger);
}

.progress-step[data-state="cancelled"] {
  border-color: rgba(184, 75, 56, 0.48);
}

.progress-step[data-state="cancelled"] .progress-step-marker {
  color: #fff4ef;
  background: #843326;
}

@keyframes orbit-note {
  from { transform: rotate(0deg) translateX(11.5rem) rotate(0deg); }
  to { transform: rotate(1turn) translateX(11.5rem) rotate(-1turn); }
}

@keyframes scan-beat {
  0% { transform: translateX(-110%); }
  100% { transform: translateX(250%); }
}

@keyframes pulse-step {
  from { transform: scale(0.96); }
  to { transform: scale(1.04); }
}

@media (prefers-reduced-motion: reduce) {
  .progress-drum::before,
  .progress-drum::after,
  .progress-beatline span,
  .progress-step[data-state="running"] .progress-step-marker {
    animation: none;
  }
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

.notice-stack {
  display: grid;
  gap: 0.75rem;
  margin: 1rem 0;
}

.notice-stack .notice {
  margin: 0;
}

.warning {
  color: #ffe9bd;
  background: rgba(214, 168, 95, 0.14);
  border-color: rgba(214, 168, 95, 0.42);
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
  .summary-grid,
  .progress-shell {
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
  const aiToggle = form.querySelector('[data-role="ai-toggle"]');
  const aiOptions = form.querySelector('[data-role="ai-options"]');
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
  if (aiToggle && aiOptions) {
    const aiInputs = Array.from(aiOptions.querySelectorAll('input'));
    const syncAiOptions = () => {
      aiInputs.forEach((input) => {
        input.disabled = !aiToggle.checked;
      });
    };
    aiToggle.addEventListener('change', syncAiOptions);
    syncAiOptions();
  }
}

function setupProgressPage(root) {
  const statusUrl = root.dataset.statusUrl;
  const cancelUrl = root.dataset.cancelUrl;
  const message = root.querySelector('[data-role="progress-message"]');
  const meter = root.querySelector('[data-role="progress-meter"]');
  const notices = root.querySelector('[data-role="progress-notices"]');
  const error = root.querySelector('[data-role="progress-error"]');
  const retryPanel = root.querySelector('[data-role="metadata-retry"]');
  const retryTitle = root.querySelector('[data-role="metadata-retry-title"]');
  const retryArtist = root.querySelector('[data-role="metadata-retry-artist"]');
  const cancelButton = root.querySelector('[data-role="cancel-job"]');
  const steps = Array.from(root.querySelectorAll('[data-progress-step]'));
  let metadataRetryFilled = false;
  if (!statusUrl) return;

  function labelForState(state) {
    if (state === 'done') return '完成';
    if (state === 'running') return '进行中';
    if (state === 'error') return '出错';
    if (state === 'cancelled') return '已终止';
    return '等待';
  }

  function renderNotices(items) {
    if (!notices) return;
    notices.replaceChildren();
    (items || []).forEach((item) => {
      const notice = document.createElement('p');
      const level = item.level === 'error' ? 'error' : item.level === 'warning' ? 'warning' : '';
      notice.className = `notice ${level}`.trim();
      const scope = item.scope ? `${item.scope}：` : '';
      const detail = item.detail ? ` ${item.detail}` : '';
      notice.textContent = `${scope}${item.message || '生成过程有一条提示。'}${detail}`;
      notices.appendChild(notice);
    });
    notices.hidden = notices.childElementCount === 0;
  }

  function renderProgress(data) {
    if (message) message.textContent = data.message || '正在处理音频。';
    renderNotices(data.notices);
    if (meter) meter.style.width = `${Math.max(0, Math.min(100, Number(data.progress) || 0))}%`;
    const stepStateByKey = Object.fromEntries((data.steps || []).map((step) => [step.key, step.state]));
    steps.forEach((step) => {
      const state = stepStateByKey[step.dataset.progressStep] || 'pending';
      step.dataset.state = state;
      const stateLabel = step.querySelector('[data-role="progress-step-state"]');
      if (stateLabel) stateLabel.textContent = labelForState(state);
    });
    if (cancelButton) {
      cancelButton.disabled = ['cancelling', 'cancelled', 'done', 'error'].includes(data.status);
      cancelButton.textContent = data.status === 'cancelling' ? '正在终止' : data.status === 'cancelled' ? '任务已终止' : '终止任务';
    }
    if (data.status === 'done' && data.result_url) {
      window.location.href = data.result_url;
      return false;
    }
    if (data.status === 'error') {
      if (error) {
        error.hidden = false;
        error.textContent = data.error || '任务失败，请返回首页重试。';
      }
      if (retryPanel && data.can_retry_metadata) {
        retryPanel.hidden = false;
        if (!metadataRetryFilled) {
          if (retryTitle) retryTitle.value = data.title || '';
          if (retryArtist) retryArtist.value = data.artist || '';
          metadataRetryFilled = true;
        }
      }
      return false;
    }
    if (data.status === 'cancelled') return false;
    return true;
  }

  if (cancelButton && cancelUrl) {
    cancelButton.addEventListener('click', () => {
      cancelButton.disabled = true;
      cancelButton.textContent = '正在终止';
      fetch(cancelUrl, { method: 'POST', cache: 'no-store' })
        .then(async (response) => {
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || '终止任务失败。');
          renderProgress(data);
        })
        .catch((cancelError) => {
          cancelButton.disabled = false;
          cancelButton.textContent = '终止任务';
          if (error) {
            error.hidden = false;
            error.textContent = cancelError.message || '终止任务失败。';
          }
        });
    });
  }

  function poll() {
    fetch(statusUrl, { cache: 'no-store' })
      .then((response) => response.json())
      .then((data) => {
        if (renderProgress(data)) window.setTimeout(poll, 900);
      })
      .catch(() => {
        if (message) message.textContent = '暂时无法读取进度，正在重试。';
        window.setTimeout(poll, 1400);
      });
  }

  poll();
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
document.querySelectorAll('[data-progress-page]').forEach(setupProgressPage);
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


def create_app(
    output_dir: Path = DEFAULT_WEB_OUTPUT_DIR,
    *,
    remote_mode: bool = False,
    allow_instrument_analysis: bool = False,
) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="tja-ai-chartgen Web UI")
    app.state.output_dir = output_dir
    app.state.remote_mode = remote_mode
    app.state.allow_instrument_analysis = not remote_mode or allow_instrument_analysis
    app.state.job_cancellations: dict[str, Event] = {}
    app.state.job_cancellations_lock = Lock()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _page(
            "tja-ai-chartgen",
            _analysis_form(
                allow_instrument_analysis=app.state.allow_instrument_analysis
            ),
        )

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
        use_instrument_analysis: Annotated[bool, Form()] = False,
        instrument_device: Annotated[str, Form()] = "auto",
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
        ai_request_timeout: Annotated[float, Form()] = DEFAULT_AI_REQUEST_TIMEOUT,
        ai_transport_retries: Annotated[int, Form()] = DEFAULT_AI_TRANSPORT_RETRIES,
    ) -> HTMLResponse:
        job_dir: Path | None = None
        try:
            validate_density(density)
            validate_style(style)
            if instrument_device not in {"auto", "cpu", "cuda", "mps"}:
                raise ValueError("Instrument device must be auto, cpu, cuda, or mps")
            if use_instrument_analysis and not app.state.allow_instrument_analysis:
                raise ValueError(
                    "Remote instrument analysis is disabled by the server administrator"
                )
            if ai_repair_retries < 0:
                raise ValueError("AI repair retries must be greater than or equal to 0")
            _validate_ai_transport_settings(ai_request_timeout, ai_transport_retries)
            resolved_ai_base_url, resolved_ai_api_key = (None, None)
            if use_ai:
                resolved_ai_base_url, resolved_ai_api_key = _resolve_web_ai_credentials(
                    ai_base_url, ai_api_key
                )
            job_dir = _new_job_dir(app.state.output_dir)
            input_path = _save_upload(job_dir, audio)
            _write_progress(
                job_dir,
                status=_PROGRESS_RUNNING,
                step="upload",
                message="音频已接收，准备转换为 OGG。",
            )
            cancel_event = Event()
            with app.state.job_cancellations_lock:
                app.state.job_cancellations[job_dir.name] = cancel_event
            Thread(
                target=_run_tracked_analyze_job,
                kwargs={
                    "cancellation_registry": app.state.job_cancellations,
                    "registry_lock": app.state.job_cancellations_lock,
                    "cancel_event": cancel_event,
                    "job_dir": job_dir,
                    "input_path": input_path,
                    "title": title,
                    "artist": artist,
                    "max_bars": max_bars,
                    "bpm": bpm,
                    "offset": offset,
                    "time_signature": time_signature,
                    "use_beatnet": use_beatnet,
                    "use_instrument_analysis": use_instrument_analysis,
                    "instrument_device": instrument_device,
                    "course": course,
                    "level": level,
                    "style": style,
                    "density": density,
                    "special_notes": special_notes,
                    "use_ai": use_ai,
                    "ai_model": ai_model,
                    "ai_base_url": resolved_ai_base_url,
                    "ai_api_key": resolved_ai_api_key,
                    "ai_repair_retries": ai_repair_retries,
                    "ai_request_timeout": ai_request_timeout,
                    "ai_transport_retries": ai_transport_retries,
                    "remote_mode": app.state.remote_mode,
                },
                daemon=True,
            ).start()
            return HTMLResponse(_progress_page(job_dir.name))
        except UploadTooLargeError as error:
            _remove_failed_job_dir(job_dir)
            return HTMLResponse(
                _page("Analysis failed", _error_notice(str(error))),
                status_code=413,
            )
        except Exception as error:  # noqa: BLE001 - Web boundary returns a readable error page.
            _remove_failed_job_dir(job_dir)
            return HTMLResponse(
                _page("Analysis failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.get("/jobs/{job_id}/progress", response_class=HTMLResponse)
    async def progress_page(job_id: str) -> str:
        try:
            _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            return _page("Progress unavailable", _error_notice(str(error)))
        return _progress_page(job_id)

    @app.get("/jobs/{job_id}/status")
    async def job_status(job_id: str) -> JSONResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse(_read_progress(job_dir))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> JSONResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        progress = _read_progress(job_dir)
        if progress.get("status") in {_PROGRESS_DONE, _PROGRESS_ERROR, _PROGRESS_CANCELLED}:
            raise HTTPException(status_code=409, detail="Job is no longer running")

        with app.state.job_cancellations_lock:
            cancel_event = app.state.job_cancellations.get(job_id)
        if cancel_event is None:
            raise HTTPException(status_code=409, detail="Job is not active in this server process")

        step = str(progress.get("step", "upload"))
        _write_progress(
            job_dir,
            status=_PROGRESS_CANCELLING,
            step=step,
            message="正在终止任务；若当前正在调用 AI，将立即取消该请求。",
        )
        cancel_event.set()
        return JSONResponse(_read_progress(job_dir), status_code=202)

    @app.get("/jobs/{job_id}/result", response_class=HTMLResponse)
    async def job_result(job_id: str):
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            return HTMLResponse(_page("Result unavailable", _error_notice(str(error))), status_code=404)
        progress = _read_progress(job_dir)
        if progress.get("status") != _PROGRESS_DONE:
            return RedirectResponse(f"/jobs/{job_id}/progress", status_code=303)
        result_path = job_dir / _PROGRESS_RESULT_HTML
        if not result_path.is_file():
            return HTMLResponse(_page("Result unavailable", _error_notice("结果页面尚未写入。")), status_code=404)
        return HTMLResponse(result_path.read_text(encoding="utf-8"))

    @app.post("/jobs/{job_id}/retry-metadata", response_class=HTMLResponse)
    async def retry_metadata(
        job_id: str,
        title: Annotated[str, Form()],
        artist: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
            result_path = _render_job_preview_with_metadata(
                job_dir,
                title=title,
                artist=artist,
                remote_mode=app.state.remote_mode,
            )
            return HTMLResponse(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Metadata retry failed", _error_notice(str(error))),
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
        job_dir: Path | None = None
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
        except UploadTooLargeError as error:
            _remove_failed_job_dir(job_dir)
            return HTMLResponse(
                _page("TJA preview failed", _error_notice(str(error))),
                status_code=413,
            )
        except (UnicodeDecodeError, ValueError) as error:
            _remove_failed_job_dir(job_dir)
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
        safe_filename = Path(filename).name
        if safe_filename != filename or not _is_public_job_file(job_dir, safe_filename):
            raise HTTPException(status_code=404, detail=f"Job file not found: {filename}")
        path = job_dir / safe_filename
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
        ai_request_timeout: Annotated[float, Form()] = DEFAULT_AI_REQUEST_TIMEOUT,
        ai_transport_retries: Annotated[int, Form()] = DEFAULT_AI_TRANSPORT_RETRIES,
    ) -> HTMLResponse:
        try:
            validate_density(density)
            validate_style(style)
            if ai_repair_retries < 0:
                raise ValueError("AI repair retries must be greater than or equal to 0")
            _validate_ai_transport_settings(ai_request_timeout, ai_transport_retries)
            resolved_ai_base_url, resolved_ai_api_key = (None, None)
            if use_ai:
                resolved_ai_base_url, resolved_ai_api_key = _resolve_web_ai_credentials(
                    ai_base_url, ai_api_key
                )
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            selected_bars = _select_bars(analysis, start_bar, end_bar)
            generation_kwargs = {
                "analysis": analysis,
                "selected_bars": selected_bars,
                "course": course,
                "level": level,
                "style": style,
                "density": density,
                "special_notes": special_notes,
                "use_ai": use_ai,
                "model": _optional_form_text(ai_model),
                "api_base": resolved_ai_base_url,
                "api_key": resolved_ai_api_key,
                "ai_repair_retries": ai_repair_retries,
                "ai_request_timeout": ai_request_timeout,
                "ai_transport_retries": ai_transport_retries,
                "ai_input_path": job_dir / f"ai_input_{start_bar}_{end_bar}.json",
                "ai_output_path": job_dir / f"ai_output_{start_bar}_{end_bar}.json",
                "ai_attempts_path": job_dir / f"ai_attempts_{start_bar}_{end_bar}.json",
            }
            generation_result = (
                await asyncio.to_thread(generate_chart_bars, **generation_kwargs)
                if use_ai
                else generate_chart_bars(**generation_kwargs)
            )
            chart_bars = generation_result.chart_bars
            notices = [
                notice
                for notice in _read_job_notices(job_dir)
                if notice.stage == "analysis"
            ]
            notices.extend(generation_result.notices)
            write_json(
                job_dir / f"quality_report_{start_bar}_{end_bar}.json",
                generation_result.quality_report,
            )
            write_json(
                job_dir / f"generation_notices_{start_bar}_{end_bar}.json",
                notices,
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
                    _generation_notices_panel(
                        notices,
                        include_details=not app.state.remote_mode,
                    ),
                    _result_panel(
                        job_id=job_id,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(
                        job_id,
                        len(analysis.bars),
                        course,
                        ai_request_timeout=ai_request_timeout,
                        ai_transport_retries=ai_transport_retries,
                    ),
                ]
            )
            return HTMLResponse(_page("Regenerated bars", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page(
                    "Regeneration failed",
                    _error_notice(
                        _web_error_detail(str(error), error, remote_mode=app.state.remote_mode)
                    ),
                ),
                status_code=400,
            )

    @app.post("/export-chart", response_class=HTMLResponse)
    async def export_chart(
        job_id: Annotated[str, Form()],
        tja_filename: Annotated[str, Form()],
        output_dir: Annotated[str, Form()],
        conflict_action: Annotated[str, Form()] = "error",
        output_stem: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        if app.state.remote_mode:
            return HTMLResponse(
                _page(
                    "Export forbidden",
                    _error_notice("Server-side export is disabled in remote mode."),
                ),
                status_code=403,
            )
        try:
            job_dir, analysis, tja_path = _load_export_context(
                app.state.output_dir, job_id, tja_filename
            )
            target_dir = _parse_export_output_dir(output_dir)
            tja_text = read_tja_text(tja_path)
            try:
                export_ogg_path, export_tja_path = _export_chart_files(
                    tja_path=tja_path,
                    analysis=analysis,
                    output_dir=target_dir,
                    conflict_action=conflict_action,
                    output_stem=output_stem,
                )
            except (FileExistsError, ValueError) as error:
                if isinstance(error, FileExistsError) or conflict_action == "rename":
                    body = _export_conflict_panel(
                        job_id=job_id,
                        tja_filename=tja_filename,
                        analysis=analysis,
                        output_dir=target_dir,
                        message=str(error),
                        requested_stem=output_stem,
                    )
                    return HTMLResponse(
                        _page("Export target exists", body),
                        status_code=409 if isinstance(error, FileExistsError) else 400,
                    )
                raise
            _parsed_analysis, chart_bars, course, level = _parse_tja_preview(
                tja_text,
                audio_file=Path(analysis.audio_file),
                ogg_file=Path(analysis.ogg_file),
            )
            body = "".join(
                [
                    _export_success_notice(export_ogg_path, export_tja_path),
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
        except (OSError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Export failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.post("/open-export-directory", response_class=HTMLResponse)
    async def open_export_directory(
        job_id: Annotated[str, Form()],
        tja_filename: Annotated[str, Form()],
        output_dir: Annotated[str, Form()],
    ) -> HTMLResponse:
        if app.state.remote_mode:
            return HTMLResponse(
                _page(
                    "Open directory forbidden",
                    _error_notice("Opening server directories is disabled in remote mode."),
                ),
                status_code=403,
            )
        try:
            _job_dir, analysis, _tja_path = _load_export_context(
                app.state.output_dir, job_id, tja_filename
            )
            target_dir = _parse_export_output_dir(output_dir)
            _open_directory(target_dir)
            body = _export_conflict_panel(
                job_id=job_id,
                tja_filename=tja_filename,
                analysis=analysis,
                output_dir=target_dir,
                message="目标目录中已有同名文件，请选择处理方式。",
                notice="已打开文件所在目录。",
            )
            return HTMLResponse(_page("Export target exists", body))
        except (FileNotFoundError, OSError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Open directory failed", _error_notice(str(error))),
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


def _run_tracked_analyze_job(
    *,
    cancellation_registry: dict[str, Event],
    registry_lock: LockType,
    cancel_event: Event,
    job_dir: Path,
    **job_kwargs: object,
) -> None:
    try:
        _run_analyze_job(
            job_dir=job_dir,
            cancel_event=cancel_event,
            **job_kwargs,
        )
    finally:
        with registry_lock:
            cancellation_registry.pop(job_dir.name, None)


def _run_analyze_job(
    *,
    job_dir: Path,
    input_path: Path,
    title: str,
    artist: str,
    max_bars: int | None,
    bpm: float | None,
    offset: float | None,
    time_signature: str,
    use_beatnet: bool,
    use_instrument_analysis: bool,
    instrument_device: str,
    course: str,
    level: int,
    style: str,
    density: str,
    special_notes: bool,
    use_ai: bool,
    ai_model: str,
    ai_base_url: str | None,
    ai_api_key: str | None,
    ai_repair_retries: int,
    ai_request_timeout: float,
    ai_transport_retries: int,
    remote_mode: bool,
    cancel_event: Event,
) -> None:
    try:
        raise_if_cancelled(cancel_event)
        ogg_path = job_dir / f"{input_path.stem}.ogg"

        def report_analysis_stage(stage: str) -> None:
            raise_if_cancelled(cancel_event)
            if stage == "convert":
                _write_progress(
                    job_dir,
                    status=_PROGRESS_RUNNING,
                    step="convert",
                    message="正在调用 ffmpeg 转换音频，完成后会进入节拍分析。",
                )
            elif stage == "analyze":
                _write_progress(
                    job_dir,
                    status=_PROGRESS_RUNNING,
                    step="analyze",
                    message="正在提取 BPM、OFFSET、拍号和小节能量。",
                )
            elif stage == "instruments":
                _write_progress(
                    job_dir,
                    status=_PROGRESS_RUNNING,
                    step="instruments",
                    message="正在使用本地模型分析人声、鼓、贝斯和伴奏乐器。",
                )

        analysis = build_song_analysis(
            input_audio=input_path,
            ogg_path=ogg_path,
            title=title,
            artist=artist or None,
            max_bars=max_bars,
            bpm_override=bpm,
            offset_override=offset,
            time_signature_override=time_signature or None,
            use_beatnet=use_beatnet,
            use_instrument_analysis=use_instrument_analysis,
            instrument_device=instrument_device,
            stage_callback=report_analysis_stage,
        )
        raise_if_cancelled(cancel_event)
        bars = analysis.bars
        write_json(job_dir / "analysis.json", analysis)
        notices = build_analysis_notices(
            analysis,
            requested_beatnet=use_beatnet,
            requested_instrument_analysis=use_instrument_analysis,
        )
        _write_job_notices(job_dir, notices)

        _write_progress(
            job_dir,
            status=_PROGRESS_RUNNING,
            step="generate",
            message=(
                "正在调用 AI 生成全曲谱面，失败时会自动回退规则生成。"
                if use_ai
                else "正在用规则生成器生成全曲谱面。"
            ),
            notices=notices,
            include_notice_details=not remote_mode,
        )
        generation_result = generate_chart_bars(
            analysis=analysis,
            selected_bars=bars,
            course=course,
            level=level,
            style=style,
            density=density,
            special_notes=special_notes,
            use_ai=use_ai,
            model=_optional_form_text(ai_model),
            api_base=ai_base_url,
            api_key=ai_api_key,
            ai_repair_retries=ai_repair_retries,
            ai_request_timeout=ai_request_timeout,
            ai_transport_retries=ai_transport_retries,
            ai_input_path=job_dir / f"ai_input_1_{len(bars)}.json",
            ai_output_path=job_dir / f"ai_output_1_{len(bars)}.json",
            ai_attempts_path=job_dir / f"ai_attempts_1_{len(bars)}.json",
            cancel_event=cancel_event,
        )
        raise_if_cancelled(cancel_event)
        chart_bars = generation_result.chart_bars
        ai_failure = generation_result.ai_failure
        notices.extend(generation_result.notices)
        _write_job_notices(job_dir, notices)
        write_json(
            job_dir / f"quality_report_1_{len(bars)}.json",
            generation_result.quality_report,
        )
        write_json(job_dir / _PROGRESS_CHART_BARS_JSON, chart_bars)
        write_json(
            job_dir / _PROGRESS_CHART_OPTIONS_JSON,
            {
                "course": course,
                "level": level,
                "use_instrument_analysis": use_instrument_analysis,
                "instrument_device": instrument_device,
                "use_ai": use_ai,
                "ai_failure": ai_failure if use_ai else None,
                "ai_request_timeout": ai_request_timeout,
                "ai_transport_retries": ai_transport_retries,
            },
        )

        _write_progress(
            job_dir,
            status=_PROGRESS_RUNNING,
            step="render",
            message="正在写入 preview.tja，并准备可视化游玩预览。",
            notices=notices,
            include_notice_details=not remote_mode,
        )
        raise_if_cancelled(cancel_event)
        _write_preview_result(
            job_dir=job_dir,
            analysis=analysis,
            chart_bars=chart_bars,
            course=course,
            level=level,
            notices=notices,
            remote_mode=remote_mode,
            ai_request_timeout=ai_request_timeout,
            ai_transport_retries=ai_transport_retries,
        )
        _write_progress(
            job_dir,
            status=_PROGRESS_DONE,
            step="render",
            message="谱面生成完成，正在打开游玩预览。",
            result_url=f"/jobs/{job_dir.name}/result",
            notices=notices,
            include_notice_details=not remote_mode,
        )
    except GenerationCancelledError:
        step = str(_read_progress(job_dir).get("step", "upload"))
        _write_progress(
            job_dir,
            status=_PROGRESS_CANCELLED,
            step=step,
            message="任务已终止，未继续生成或写入谱面。",
        )
    except Exception as error:  # noqa: BLE001 - Background job reports failures through status JSON.
        step = str(_read_progress(job_dir).get("step", "upload"))
        can_retry_metadata, retry_analysis = _metadata_retry_state(job_dir)
        private_error = _redact_secret(str(error), ai_api_key)
        notices = _read_job_notices(job_dir)
        notices.append(
            GenerationNotice(
                code=f"{step}-failed",
                level="error",
                stage=_notice_stage_for_progress_step(step),
                message="任务停止，请根据错误信息调整参数后重试。",
                detail=private_error,
            )
        )
        _write_job_notices(job_dir, notices)
        _write_progress(
            job_dir,
            status=_PROGRESS_ERROR,
            step=step,
            message="任务停止，请根据错误信息调整参数后重试。",
            error=_web_error_detail(private_error, error, remote_mode=remote_mode),
            can_retry_metadata=can_retry_metadata,
            title=retry_analysis.title if retry_analysis else None,
            artist=retry_analysis.artist if retry_analysis else None,
            notices=notices,
            include_notice_details=not remote_mode,
        )


def _write_preview_result(
    *,
    job_dir: Path,
    analysis: SongAnalysis,
    chart_bars: list[ChartBar],
    course: str,
    level: int,
    notices: list[GenerationNotice] | None = None,
    remote_mode: bool = False,
    ai_request_timeout: float = DEFAULT_AI_REQUEST_TIMEOUT,
    ai_transport_retries: int = DEFAULT_AI_TRANSPORT_RETRIES,
) -> Path:
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
    output_path = job_dir / "preview.tja"
    write_tja_text(output_path, tja_text)
    result_body = (
        _generation_notices_panel(notices or [], include_details=not remote_mode)
        + _result_panel(
            job_id=job_dir.name,
            output_path=output_path,
            tja_text=tja_text,
            analysis=analysis,
            chart_bars=chart_bars,
            course=course,
            level=level,
        )
        + _regenerate_form(
            job_dir.name,
            len(analysis.bars),
            course,
            ai_request_timeout=ai_request_timeout,
            ai_transport_retries=ai_transport_retries,
        )
    )
    result_path = job_dir / _PROGRESS_RESULT_HTML
    result_path.write_text(_page("Game preview", result_body), encoding="utf-8")
    return result_path


def _render_job_preview_with_metadata(
    job_dir: Path,
    *,
    title: str,
    artist: str,
    remote_mode: bool = False,
) -> Path:
    normalized_title = title.strip()
    if not normalized_title:
        raise ValueError("Title is required")
    analysis = SongAnalysis.model_validate_json((job_dir / "analysis.json").read_text(encoding="utf-8"))
    analysis = analysis.model_copy(
        update={"title": normalized_title, "artist": _optional_form_text(artist)}
    )
    chart_bars = _read_job_chart_bars(job_dir)
    options = _read_job_chart_options(job_dir)
    course = str(options.get("course") or "Oni")
    level = int(options.get("level") or 10)
    ai_request_timeout = float(options.get("ai_request_timeout", DEFAULT_AI_REQUEST_TIMEOUT))
    ai_transport_retries = int(options.get("ai_transport_retries", DEFAULT_AI_TRANSPORT_RETRIES))
    notices = [
        notice
        for notice in _read_job_notices(job_dir)
        if not (notice.stage == "render" and notice.level == "error")
    ]
    notices.append(
        GenerationNotice(
            code="metadata-retry-succeeded",
            level="info",
            stage="render",
            message="已复用现有分析与谱面，仅更新元数据并重新写入预览。",
        )
    )
    _write_job_notices(job_dir, notices)
    result_path = _write_preview_result(
        job_dir=job_dir,
        analysis=analysis,
        chart_bars=chart_bars,
        course=course,
        level=level,
        notices=notices,
        remote_mode=remote_mode,
        ai_request_timeout=ai_request_timeout,
        ai_transport_retries=ai_transport_retries,
    )
    write_json(job_dir / "analysis.json", analysis)
    _write_progress(
        job_dir,
        status=_PROGRESS_DONE,
        step="render",
        message="谱面生成完成，正在打开游玩预览。",
        result_url=f"/jobs/{job_dir.name}/result",
        notices=notices,
        include_notice_details=not remote_mode,
    )
    return result_path


def _read_job_chart_bars(job_dir: Path) -> list[ChartBar]:
    path = job_dir / _PROGRESS_CHART_BARS_JSON
    if not path.is_file():
        raise FileNotFoundError("Generated chart bars are not available for this job")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Generated chart bars file is invalid")
    return [ChartBar.model_validate(item) for item in data]


def _read_job_chart_options(job_dir: Path) -> dict[str, object]:
    path = job_dir / _PROGRESS_CHART_OPTIONS_JSON
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Generated chart options file is invalid")
    return data


def _write_job_notices(job_dir: Path, notices: list[GenerationNotice]) -> Path:
    return write_json(job_dir / _PROGRESS_NOTICES_JSON, notices)


def _read_job_notices(job_dir: Path) -> list[GenerationNotice]:
    path = job_dir / _PROGRESS_NOTICES_JSON
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Generation notices file is invalid")
    return [GenerationNotice.model_validate(item) for item in data]


def _metadata_retry_state(job_dir: Path) -> tuple[bool, SongAnalysis | None]:
    try:
        analysis = SongAnalysis.model_validate_json((job_dir / "analysis.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValidationError, ValueError):
        return False, None
    return (job_dir / _PROGRESS_CHART_BARS_JSON).is_file(), analysis


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


def _remove_failed_job_dir(job_dir: Path | None) -> None:
    if job_dir is not None:
        rmtree(job_dir, ignore_errors=True)


def _is_public_job_file(job_dir: Path, filename: str) -> bool:
    if filename == "preview.tja":
        return True
    if re.fullmatch(r"(?:regenerated|edited)_\d+_\d+\.tja", filename):
        return True
    if not filename.endswith(".ogg"):
        return False
    analysis_path = job_dir / "analysis.json"
    if not analysis_path.is_file():
        return False
    try:
        analysis = SongAnalysis.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return False
    return filename == Path(analysis.ogg_file).name


def _save_upload(
    job_dir: Path,
    audio: UploadFile,
    *,
    max_bytes: int = WEB_UPLOAD_MAX_BYTES,
    chunk_size: int = _WEB_UPLOAD_CHUNK_BYTES,
) -> Path:
    filename = Path(audio.filename or "upload.audio").name
    path = job_dir / filename
    written = 0
    try:
        with path.open("wb") as output:
            while chunk := audio.file.read(chunk_size):
                written += len(chunk)
                if written > max_bytes:
                    limit_mib = max_bytes // (1024 * 1024)
                    raise UploadTooLargeError(
                        f"Uploaded file exceeds the {limit_mib} MiB limit"
                    )
                output.write(chunk)
        if written == 0:
            raise ValueError("Uploaded audio is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


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


def _parse_export_output_dir(value: str) -> Path:
    if not value.strip():
        raise ValueError("Output directory is required")
    return Path(value)


def _load_export_context(
    jobs_dir: Path, job_id: str, tja_filename: str
) -> tuple[Path, SongAnalysis, Path]:
    job_dir = _job_dir(jobs_dir, job_id)
    analysis = SongAnalysis.model_validate_json(
        (job_dir / "analysis.json").read_text(encoding="utf-8")
    )
    return job_dir, analysis, _job_file_path(job_dir, tja_filename)


def _export_chart_files(
    *,
    tja_path: Path,
    analysis: SongAnalysis,
    output_dir: Path,
    conflict_action: str = "error",
    output_stem: str = "",
) -> tuple[Path, Path]:
    if not str(output_dir).strip():
        raise ValueError("Output directory is required")
    if conflict_action not in {"error", "overwrite", "rename", "backup"}:
        raise ValueError(f"Unsupported export conflict action: {conflict_action}")

    ogg_path = Path(analysis.ogg_file)
    if not ogg_path.is_file():
        raise FileNotFoundError(f"OGG file not found: {ogg_path}")
    if tja_path.suffix.lower() != ".tja":
        raise ValueError("Only .tja files can be exported")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        _validate_export_stem(output_stem)
        if conflict_action == "rename"
        else ogg_path.stem
    )
    export_ogg_path, export_tja_path = _export_target_paths(output_dir, stem)
    existing_paths = _existing_export_paths(export_ogg_path, export_tja_path)
    if existing_paths and conflict_action in {"error", "rename"}:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Export target already exists: {existing}")

    temp_ogg_path = output_dir / f".{export_ogg_path.name}.{uuid4().hex}.tmp"
    temp_tja_path = output_dir / f".{export_tja_path.name}.{uuid4().hex}.tmp"
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        copy2(ogg_path, temp_ogg_path)
        tja_text = _set_tja_wave(read_tja_text(tja_path), export_ogg_path.name)
        write_tja_text(temp_tja_path, tja_text)

        if conflict_action == "backup":
            for target_path in existing_paths:
                backup_path = _next_backup_path(target_path)
                target_path.replace(backup_path)
                backups.append((target_path, backup_path))

        temp_ogg_path.replace(export_ogg_path)
        installed.append(export_ogg_path)
        temp_tja_path.replace(export_tja_path)
        installed.append(export_tja_path)
        return export_ogg_path, export_tja_path
    except Exception:
        if conflict_action == "backup":
            for installed_path in reversed(installed):
                installed_path.unlink(missing_ok=True)
            for target_path, backup_path in reversed(backups):
                if backup_path.exists():
                    backup_path.replace(target_path)
        elif conflict_action != "overwrite":
            for installed_path in reversed(installed):
                installed_path.unlink(missing_ok=True)
        raise
    finally:
        temp_ogg_path.unlink(missing_ok=True)
        temp_tja_path.unlink(missing_ok=True)


def _export_target_paths(output_dir: Path, stem: str) -> tuple[Path, Path]:
    return output_dir / f"{stem}.ogg", output_dir / f"{stem}.tja"


def _existing_export_paths(ogg_path: Path, tja_path: Path) -> list[Path]:
    return [path for path in (ogg_path, tja_path) if path.exists()]


def _validate_export_stem(value: str) -> str:
    stem = value.strip()
    for suffix in (".ogg", ".tja"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)].rstrip()
            break
    if not stem or stem in {".", ".."}:
        raise ValueError("New export filename is required")
    if any(character in stem for character in '<>:"/\\|?*') or any(
        ord(character) < 32 for character in stem
    ):
        raise ValueError("New export filename contains invalid characters")
    if stem.endswith((".", " ")):
        raise ValueError("New export filename cannot end with a dot or space")
    return stem


def _suggest_export_stem(output_dir: Path, base_stem: str) -> str:
    base_stem = _validate_export_stem(base_stem)
    index = 1
    while True:
        candidate = f"{base_stem} ({index})"
        paths = _export_target_paths(output_dir, candidate)
        if not _existing_export_paths(*paths):
            return candidate
        index += 1


def _next_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{index}")
        index += 1
    return candidate


def _set_tja_wave(tja_text: str, wave_filename: str) -> str:
    lines = tja_text.splitlines()
    wave_line = f"WAVE:{wave_filename}"
    for index, line in enumerate(lines):
        if line.upper().startswith("WAVE:"):
            lines[index] = wave_line
            break
    else:
        insert_at = next(
            (
                index
                for index, line in enumerate(lines)
                if line.upper().startswith(("OFFSET:", "COURSE:", "#START"))
            ),
            len(lines),
        )
        lines.insert(insert_at, wave_line)
    return "\n".join(lines) + "\n"


def _open_directory(path: Path) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Export directory is not a directory: {resolved}")
    startfile = getattr(os, "startfile", None)
    if startfile is None:
        raise OSError("Opening the export directory is only supported on Windows")
    startfile(str(resolved))


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
        accent_grids = meter.accent_grids_for_resolution(len(chart_bar.notes))
        features.append(
            BarFeature(
                index=chart_bar.index,
                start_time=round(current_time, 6),
                end_time=round(current_time + bar_length, 6),
                energy=round(min(1.0, len(note_grids) / max(1, len(chart_bar.notes))), 3),
                time_signature=chart_bar.time_signature,
                grids_per_bar=len(chart_bar.notes),
                onset_grids=note_grids,
                accent_grids=[grid for grid in note_grids if grid in accent_grids],
                beat_grids=list(accent_grids),
                downbeat_grid=0,
                section="tja",
            )
        )
        current_time += bar_length
    return features


def _optional_form_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _validate_ai_transport_settings(request_timeout: float, transport_retries: int) -> None:
    if not 1 <= request_timeout <= 600:
        raise ValueError("AI request timeout must be between 1 and 600 seconds")
    if not 0 <= transport_retries <= MAX_AI_TRANSPORT_RETRIES:
        raise ValueError(
            f"AI transport retries must be between 0 and {MAX_AI_TRANSPORT_RETRIES}"
        )


def _resolve_web_ai_credentials(ai_base_url: str, ai_api_key: str) -> tuple[str | None, str | None]:
    request_base_url = _optional_form_text(ai_base_url)
    request_api_key = _optional_form_text(ai_api_key)
    if request_base_url or request_api_key:
        if not request_base_url or not request_api_key:
            raise ValueError("Custom AI base URL and API key must be provided together")
        return request_base_url, request_api_key

    environment_base_url = _optional_form_text(os.getenv("OPENAI_BASE_URL", ""))
    environment_api_key = _optional_form_text(os.getenv("OPENAI_API_KEY", ""))
    if environment_base_url or environment_api_key:
        if not environment_base_url or not environment_api_key:
            raise ValueError("Server OPENAI_BASE_URL and OPENAI_API_KEY must be configured together")
        return environment_base_url, environment_api_key
    return None, None


def _analysis_form(*, allow_instrument_analysis: bool = True) -> str:
    instrument_disabled = "" if allow_instrument_analysis else " disabled"
    instrument_hint = (
        "需要先运行 prepare-instrument-models；分析会增加耗时和内存占用。"
        if allow_instrument_analysis
        else "远程模式未由服务器管理员开放重型分析。"
    )
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
        <label class="checkbox-card field-wide">
          <input name="use_instrument_analysis" type="checkbox" value="true"{instrument_disabled}>
          <span>人声与乐器分析 <span class="field-hint">{_escape(instrument_hint)}</span></span>
        </label>
        <label class="field">
          分析设备
          <select name="instrument_device"{instrument_disabled}>
            {_option_tags(("auto", "cpu", "cuda", "mps"), "auto")}
          </select>
        </label>
        <label class="field">
          难度类型
          <select name="course">{_course_option_tags("Oni")}</select>
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
          <input name="special_notes" type="checkbox" value="true">
          <span>特殊音符 <span class="field-hint">允许简单滚奏和气球。</span></span>
        </label>
        <label class="checkbox-card field-wide">
          <input name="use_ai" type="checkbox" value="true" data-role="ai-toggle" checked>
          <span>使用 AI 增强 <span class="field-hint">分析完成后立即用全曲小节调用 AI，失败时自动回退规则生成。</span></span>
        </label>
        <details class="advanced-panel field-wide" data-role="ai-options">
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
            <label class="field">
              请求超时（秒）
              <input name="ai_request_timeout" type="number" min="1" max="600" value="300">
            </label>
            <label class="field">
              网络重试
              <input name="ai_transport_retries" type="number" min="0" max="{MAX_AI_TRANSPORT_RETRIES}" value="3">
            </label>
          </div>
        </details>
      </div>
      <div class="helper-strip">
        <button type="submit" data-loading-text="分析中">上传并分析</button>
        <span>AI API Key 只用于本次请求，不写入输出文件。</span>
      </div>
    </form>
    {_tja_preview_form()}
  </section>
</section>
"""


def _progress_page(job_id: str) -> str:
    steps = "".join(
        f"""
        <li class="progress-step" data-progress-step="{_escape(str(step['key']))}" data-state="{'running' if index == 0 else 'pending'}">
          <span class="progress-step-marker">{index + 1}</span>
          <span>
            <span class="progress-step-title">
              {_escape(str(step['label']))}
              <span class="progress-step-state" data-role="progress-step-state">{'进行中' if index == 0 else '等待'}</span>
            </span>
            <span class="progress-step-copy">{_escape(str(step['detail']))}</span>
          </span>
        </li>
        """
        for index, step in enumerate(_PROGRESS_STEPS)
    )
    return _page(
        "Generating chart",
        f"""
<section class="progress-shell" data-progress-page data-status-url="/jobs/{_escape(job_id)}/status" data-cancel-url="/jobs/{_escape(job_id)}/cancel" aria-labelledby="progress-heading">
  <section class="progress-visual" aria-hidden="true">
    <div class="progress-drum"><span class="progress-drum-core">太</span></div>
    <div class="progress-beatline"><span></span></div>
  </section>
  <section class="panel progress-panel">
    <p class="eyebrow">生成进度</p>
    <h1 id="progress-heading">正在制谱</h1>
    <p class="lede" data-role="progress-message">音频已接收，准备转换为 OGG。</p>
    <div class="progress-meter" aria-label="任务进度">
      <span class="progress-meter-fill" data-role="progress-meter"></span>
    </div>
    <ol class="progress-stage-list" aria-label="当前生成阶段">
      {steps}
    </ol>
    <div class="notice-stack" data-role="progress-notices" aria-live="polite" hidden></div>
    <p class="notice error" data-role="progress-error" hidden></p>
    <section class="metadata-retry-card" data-role="metadata-retry" aria-labelledby="metadata-retry-heading" hidden>
      <p class="eyebrow">元数据修正</p>
      <h2 id="metadata-retry-heading">修改歌名后重试</h2>
      <p class="lede">如果只是 TJA 编码不兼容，可以把歌名或歌手改成 CP932 / Shift-JIS 可保存的文本，然后直接重新写入预览。</p>
      <form action="/jobs/{_escape(job_id)}/retry-metadata" method="post">
        <div class="form-grid">
          <label class="field">
            歌名
            <input name="title" data-role="metadata-retry-title" required>
          </label>
          <label class="field">
            歌手
            <input name="artist" data-role="metadata-retry-artist" placeholder="可选">
          </label>
        </div>
        <div class="helper-strip">
          <button type="submit" data-loading-text="重试中">重试写入预览</button>
          <span>不会重新分析音频，也不会重新调用 AI。</span>
        </div>
      </form>
    </section>
    <div class="helper-strip">
      <span>任务 <code>{_escape(job_id)}</code></span>
      <button class="danger-button" type="button" data-role="cancel-job">终止任务</button>
      <a class="button-link" href="/">返回首页</a>
    </div>
  </section>
</section>
""",
    )


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


def _regenerate_form(
    job_id: str,
    bar_count: int,
    course: str = "Oni",
    *,
    ai_request_timeout: float = DEFAULT_AI_REQUEST_TIMEOUT,
    ai_transport_retries: int = DEFAULT_AI_TRANSPORT_RETRIES,
) -> str:
    end_bar = max(1, bar_count)
    request_timeout_value = f"{ai_request_timeout:g}"
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
          <label class="field">
            请求超时（秒）
            <input name="ai_request_timeout" type="number" min="1" max="600" value="{request_timeout_value}">
          </label>
          <label class="field">
            网络重试
            <input name="ai_transport_retries" type="number" min="0" max="{MAX_AI_TRANSPORT_RETRIES}" value="{ai_transport_retries}">
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


def _instrument_analysis_summary(analysis: SongAnalysis) -> str:
    if analysis.instrument_analysis_status == "unavailable":
        return ""
    status_text = {
        "complete": "完整",
        "partial": "部分生效",
        "fallback": "已降级",
    }.get(analysis.instrument_analysis_status, analysis.instrument_analysis_status)
    source_counts = {
        source: sum(bar.instrument.dominant_source == source for bar in analysis.bars)
        for source in ("vocals", "drums", "bass", "other")
    }
    label_names = {
        "guitar": "吉他",
        "piano_keyboard": "钢琴/键盘",
        "strings": "弦乐",
        "brass": "铜管",
        "woodwind": "木管",
        "synth": "合成器",
        "organ": "风琴",
        "other_instrument": "其他乐器",
    }
    detected = [
        display
        for field, display in label_names.items()
        if max((getattr(bar.instrument, field) for bar in analysis.bars), default=0.0)
        >= 0.25
    ]
    detected_text = "、".join(detected) if detected else "未得到高置信细分类别"
    device = analysis.instrument_analysis_device or "未记录"
    return f"""
  <section class="panel" aria-labelledby="instrument-summary-heading">
    <p class="eyebrow">阶段 C</p>
    <h2 id="instrument-summary-heading">人声与乐器分析</h2>
    <p class="lede">状态：{_escape(status_text)}；设备：{_escape(device)}。</p>
    <div class="helper-strip">
      <span>人声主导 {source_counts['vocals']} 小节</span>
      <span>鼓组主导 {source_counts['drums']} 小节</span>
      <span>贝斯主导 {source_counts['bass']} 小节</span>
      <span>伴奏主导 {source_counts['other']} 小节</span>
    </div>
    <p class="field-hint">主要乐器：{_escape(detected_text)}。识别结果只作为结构和谱面生成的软证据。</p>
  </section>
"""


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
  {_instrument_analysis_summary(analysis)}
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


def _export_conflict_panel(
    *,
    job_id: str,
    tja_filename: str,
    analysis: SongAnalysis,
    output_dir: Path,
    message: str,
    requested_stem: str = "",
    notice: str = "",
) -> str:
    original_stem = Path(analysis.ogg_file).stem
    candidate_base = requested_stem.strip() or original_stem
    try:
        candidate_base = _validate_export_stem(candidate_base)
    except ValueError:
        candidate_base = original_stem
    suggestion = _suggest_export_stem(output_dir, candidate_base)
    rename_value = requested_stem.strip() or suggestion
    hidden_fields = f"""
      <input name="job_id" type="hidden" value="{_escape(job_id)}">
      <input name="tja_filename" type="hidden" value="{_escape(tja_filename)}">
      <input name="output_dir" type="hidden" value="{_escape(str(output_dir))}">
    """
    notice_html = f'<p class="notice">{_escape(notice)}</p>' if notice else ""
    return f"""
<section class="stack" aria-labelledby="export-conflict-heading">
  <section class="panel">
    <p class="eyebrow">发现同名文件</p>
    <h1 id="export-conflict-heading">请选择保存方式</h1>
    {notice_html}
    <p class="notice error">{_escape(message)}</p>
    <p class="lede">默认不会覆盖现有文件。下面的操作只处理 <code>{_escape(original_stem)}.ogg</code> 和 <code>{_escape(original_stem)}.tja</code> 这一对导出文件。</p>
  </section>
  <section class="panel" aria-labelledby="rename-export-heading">
    <p class="eyebrow">保留现有文件</p>
    <h2 id="rename-export-heading">使用新名称保存</h2>
    <form action="/export-chart" method="post">
      {hidden_fields}
      <input name="conflict_action" type="hidden" value="rename">
      <div class="form-grid">
        <label class="field field-wide">
          新文件名
          <input name="output_stem" value="{_escape(rename_value)}" required>
          <span class="field-hint">OGG、TJA 和 TJA 内的 WAVE 将统一使用该名称。下一个可用名称：<code>{_escape(suggestion)}</code></span>
        </label>
      </div>
      <div class="helper-strip">
        <button type="submit" data-loading-text="保存中">重新命名保存</button>
      </div>
    </form>
  </section>
  <section class="panel" aria-labelledby="existing-export-heading">
    <p class="eyebrow">处理现有文件</p>
    <h2 id="existing-export-heading">覆盖或备份后保存</h2>
    <div class="conflict-actions">
      <form action="/export-chart" method="post">
        {hidden_fields}
        <input name="conflict_action" type="hidden" value="overwrite">
        <button type="submit" data-loading-text="覆盖中">覆盖原文件</button>
      </form>
      <form action="/export-chart" method="post">
        {hidden_fields}
        <input name="conflict_action" type="hidden" value="backup">
        <button type="submit" data-loading-text="备份中">重命名原文件并保存</button>
      </form>
      <form action="/open-export-directory" method="post">
        {hidden_fields}
        <button type="submit" data-loading-text="打开中">打开文件所在目录</button>
      </form>
    </div>
    <p class="field-hint">“重命名原文件并保存”会把旧文件改为 <code>.bak</code>；已有备份时使用 <code>.bak.1</code>、<code>.bak.2</code>。</p>
  </section>
  <div class="helper-strip">
    <a class="button-link" href="/jobs/{_escape(job_id)}/result">返回当前预览</a>
    <a class="button-link" href="/">返回上传页面</a>
  </div>
</section>
"""


def _export_success_notice(export_ogg_path: Path, export_tja_path: Path) -> str:
    return (
        '<p class="notice">已保存：'
        f'<code>{_escape(str(export_ogg_path))}</code> 和 '
        f'<code>{_escape(str(export_tja_path))}</code></p>'
    )


def _generation_notices_panel(
    notices: list[GenerationNotice],
    *,
    include_details: bool = True,
) -> str:
    if not notices:
        return ""

    items: list[str] = []
    for notice in notices:
        css_class = (
            " error"
            if notice.level == "error"
            else " warning"
            if notice.level == "warning"
            else ""
        )
        scope = f"{notice.scope}：" if notice.scope else ""
        detail = f" {_escape(notice.detail)}" if include_details and notice.detail else ""
        items.append(
            f'<p class="notice{css_class}">{_escape(scope + notice.message)}{detail}</p>'
        )
    return '<section class="notice-stack" aria-label="生成提示">' + "".join(items) + "</section>"


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


_PROGRESS_STEPS = (
    {
        "key": "upload",
        "label": "接收音频",
        "detail": "把上传文件写入本地 job 目录。",
    },
    {
        "key": "convert",
        "label": "转换 OGG",
        "detail": "调用 ffmpeg 准备可预览的音频。",
    },
    {
        "key": "analyze",
        "label": "分析节拍",
        "detail": "提取 BPM、OFFSET、拍号、小节和段落特征。",
    },
    {
        "key": "instruments",
        "label": "识别人声与乐器",
        "detail": "显式启用时使用本地 Demucs 与 AST 模型提取语义。",
    },
    {
        "key": "generate",
        "label": "生成谱面",
        "detail": "按选择的难度、风格和密度生成 TJA 小节。",
    },
    {
        "key": "render",
        "label": "写入预览",
        "detail": "渲染 preview.tja 并准备游玩预览页面。",
    },
)


_PROGRESS_STEP_INDEX = {step["key"]: index for index, step in enumerate(_PROGRESS_STEPS)}
_PROGRESS_PENDING = "pending"
_PROGRESS_RUNNING = "running"
_PROGRESS_CANCELLING = "cancelling"
_PROGRESS_CANCELLED = "cancelled"
_PROGRESS_DONE = "done"
_PROGRESS_ERROR = "error"
_PROGRESS_JSON = "progress.json"
_PROGRESS_RESULT_HTML = "result.html"
_PROGRESS_CHART_BARS_JSON = "chart_bars.json"
_PROGRESS_CHART_OPTIONS_JSON = "chart_options.json"
_PROGRESS_NOTICES_JSON = "generation_notices.json"


def _notice_payloads(
    notices: list[GenerationNotice],
    *,
    include_details: bool,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for notice in notices:
        payload = notice.model_dump(exclude_none=True)
        if not include_details:
            payload.pop("detail", None)
        payloads.append(payload)
    return payloads


def _notice_stage_for_progress_step(step: str) -> str:
    if step in {"upload", "convert", "analyze", "instruments"}:
        return "analysis"
    if step == "generate":
        return "generation"
    return "render"


def _redact_secret(message: str, secret: str | None) -> str:
    return message.replace(secret, "[REDACTED]") if secret else message


def _web_error_detail(message: str, error: Exception, *, remote_mode: bool) -> str:
    if not remote_mode:
        return message
    return (
        f"任务失败（{type(error).__name__}）。"
        "远程模式已隐藏服务器路径、连接地址和 provider 响应详情。"
    )


def _progress_payload(
    *,
    status: str,
    step: str,
    message: str,
    result_url: str | None = None,
    error: str | None = None,
    can_retry_metadata: bool = False,
    title: str | None = None,
    artist: str | None = None,
    notices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    current_index = _PROGRESS_STEP_INDEX.get(step, -1)
    steps = []
    for index, item in enumerate(_PROGRESS_STEPS):
        if status in {_PROGRESS_ERROR, _PROGRESS_CANCELLED} and index == current_index:
            state = status
        elif index < current_index or status == _PROGRESS_DONE:
            state = _PROGRESS_DONE
        elif index == current_index:
            state = status if status in {_PROGRESS_RUNNING, _PROGRESS_ERROR} else _PROGRESS_RUNNING
        else:
            state = _PROGRESS_PENDING
        steps.append({**item, "state": state})

    return {
        "status": status,
        "step": step,
        "message": message,
        "progress": _progress_percent(status, current_index),
        "steps": steps,
        "result_url": result_url,
        "error": error,
        "can_retry_metadata": can_retry_metadata,
        "title": title,
        "artist": artist,
        "notices": notices or [],
    }


def _progress_percent(status: str, current_index: int) -> int:
    if status == _PROGRESS_DONE:
        return 100
    if current_index < 0:
        return 0
    unit = 100 / max(1, len(_PROGRESS_STEPS))
    if status in {_PROGRESS_ERROR, _PROGRESS_CANCELLED}:
        return round((current_index + 1) * unit)
    return round((current_index + 0.35) * unit)


def _write_progress(
    job_dir: Path,
    *,
    status: str,
    step: str,
    message: str,
    result_url: str | None = None,
    error: str | None = None,
    can_retry_metadata: bool = False,
    title: str | None = None,
    artist: str | None = None,
    notices: list[GenerationNotice] | None = None,
    include_notice_details: bool = True,
) -> None:
    if notices is None:
        existing_notices = _read_progress(job_dir).get("notices", [])
        notice_payloads = existing_notices if isinstance(existing_notices, list) else []
    else:
        notice_payloads = _notice_payloads(
            notices,
            include_details=include_notice_details,
        )
    write_json(
        job_dir / _PROGRESS_JSON,
        _progress_payload(
            status=status,
            step=step,
            message=message,
            result_url=result_url,
            error=error,
            can_retry_metadata=can_retry_metadata,
            title=title,
            artist=artist,
            notices=notice_payloads,
        ),
    )


def _read_progress(job_dir: Path) -> dict[str, object]:
    path = job_dir / _PROGRESS_JSON
    if not path.is_file():
        return _progress_payload(
            status=_PROGRESS_RUNNING,
            step="upload",
            message="任务已创建，正在准备接收音频。",
        )
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 9:
                raise
            sleep(0.01)
    raise RuntimeError("unreachable progress read retry state")
