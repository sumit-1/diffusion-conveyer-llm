from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import torch
from torch.utils.data import DataLoader

from conveyer.checkpoint import load_checkpoint
from conveyer.config import load_config
from conveyer.data import build_dataset_and_tokenizer, collate_conveyer_batch
from conveyer.eval import evaluate_crystallization_curve
from conveyer.infer import generate_conveyer
from conveyer.model import ConveyerDiffusionLM

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Conveyor-Belt Diffusion LLM — Elastic Planning & Oven Simulator</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #151b26;
    --panel-alt: #1b2332;
    --border: #2a3447;
    --text: #e6edf3;
    --muted: #8b949e;
    --front: #2ea043;
    --front-glow: rgba(46, 160, 67, 0.25);
    --mid: #d29922;
    --back: #8957e5;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 18px 24px 32px 24px;
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 14px;
  }
  h1 { margin: 0 0 4px 0; font-size: 22px; letter-spacing: -0.3px; }
  .subtitle { color: var(--muted); font-size: 13px; }
  .badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .metric-pill {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
  }
  .metric-pill strong { color: #3fb950; font-size: 14px; margin-left: 4px; }

  .prompt-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }
  .prompt-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .prompt-top label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; }
  .presets { display: flex; gap: 6px; flex-wrap: wrap; }
  .preset-btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .preset-btn:hover { border-color: #58a6ff; }
  .prompt-input-row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  #promptInput {
    flex: 1;
    min-width: 320px;
    background: #080b11;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 14px;
  }
  .knob {
    display: flex;
    flex-direction: column;
    font-size: 11px;
    color: var(--muted);
    gap: 3px;
  }
  .knob input {
    width: 72px;
    background: #080b11;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 6px 8px;
    font-size: 13px;
  }
  .btn-primary {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
  }
  .btn-primary:hover { background: #2ea043; }
  .btn-secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 14px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
  }

  .story-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }
  @media (max-width: 1000px) { .story-grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .card h3 {
    margin: 0 0 10px 0;
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .story-box {
    font-size: 17px;
    line-height: 1.65;
    background: #080b11;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 112px;
  }
  .story-prompt { color: #a5d6ff; }
  .story-gen { color: #3fb950; font-weight: 600; }
  .story-latest {
    background: rgba(63, 185, 80, 0.22);
    border-bottom: 2px solid #3fb950;
    padding: 1px 4px;
    border-radius: 4px;
  }
  .story-upcoming { color: #8b949e; font-style: italic; font-size: 14px; }

  .cryst-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 1px solid #1f2735;
  }
  .cryst-bar-wrap {
    width: 100px;
    height: 8px;
    background: #080b11;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-left: 8px;
  }
  .cryst-bar { height: 100%; background: #3fb950; }

  .oven-card {
    background: var(--panel);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 16px;
  }
  .oven-controls {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .conveyor-track-wrapper {
    position: relative;
    background: #070a0f;
    border: 2px solid #283246;
    border-radius: 10px;
    padding: 14px 12px;
    overflow-x: auto;
  }
  .conveyor-belt-row {
    display: flex;
    gap: 8px;
    align-items: stretch;
    min-width: max-content;
  }
  .dough-ball {
    width: 108px;
    flex-shrink: 0;
    background: #101622;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
  }
  .dough-ball.slot-zero {
    width: 128px;
    background: rgba(46, 160, 67, 0.14);
    border: 2px solid #3fb950;
    box-shadow: 0 0 14px var(--front-glow);
  }
  .dough-ball.zone-front { border-top: 3px solid var(--front); }
  .dough-ball.zone-mid   { border-top: 3px solid var(--mid); }
  .dough-ball.zone-back  { border-top: 3px solid var(--back); }

  .ball-top {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .ball-word {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 5px;
  }
  .ball-cand {
    font-size: 10px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.35;
  }
  .conf-bar {
    height: 4px;
    background: #1f2735;
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .conf-fill { height: 100%; background: #3fb950; }
  .zone-legend {
    display: flex;
    gap: 18px;
    font-size: 12px;
    color: var(--muted);
    margin-top: 10px;
    flex-wrap: wrap;
  }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
</style>
</head>
<body>

  <div class="header-row">
    <div>
      <h1>64-Slot Dual-Channel Elastic Conveyor-Belt Diffusion LLM</h1>
      <div class="subtitle" id="modelSubtitle">Loading model info...</div>
    </div>
    <div class="badge-row">
      <div class="metric-pill">Model Params: <strong id="paramBadge">--</strong></div>
      <div class="metric-pill">Slot-0 Val Accuracy: <strong id="s0AccBadge">--</strong></div>
      <div class="metric-pill">Back 75% Planning Pause: <strong id="pauseBadge">--</strong></div>
      <div class="metric-pill">Sample Token Match: <strong id="sampleAccBadge">--</strong></div>
    </div>
  </div>

  <!-- 1. Prompt Input & Controls -->
  <div class="prompt-card">
    <div class="prompt-top">
      <label>1. Enter Any Initial Text Prompt (Press Enter or Click "Bake &amp; Continue Text"):</label>
      <div class="presets">
        <button class="preset-btn" onclick="setPreset(0)">Preset 1: Lion in the Forest</button>
        <button class="preset-btn" onclick="setPreset(1)">Preset 2: Robot &amp; Lost Kitten</button>
        <button class="preset-btn" onclick="setPreset(2)">Preset 3: Melting Snowman</button>
        <button class="preset-btn" onclick="setPreset(3)">Preset 4: Missing Blue Crayon</button>
      </div>
    </div>
    <div class="prompt-input-row">
      <input type="text" id="promptInput" onkeydown="if(event.key==='Enter'){runGenerate(-1);}" value="Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion" />
      <div class="knob">
        <span>Pre-Bake Overlap (W)</span>
        <input type="number" id="overlapInput" value="16" min="0" max="60" />
      </div>
      <div class="knob">
        <span>Oven Slots (B)</span>
        <input type="number" id="bakeInput" value="96" min="1" max="96" />
      </div>
      <div class="knob">
        <span>Generate Tokens</span>
        <input type="number" id="tokensInput" value="64" min="6" max="200" />
      </div>
      <div class="knob">
        <span>Temp (&tau;)</span>
        <input type="number" id="tempInput" value="0.0" step="0.1" min="0.0" max="1.5" />
      </div>
      <button class="btn-primary" id="bakeBtn" onclick="runGenerate(-1)">Bake &amp; Continue Text</button>
    </div>
  </div>

  <!-- 2. Live Text Continuation + Accuracy Panel -->
  <div class="story-grid">
    <div class="card">
      <h3>
        <span>2. Live Text Continuation (Blue = Input Prompt &bull; Green = Popped Out of Slot 0)</span>
        <span id="tickModeLabel" style="color:#3fb950;"></span>
      </h3>
      <div class="story-box">
        <span id="storyPromptSpan" class="story-prompt"></span><span id="storyGenSpan" class="story-gen"></span><span id="storyLatestSpan" class="story-latest" style="display:none;"></span>
        <span id="storyPreviewSpan" class="story-upcoming"></span>
      </div>
      <div id="groundTruthBox" style="margin-top:10px; font-size:12px; color:var(--muted);"></div>
    </div>

    <div class="card">
      <h3>
        <span>Oven Crystallization &amp; Accuracy</span>
        <span style="font-size:11px; color:#58a6ff;">Back (k=63) &rarr; Front (k=0)</span>
      </h3>
      <div id="crystTable"></div>
    </div>
  </div>

  <!-- 3. Bottom Conveyor Belt Oven -->
  <div class="oven-card">
    <div class="oven-controls">
      <strong style="font-size:13px; text-transform:uppercase; color:var(--muted);">3. Live 64-Slot Conveyor &amp; Planning Belt (Bottom):</strong>
      <button class="btn-secondary" onclick="stepTick(-1)">&larr; Prev Tick</button>
      <button class="btn-primary" id="playBtn" onclick="togglePlay()" style="padding:7px 16px;">Pause</button>
      <button class="btn-secondary" onclick="stepTick(1)">Next Tick &rarr;</button>
      <label style="font-size:12px; color:var(--muted); display:flex; align-items:center; gap:8px; flex:1; min-width:220px;">
        <span id="scrubberLabel">Tick 0/0</span>
        <input type="range" id="tickSlider" min="0" max="0" value="0" style="flex:1;" oninput="onScrub(this.value)" />
      </label>
      <label style="font-size:12px; color:var(--muted);">
        Speed:
        <select id="speedSelect" onchange="updateSpeed()" style="background:#080b11; color:var(--text); border:1px solid var(--border); padding:4px; border-radius:4px;">
          <option value="450">Slow (2.2 Hz)</option>
          <option value="240" selected>Normal (4 Hz)</option>
          <option value="110">Fast (9 Hz)</option>
        </select>
      </label>
    </div>

    <div class="conveyor-track-wrapper">
      <div id="conveyorRow" class="conveyor-belt-row"></div>
    </div>

    <div class="zone-legend">
      <span><i class="dot" style="background:#3fb950;"></i><strong>Slot 0 (Front Exit, &alpha;<sub>0</sub> = 0):</strong> Confirmed Token Popping Out</span>
      <span><i class="dot" style="background:#2ea043;"></i><strong>Slots 1..15 (Front 25% Conveyor):</strong> Shifts Left Every Tick (100% Speed)</span>
      <span><i class="dot" style="background:#d29922;"></i><strong>Slot 15..20 (Mid-Belt Expansion):</strong> New Clauses Form Here</span>
      <span><i class="dot" style="background:#8957e5;"></i><strong>Slots 16..63 (Back 75% Elastic Planning):</strong> Pauses in Place Holding Long-Range Sentence Plan</span>
    </div>
  </div>

<script>
const PRESETS = [
  "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion",
  "Once upon a time, there was a little robot named Beep who lived in a noisy workshop. One rainy morning, Beep found a tiny lost kitten by the door. Beep",
  "One winter day, Max made a snowman in the yard with a carrot nose. When the warm sun came out in the afternoon, the snowman began to",
  "Lucy had a box of bright crayons, but she could not find the blue crayon to color the sky. She looked under her bed and"
];

let traceData = null;
let curFrame = 0;
let playing = true;
let intervalMs = 240;
let timerHandle = null;

function setPreset(idx) {
  const inp = document.getElementById('promptInput');
  inp.value = PRESETS[idx];
  runGenerate(-1);
}

async function runGenerate(sampleOverride = -1) {
  const btn = document.getElementById('bakeBtn');
  btn.textContent = "Baking...";
  const promptText = document.getElementById('promptInput').value.trim();
  const ov = document.getElementById('overlapInput').value;
  const bs = document.getElementById('bakeInput').value;
  const mt = document.getElementById('tokensInput').value;
  const tp = document.getElementById('tempInput').value;

  let url = `/api/trace?overlap=${ov}&bake_steps=${bs}&tokens=${mt}&temp=${tp}&_t=${Date.now()}`;
  if (sampleOverride >= 0 || promptText === "") {
    url += `&sample=${Math.max(0, sampleOverride)}`;
  } else {
    url += `&prompt=${encodeURIComponent(promptText)}`;
  }

  const resp = await fetch(url, { cache: "no-store" });
  traceData = await resp.json();
  btn.textContent = "Bake & Continue Text";

  if (sampleOverride >= 0 || promptText === "") {
    document.getElementById('promptInput').value = traceData.prompt;
  }

  document.getElementById('modelSubtitle').textContent =
    `Checkpoint: ${traceData.ckpt_name} | Tokenizer: Tiktoken BPE (${traceData.vocab_size} vocab) | Input: [All Prefix + 64 Belt] -> Output: [1 Token + 64 Belt]`;
  document.getElementById('paramBadge').textContent = `${traceData.params_m}M`;
  document.getElementById('s0AccBadge').textContent = `${(traceData.val_slot0_acc * 100).toFixed(1)}%`;
  document.getElementById('sampleAccBadge').textContent =
    traceData.sample_token_acc !== null ? `${(traceData.sample_token_acc * 100).toFixed(1)}%` : "Custom Input";

  renderCrystallizationTable(traceData.crystallization);

  // Jump straight to the first live emitted token tick so the user immediately sees their prompt continuation
  let firstEmitIdx = 0;
  for (let i = 0; i < traceData.snapshots.length; i++) {
    if (!traceData.snapshots[i].is_warmup) {
      firstEmitIdx = i;
      break;
    }
  }
  curFrame = firstEmitIdx;
  const slider = document.getElementById('tickSlider');
  slider.max = Math.max(0, traceData.snapshots.length - 1);
  slider.value = curFrame;
  renderFrame(curFrame);
}

function renderCrystallizationTable(cryst) {
  const container = document.getElementById('crystTable');
  if (!cryst || !cryst.length) {
    container.innerHTML = '<div style="color:var(--muted);font-size:12px;">No crystallization stats available.</div>';
    return;
  }
  container.innerHTML = cryst.map(row => {
    const pct = Math.round(row.acc * 100);
    return `
      <div class="cryst-row">
        <span><strong>Slot ${row.slot}</strong> (${row.zone})</span>
        <span>
          H=${row.entropy.toFixed(2)} | <strong>${pct}%</strong>
          <span class="cryst-bar-wrap"><span class="cryst-bar" style="width:${pct}%"></span></span>
        </span>
      </div>
    `;
  }).join('');
}

function onScrub(val) {
  curFrame = parseInt(val, 10);
  renderFrame(curFrame);
}

function stepTick(delta) {
  if (!traceData || !traceData.snapshots.length) return;
  playing = false;
  document.getElementById('playBtn').textContent = 'Play';
  curFrame = (curFrame + delta + traceData.snapshots.length) % traceData.snapshots.length;
  document.getElementById('tickSlider').value = curFrame;
  renderFrame(curFrame);
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
}

function updateSpeed() {
  intervalMs = parseInt(document.getElementById('speedSelect').value, 10);
  if (timerHandle) clearInterval(timerHandle);
  timerHandle = setInterval(tickLoop, intervalMs);
}

function renderFrame(idx) {
  if (!traceData || !traceData.snapshots.length) return;
  const snap = traceData.snapshots[idx];
  document.getElementById('scrubberLabel').textContent =
    `Tick ${idx + 1}/${traceData.snapshots.length} (Pos t=${snap.tick})`;
  document.getElementById('pauseBadge').textContent = `${Math.round(snap.back_pause_pct)}% Paused`;

  document.getElementById('tickModeLabel').textContent = snap.is_warmup
    ? `[PRE-BAKING WARMUP: Clamping Input "${snap.emitted_piece}" while Belt Plans Ahead]`
    : `[LIVE BAKING: Slot 0 Popped "${snap.emitted_piece}" | Back 75% Belt ${Math.round(snap.back_pause_pct)}% Paused]`;

  document.getElementById('storyPromptSpan').textContent = traceData.prompt;

  const genPieces = [];
  for (let i = 0; i <= idx; i++) {
    if (!traceData.snapshots[i].is_warmup) {
      genPieces.push(traceData.snapshots[i].emitted_piece);
    }
  }

  const latestSpan = document.getElementById('storyLatestSpan');
  if (genPieces.length > 0) {
    const prevText = genPieces.slice(0, -1).join('');
    const lastPiece = genPieces[genPieces.length - 1];
    document.getElementById('storyGenSpan').textContent = prevText;
    latestSpan.style.display = 'inline';
    latestSpan.textContent = lastPiece;
  } else {
    document.getElementById('storyGenSpan').textContent = "";
    latestSpan.style.display = 'none';
  }

  const previewWords = snap.slots.slice(1, 7).map(s => s.piece).join('');
  document.getElementById('storyPreviewSpan').textContent = `  [next on belt: ${previewWords} ...]`;

  const gtBox = document.getElementById('groundTruthBox');
  if (traceData.ground_truth_continuation) {
    gtBox.innerHTML = `<strong>Reference Continuation:</strong> ${traceData.ground_truth_continuation}`;
  } else {
    gtBox.innerHTML = `<strong>Dynamic Prompt Active:</strong> Continuing "${traceData.prompt.slice(0, 64)}..." in real time`;
  }

  const row = document.getElementById('conveyorRow');
  row.innerHTML = '';
  for (let k = 0; k < snap.slots.length; k++) {
    const s = snap.slots[k];
    const zoneClass = k === 0 ? 'slot-zero zone-front' : (k < 16 ? 'zone-front' : (k < 22 ? 'zone-mid' : 'zone-back'));
    const pct = Math.round(s.prob * 100);
    const pausePct = Math.round((1.0 - s.shift_prob) * 100);
    const div = document.createElement('div');
    div.className = `dough-ball ${zoneClass}`;
    const label = k === 0 ? 'SLOT 0 (POP)' : (k === 15 ? 'SLOT 15 (EXPAND)' : (k >= 16 ? `Plan ${k}` : `Conv ${k}`));
    const top2 = s.top3.slice(1, 3).map(c => `${c.w.trim() || c.w}(${Math.round(c.p*100)}%)`).join(', ');
    const motionTag = k < 16 ? `Shift 100%` : `Pause ${pausePct}%`;
    div.innerHTML = `
      <div class="ball-top"><span>${label}</span><span>${pct}%</span></div>
      <div class="ball-word">${s.piece.trim() || s.piece}</div>
      <div class="ball-cand">2nd: ${top2 || '-'}</div>
      <div class="ball-cand">${motionTag} | H=${s.entropy.toFixed(1)}</div>
      <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
    `;
    row.appendChild(div);
  }
}

function tickLoop() {
  if (playing && traceData && traceData.snapshots.length > 0) {
    curFrame = (curFrame + 1) % traceData.snapshots.length;
    document.getElementById('tickSlider').value = curFrame;
    renderFrame(curFrame);
  }
}

timerHandle = setInterval(tickLoop, intervalMs);
runGenerate(-1);
</script>
</body>
</html>
"""


class ConveyerWebApp:
    def __init__(
        self,
        model: ConveyerDiffusionLM,
        cfg: Any,
        ckpt_path: str,
        ckpt_meta: dict[str, Any],
        val_ds: Any,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.ckpt_meta = ckpt_meta
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.device = device
        self.cached_cryst = self._compute_initial_stats()

    def _compute_initial_stats(self) -> list[dict[str, Any]]:
        metrics = self.ckpt_meta.get("metrics", {})
        acc_curve = metrics.get("val_per_slot_acc")
        ent_curve = metrics.get("val_per_slot_entropy")
        if not acc_curve or not ent_curve:
            val_loader = DataLoader(
                self.val_ds,
                batch_size=min(32, len(self.val_ds)),
                shuffle=False,
                collate_fn=lambda b: collate_conveyer_batch(b, pad_id=self.tokenizer.pad_id),
            )
            c = evaluate_crystallization_curve(
                self.model, self.cfg, val_loader, self.device, pad_id=self.tokenizer.pad_id, max_batches=2
            )
            acc_curve = c["per_slot_acc"]
            ent_curve = c["per_slot_entropy"]

        B = len(acc_curve)
        slots_to_show = [0, 3, 8, 15, 24, 36, 48, B - 1]
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for k in slots_to_show:
            k = min(k, B - 1)
            if k in seen:
                continue
            seen.add(k)
            zone = (
                "Slot 0 Pop"
                if k == 0
                else ("Front Conveyor" if k < 16 else ("Mid Expansion" if k == 15 else "Back 75% Plan"))
            )
            rows.append(
                {
                    "slot": k,
                    "zone": zone,
                    "acc": float(acc_curve[k]),
                    "entropy": float(ent_curve[k]),
                }
            )
        return rows

    def compute_trace(
        self,
        prompt_text: str | None,
        sample_idx: int,
        overlap: int,
        bake_steps: int,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        gt_continuation: str | None = None
        gt_target_ids: list[int] = []

        if prompt_text is not None and prompt_text.strip() != "":
            encoded = self.tokenizer.encode(prompt_text.strip())
            prompt_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
            for i in range(min(300, len(self.val_ds))):
                seq = self.val_ds[i]["input_ids"]
                if seq[: len(encoded)].tolist() == encoded:
                    rem = [
                        int(t)
                        for t in seq[len(encoded) :].tolist()
                        if int(t) not in (self.tokenizer.pad_id, self.tokenizer.eos_id)
                    ]
                    gt_target_ids = rem[:max_tokens]
                    gt_continuation = self.tokenizer.decode(gt_target_ids)
                    break
        else:
            idx = max(0, min(sample_idx, len(self.val_ds) - 1))
            item = self.val_ds[idx]
            plen = int(item["prompt_len"])
            seq = item["input_ids"]
            prompt_ids = seq[:plen].unsqueeze(0).to(self.device)
            rem = [int(t) for t in seq[plen:].tolist() if int(t) not in (self.tokenizer.pad_id, self.tokenizer.eos_id)]
            gt_target_ids = rem[:max_tokens]
            gt_continuation = self.tokenizer.decode(gt_target_ids)

        full_out, snapshots = generate_conveyer(
            model=self.model,
            prompt_tokens=prompt_ids,
            max_new_tokens=max_tokens,
            prompt_overlap=overlap,
            bake_steps=bake_steps,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_id,
            deterministic_spawn=(temperature <= 1e-5),
        )

        plen_actual = prompt_ids.size(1)
        gen_ids = full_out[0, plen_actual:].tolist()

        sample_token_acc: float | None = None
        if gt_target_ids:
            comp_len = min(len(gen_ids), len(gt_target_ids))
            if comp_len > 0:
                matches = sum(1 for a, b in zip(gen_ids[:comp_len], gt_target_ids[:comp_len]) if a == b)
                sample_token_acc = matches / comp_len

        F_slots = min(self.model.front_slots, bake_steps)
        serialized_snaps: list[dict[str, Any]] = []
        for s in snapshots:
            slots_info = []
            B = len(s.slot_top1_tokens)
            for k in range(B):
                top3_list = [
                    {"w": self.tokenizer.id_to_piece(tid), "p": round(float(pr), 4)}
                    for tid, pr in zip(s.slot_top3_tokens[k], s.slot_top3_probs[k])
                ]
                slots_info.append(
                    {
                        "k": k,
                        "piece": self.tokenizer.id_to_piece(s.slot_top1_tokens[k]),
                        "prob": round(float(s.slot_top1_probs[k]), 4),
                        "entropy": round(float(s.slot_entropies[k]), 3),
                        "gate": round(float(s.slot_gate_means[k]), 3),
                        "shift_prob": round(float(s.slot_shift_probs[k]), 3),
                        "top3": top3_list,
                    }
                )
            back_pause = (
                100.0 * (1.0 - sum(s.slot_shift_probs[F_slots:]) / max(1, B - F_slots))
                if B > F_slots
                else 0.0
            )
            serialized_snaps.append(
                {
                    "tick": s.tick,
                    "is_warmup": s.is_warmup,
                    "emitted_piece": self.tokenizer.id_to_piece(s.emitted_token),
                    "mean_gate": sum(s.slot_gate_means) / max(1, B),
                    "back_pause_pct": back_pause,
                    "slots": slots_info,
                }
            )

        metrics = self.ckpt_meta.get("metrics", {})
        val_s0_acc = float(metrics.get("val_slot0_acc", self.cached_cryst[0]["acc"] if self.cached_cryst else 0.0))

        return {
            "ckpt_name": self.ckpt_path,
            "dataset": self.cfg.data.dataset,
            "vocab_size": self.tokenizer.vocab_size,
            "params_m": f"{self.model.num_parameters() / 1e6:.2f}",
            "val_slot0_acc": val_s0_acc,
            "sample_token_acc": sample_token_acc,
            "prompt": self.tokenizer.decode(prompt_ids[0].tolist()),
            "ground_truth_continuation": gt_continuation,
            "crystallization": self.cached_cryst,
            "snapshots": serialized_snaps,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Web Simulator for the 64-Slot Conveyor-Belt Diffusion LLM")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/tinystories.yaml", help="Fallback config if no checkpoint")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8766")))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if Path("runs/tinystories/best.pt").exists():
            ckpt_path = "runs/tinystories/best.pt"
        elif (repo_root / "runs/tinystories/best.pt").exists():
            ckpt_path = str(repo_root / "runs/tinystories/best.pt")
        elif Path("runs/synthetic/best.pt").exists():
            ckpt_path = "runs/synthetic/best.pt"
        elif (repo_root / "runs/synthetic/best.pt").exists():
            ckpt_path = str(repo_root / "runs/synthetic/best.pt")
        else:
            ckpt_path = str(repo_root / "runs/tinystories/best.pt")

    try:
        model, cfg, ckpt_meta = load_checkpoint(ckpt_path, device=device)
    except Exception:
        cfg = load_config(args.config)
        model = ConveyerDiffusionLM(cfg.model).to(device)
        ckpt_path = "untrained"
        ckpt_meta = {}

    _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg, infer_only=True)
    if hasattr(tokenizer, "content_vocab_mask"):
        model.content_vocab_mask.copy_(getattr(tokenizer, "content_vocab_mask").to(device))
    app = ConveyerWebApp(
        model=model,
        cfg=cfg,
        ckpt_path=ckpt_path,
        ckpt_meta=ckpt_meta,
        val_ds=val_ds,
        tokenizer=tokenizer,
        device=device,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/trace":
                qs = parse_qs(parsed.query)
                prompt_text = qs.get("prompt", [None])[0]
                sample_idx = int(qs.get("sample", ["0"])[0])
                overlap = int(qs.get("overlap", ["16"])[0])
                bake_steps = int(qs.get("bake_steps", [str(model.belt_size)])[0])
                tokens = int(qs.get("tokens", ["64"])[0])
                temp = float(qs.get("temp", ["0.0"])[0])
                payload = json.dumps(
                    app.compute_trace(prompt_text, sample_idx, overlap, bake_steps, tokens, temperature=temp)
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Conveyor-Belt Diffusion LLM Web Simulator running at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
