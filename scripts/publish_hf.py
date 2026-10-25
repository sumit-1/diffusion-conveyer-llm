from __future__ import annotations

import json
from pathlib import Path
import tempfile
import torch
from huggingface_hub import HfApi

from conveyer.checkpoint import load_checkpoint
from conveyer.data import build_dataset_and_tokenizer
from conveyer.web_sim import HTML_PAGE, ConveyerWebApp


def main() -> None:
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/diffusion-conveyer-llm"

    traces_file = Path("runs/tinystories/static_space_traces.json")
    if traces_file.exists():
        traces = json.loads(traces_file.read_text(encoding="utf-8"))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, ckpt_meta = load_checkpoint("runs/tinystories/best.pt", device=device)
        _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg, infer_only=True)
        app = ConveyerWebApp(
            model=model,
            cfg=cfg,
            ckpt_path="runs/tinystories/best.pt (step 122,743 · 24h)",
            ckpt_meta=ckpt_meta,
            val_ds=val_ds,
            tokenizer=tokenizer,
            device=device,
        )
        prompts = [
            "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion",
            "Once upon a time, there was a little robot named Beep who lived in a noisy workshop. One rainy morning, Beep found a tiny lost kitten by the door. Beep",
            "One winter day, Max made a snowman in the yard with a carrot nose. When the warm sun came out in the afternoon, the snowman began to",
            "Lucy had a box of bright crayons, but she could not find the blue crayon to color the sky. She looked under her bed and",
            "Sam and his sister Mia built a huge sandcastle on the beach near the waves. Suddenly, a big wave splashed toward their castle. Sam quickly",
            "In a quiet village, an old baker dropped his round loaf of bread and it rolled out the door down the hill. A hungry puppy saw the bread and",
        ]
        traces = {
            p: app.compute_trace(p, 0, overlap=12, bake_steps=64, max_tokens=40, temperature=0.0)
            for p in prompts
        }

    prompts = list(traces.keys())
    static_js = f"const STATIC_TRACES = {json.dumps(traces)};\nconst PRESETS = {json.dumps(prompts)};"
    html_static = HTML_PAGE

    old_btns = (
        '        <button class="preset-btn" onclick="setPreset(0)">Preset 1: Lion in the Forest</button>\n'
        '        <button class="preset-btn" onclick="setPreset(1)">Preset 2: Robot &amp; Lost Kitten</button>\n'
        '        <button class="preset-btn" onclick="setPreset(2)">Preset 3: Melting Snowman</button>\n'
        '        <button class="preset-btn" onclick="setPreset(3)">Preset 4: Missing Blue Crayon</button>'
    )
    new_btns = (
        '        <button class="preset-btn" onclick="setPreset(0)">1: Lion in the Forest</button>\n'
        '        <button class="preset-btn" onclick="setPreset(1)">2: Robot &amp; Lost Kitten</button>\n'
        '        <button class="preset-btn" onclick="setPreset(2)">3: Melting Snowman</button>\n'
        '        <button class="preset-btn" onclick="setPreset(3)">4: Missing Blue Crayon</button>\n'
        '        <button class="preset-btn" onclick="setPreset(4)">5: Beach Sandcastle Wave</button>\n'
        '        <button class="preset-btn" onclick="setPreset(5)">6: Baker &amp; Hungry Puppy</button>'
    )
    html_static = html_static.replace(old_btns, new_btns)

    old_presets_start = html_static.index("const PRESETS = [")
    old_presets_end = html_static.index("];", old_presets_start) + 2
    html_static = html_static[:old_presets_start] + static_js + html_static[old_presets_end:]

    old_fetch = '  const resp = await fetch(url, { cache: "no-store" });\n  traceData = await resp.json();'
    new_fetch = (
        "  try {\n"
        '    const resp = await fetch(url, { cache: "no-store" });\n'
        '    if (!resp.ok) throw new Error("static mode");\n'
        "    traceData = await resp.json();\n"
        "  } catch (e) {\n"
        "    traceData = STATIC_TRACES[promptText] || STATIC_TRACES[PRESETS[0]];\n"
        "  }"
    )
    html_static = html_static.replace(old_fetch, new_fetch)

    space_readme = (
        Path("README.md")
        .read_text(encoding="utf-8")
        .replace("sdk: docker\napp_port: 7860", "sdk: static\napp_file: index.html")
    )

    with tempfile.TemporaryDirectory() as tmpd:
        tp = Path(tmpd)
        (tp / "index.html").write_text(html_static, encoding="utf-8")
        (tp / "README.md").write_text(space_readme, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(tp / "README.md"), path_in_repo="README.md", repo_id=repo_id, repo_type="space")
        api.upload_file(path_or_fileobj=str(tp / "index.html"), path_in_repo="index.html", repo_id=repo_id, repo_type="space")

    traces_file.unlink(missing_ok=True)
    print(f"Published interactive Space: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
