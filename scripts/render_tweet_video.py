"""
Renders a high-resolution 1280x720 MP4 video + animated GIF of the 96-Slot Conveyor-Belt Diffusion LLM
generating a story tick-by-tick (showing both the live generated text AND the moving conveyor belt slots).
"""

from __future__ import annotations

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from conveyer.checkpoint import load_checkpoint
from conveyer.data import build_dataset_and_tokenizer
from conveyer.infer import stream_conveyer_generation


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates = ["consolab.ttf" if bold else "consola.ttf", "courbd.ttf" if bold else "cour.ttf"]
    else:
        candidates = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in candidates:
        for folder in ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"]:
            p = Path(folder) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size=size)
                except Exception:
                    pass
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_frame(
    tick_idx: int,
    total_ticks: int,
    snap: object,
    prompt_str: str,
    emitted_str: str,
    latest_piece: str,
    slot_pieces: list[str],
    slot_probs: list[float],
    slot_shifts: list[float],
    slot_top3: list[list[tuple[str, float]]],
    dragon_snap_info: dict | None = None,
) -> np.ndarray:
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), (11, 15, 23))
    draw = ImageDraw.Draw(img)

    f_title = _load_font(22, bold=True)
    f_sub = _load_font(14, bold=False)
    f_sec = _load_font(15, bold=True)
    f_story = _load_font(17, bold=False)
    f_story_b = _load_font(17, bold=True)
    f_mono = _load_font(14, bold=True, mono=True)
    f_mono_sm = _load_font(12, bold=False, mono=True)
    f_badge = _load_font(13, bold=True)

    # 1. Top Header Bar
    draw.rectangle([0, 0, W, 66], fill=(18, 24, 36))
    draw.line([0, 66, W, 66], fill=(42, 52, 71), width=1)
    draw.text(
        (24, 12),
        "Conveyor-Belt Diffusion LLM — 96-Slot Rolling Dough-Ball & Duplex Frame Inspector",
        fill=(230, 237, 243),
        font=f_title,
    )
    draw.text(
        (24, 40),
        "30.47M Params  •  Last 5 Slots (0..4): Stopword & Grammar Filling  •  Slots 5..95: Content-Only Rolling Conveyer",
        fill=(139, 148, 158),
        font=f_sub,
    )

    # Right header pill
    is_warmup = getattr(snap, "is_warmup", False)
    mode_txt = f"TICK {getattr(snap, 'tick', 0):3d}  |  {'PRE-BAKE WARMUP' if is_warmup else 'LIVE GENERATION'}"
    pill_bg = (31, 111, 235) if is_warmup else (35, 134, 54)
    draw.rounded_rectangle([W - 310, 16, W - 24, 50], radius=6, fill=pill_bg)
    draw.text((W - 294, 24), mode_txt, fill=(255, 255, 255), font=f_badge)

    # 2. Story Continuation Panel (Top Half)
    draw.rounded_rectangle([24, 80, W - 24, 248], radius=10, fill=(21, 27, 38), outline=(42, 52, 71), width=1)
    draw.text(
        (40, 92),
        "1. LIVE STORY STREAM  (Blue = Input Prompt   •   Green = Emitted from Slot 0   •   Bright Pill = Current Tick Pop)",
        fill=(139, 148, 158),
        font=f_sec,
    )

    # Render wrapped story with color-coded prompt vs generated
    full_story = prompt_str + emitted_str
    lines = _wrap_text(full_story, f_story, W - 360, draw)
    y_text = 118
    chars_drawn = 0
    prompt_len_chars = len(prompt_str)
    for ln in lines[:4]:
        x_cur = 40
        for word in ln.split(" "):
            w_space = word + " "
            is_prompt_word = (chars_drawn + len(word) // 2) < prompt_len_chars
            color = (121, 192, 255) if is_prompt_word else (126, 231, 135)
            draw.text((x_cur, y_text), w_space, fill=color, font=f_story if is_prompt_word else f_story_b)
            bbox = draw.textbbox((0, 0), w_space, font=f_story if is_prompt_word else f_story_b)
            x_cur += bbox[2] - bbox[0]
            chars_drawn += len(w_space)
        y_text += 28

    # Popped token callout box on the right of the story card
    draw.rounded_rectangle([W - 290, 88, W - 38, 142], radius=6, fill=(13, 18, 28), outline=(46, 160, 67), width=2)
    draw.text((W - 278, 94), "POPPED AT SLOT 0:", fill=(139, 148, 158), font=f_mono_sm)
    draw.text((W - 278, 112), f"{repr(latest_piece)}", fill=(63, 185, 80), font=f_title)

    # 3. Zone A: Last-5 Stopword & Grammar Filling Slots (Slots 0..4)
    draw.rounded_rectangle([24, 260, W - 24, 430], radius=10, fill=(21, 27, 38), outline=(42, 52, 71), width=1)
    draw.text(
        (40, 272),
        "2. LAST-5 STOPWORD & GRAMMAR FILLING ZONE (Slots k = 0 .. 4)  —  Full 4,096 Vocab  •  Shifts Left 1 Slot / Tick",
        fill=(63, 185, 80),
        font=f_sec,
    )

    slot_w = 232
    for k in range(5):
        sx = 40 + k * (slot_w + 10)
        sy = 300
        sh = 116
        border_col = (63, 185, 80) if k == 0 else (46, 110, 60)
        bg_col = (18, 36, 26) if k == 0 else (16, 22, 32)
        draw.rounded_rectangle([sx, sy, sx + slot_w, sy + sh], radius=8, fill=bg_col, outline=border_col, width=2 if k == 0 else 1)

        label = "SLOT 0  [POP -> KV]" if k == 0 else (f"SLOT {k}  [FILLING]" if k < 4 else "SLOT 4  [<- DOUGH]")
        draw.text((sx + 12, sy + 8), label, fill=(126, 231, 135) if k == 0 else (139, 148, 158), font=f_mono_sm)
        p_val = slot_probs[k] * 100.0
        draw.text((sx + slot_w - 58, sy + 8), f"{p_val:4.0f}%", fill=(63, 185, 80), font=f_mono_sm)

        word_disp = slot_pieces[k].replace("\n", " ")
        if not word_disp.strip():
            word_disp = repr(word_disp)
        draw.text((sx + 12, sy + 30), word_disp[:14], fill=(255, 255, 255), font=f_title)

        # Probability bar
        bar_w = int((slot_w - 24) * min(1.0, max(0.05, slot_probs[k])))
        draw.rectangle([sx + 12, sy + 64, sx + slot_w - 12, sy + 70], fill=(30, 38, 52))
        draw.rectangle([sx + 12, sy + 64, sx + 12 + bar_w, sy + 70], fill=(46, 160, 67))

        # Top-3 candidates
        t3_str = " · ".join(f"{w.strip() or repr(w)}({p*100:.0f}%)" for w, p in slot_top3[k][:2])
        draw.text((sx + 12, sy + 82), f"alt: {t3_str[:24]}", fill=(139, 148, 158), font=f_mono_sm)

        # Arrow pointing left between slots
        if k > 0:
            ax = sx - 8
            ay = sy + sh // 2
            draw.text((ax - 2, ay - 10), "←", fill=(63, 185, 80), font=f_sec)

    # 4. Zone B: Content-Only Rolling Dough-Ball Conveyer (Slots 5..20+)
    draw.rounded_rectangle([24, 442, W - 24, 608], radius=10, fill=(21, 27, 38), outline=(42, 52, 71), width=1)
    mean_pause = 100.0 * (1.0 - sum(slot_shifts[5:]) / max(1, len(slot_shifts) - 5))
    draw.text(
        (40, 454),
        f"3. CONTENT-ONLY DOUGH-BALL CONVEYER (Slots k = 5 .. 95)  —  Nouns/Verbs/Adjectives Only (Stopwords Stripped)  •  Pause: {mean_pause:.0f}%",
        fill=(210, 153, 34),
        font=f_sec,
    )

    conv_slots_to_show = [5, 6, 7, 8, 10, 12, 16, 24, 36, 60]
    c_w = 112
    for idx_c, k in enumerate(conv_slots_to_show):
        sx = 40 + idx_c * (c_w + 8)
        sy = 482
        sh = 112
        is_duplex_head = k < 9  # K=4 Duplex-SimMT active slots (5, 6, 7, 8)
        border_col = (210, 153, 34) if is_duplex_head else (137, 87, 229)
        bg_col = (34, 28, 18) if is_duplex_head else (22, 18, 36)
        draw.rounded_rectangle([sx, sy, sx + c_w, sy + sh], radius=7, fill=bg_col, outline=border_col, width=2 if is_duplex_head else 1)

        tag = f"k={k} [DUPLEX]" if is_duplex_head else f"k={k} [PLAN]"
        draw.text((sx + 8, sy + 6), tag, fill=border_col, font=f_mono_sm)

        # Show top distinct content candidate across top-3 so neighboring slots display their rich vocabulary
        chosen_w = slot_pieces[k].strip()
        chosen_p = slot_probs[k]
        if idx_c > 0 and chosen_w == slot_pieces[conv_slots_to_show[idx_c - 1]].strip() and len(slot_top3[k]) > 1:
            chosen_w = slot_top3[k][idx_c % len(slot_top3[k])][0].strip()
            chosen_p = slot_top3[k][idx_c % len(slot_top3[k])][1]

        draw.text((sx + 8, sy + 30), chosen_w[:10], fill=(240, 246, 252), font=f_mono)
        draw.text((sx + 8, sy + 54), f"p={chosen_p*100:4.1f}%", fill=(139, 148, 158), font=f_mono_sm)
        draw.text((sx + 8, sy + 72), f"roll={slot_shifts[k]*100:2.0f}%", fill=border_col, font=f_mono_sm)
        alt_w = slot_top3[k][1][0].strip() if len(slot_top3[k]) > 1 else ""
        draw.text((sx + 8, sy + 90), f"+ {alt_w[:9]}", fill=(110, 118, 129), font=f_mono_sm)

    # 5. Bottom Comparison Bar: Conveyer "Dragon Injection" Steering Test
    draw.rounded_rectangle([24, 620, W - 24, 704], radius=10, fill=(24, 20, 34), outline=(137, 87, 229), width=1)
    draw.text(
        (40, 628),
        "4. CONVEYER STEERING TEST: Same Prompt ('...Suddenly a big lion'), but with 'dragon' injected into Content Conveyer Slots 5..8:",
        fill=(188, 140, 255),
        font=f_sec,
    )
    if dragon_snap_info is not None:
        d_text = dragon_snap_info["text"]
        draw.text(
            (40, 654),
            f"...Suddenly a big lion -> {d_text[:118]}",
            fill=(230, 237, 243),
            font=f_story_b,
        )
        draw.text(
            (40, 680),
            "github.com/sumit-1/diffusion-conveyer-llm   •   huggingface.co/spaces/sumitk/diffusion-conveyer-llm",
            fill=(139, 148, 158),
            font=f_mono_sm,
        )

    return np.asarray(img)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint("runs/tinystories/best.pt", device=device)
    _, _, tokenizer = build_dataset_and_tokenizer(cfg, infer_only=True)
    if hasattr(tokenizer, "content_vocab_mask"):
        model.content_vocab_mask.copy_(tokenizer.content_vocab_mask.to(device))

    prompt = (
        "Once upon a time, Lily and a little bunny walked into the green forest "
        "to find the golden key near the locked treasure chest. Suddenly a big lion"
    )
    p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    # 1. Collect normal unsteered trace
    snaps = []
    for _, snap in stream_conveyer_generation(
        model=model,
        prompt_tokens=p_ids,
        max_new_tokens=44,
        prompt_overlap=12,
        bake_steps=model.belt_size,
        temperature=0.0,
        eos_token_id=tokenizer.eos_id,
        deterministic_spawn=True,
    ):
        snaps.append(snap)

    # 2. Collect dragon-steered trace for the bottom comparison bar
    dragon_id = tokenizer.encode(" dragon")[1]
    dragon_snaps = []
    for _, d_snap in stream_conveyer_generation(
        model=model,
        prompt_tokens=p_ids,
        max_new_tokens=44,
        prompt_overlap=12,
        bake_steps=model.belt_size,
        temperature=0.0,
        eos_token_id=tokenizer.eos_id,
        deterministic_spawn=True,
        injected_conveyer_tokens=[dragon_id] * 4,
    ):
        dragon_snaps.append(d_snap)

    out_dir = Path("assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "conveyer_generation_demo.mp4"
    gif_path = out_dir / "conveyer_generation_demo.gif"
    preview_png = out_dir / "conveyer_generation_frame.png"

    frames: list[np.ndarray] = []
    emitted_ids: list[int] = []
    dragon_emitted_ids: list[int] = []

    for idx, snap in enumerate(snaps):
        if not snap.is_warmup:
            emitted_ids.append(snap.emitted_token)
        if idx < len(dragon_snaps) and not dragon_snaps[idx].is_warmup:
            dragon_emitted_ids.append(dragon_snaps[idx].emitted_token)

        emitted_str = tokenizer.decode(emitted_ids)
        dragon_str = tokenizer.decode(dragon_emitted_ids)
        latest_piece = tokenizer.id_to_piece(snap.emitted_token)
        slot_pieces = [tokenizer.id_to_piece(tid) for tid in snap.slot_top1_tokens]
        slot_top3 = [
            [(tokenizer.id_to_piece(tid), pr) for tid, pr in zip(snap.slot_top3_tokens[k], snap.slot_top3_probs[k])]
            for k in range(len(snap.slot_top1_tokens))
        ]

        frame = render_frame(
            tick_idx=idx,
            total_ticks=len(snaps),
            snap=snap,
            prompt_str=prompt,
            emitted_str=emitted_str,
            latest_piece=latest_piece,
            slot_pieces=slot_pieces,
            slot_probs=snap.slot_top1_probs,
            slot_shifts=snap.slot_shift_probs,
            slot_top3=slot_top3,
            dragon_snap_info={"text": dragon_str},
        )
        # Hold each generation tick for 3 frames at 10 fps (0.3s per token, warmup faster at 1 frame)
        repeat = 1 if snap.is_warmup else 3
        for _ in range(repeat):
            frames.append(frame)

        if idx == len(snaps) - 10:
            Image.fromarray(frame).save(preview_png)

    # Hold final frame for 2 seconds
    for _ in range(20):
        frames.append(frames[-1])

    with imageio.get_writer(str(mp4_path), fps=10, codec="libx264", quality=8, pixelformat="yuv420p") as writer:
        for f in frames:
            writer.append_data(f)

    # Also save a lightweight GIF preview
    gif_frames = [Image.fromarray(f).resize((960, 540), Image.Resampling.LANCZOS) for f in frames[::2]]
    gif_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=gif_frames[1:],
        duration=180,
        loop=0,
    )
    print(f"Saved MP4 video : {mp4_path.resolve()} ({mp4_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved GIF video : {gif_path.resolve()} ({gif_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved PNG frame : {preview_png.resolve()}")


if __name__ == "__main__":
    main()
