from __future__ import annotations

import torch
import torch.nn.functional as F

from conveyer.checkpoint import load_checkpoint
from conveyer.data import build_dataset_and_tokenizer
from conveyer.data.tinystories import load_or_prepare_tinystories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg, _ = load_checkpoint("runs/tinystories/best.pt", device=device)
model.eval()
_, val_ds, tok = build_dataset_and_tokenizer(cfg)

prompt = (
    "Once upon a time, Lily and a little bunny walked into the green forest to findthe golden key "
    "near the locked treasure chest. Suddenly a big lion "
)

print("=== 1. Tokenization of user prompt ===")
ids = tok.encode(prompt)
print([tok.id_to_piece(i) for i in ids])

print("\n=== 2. Training corpus composition ===")
stories = load_or_prepare_tinystories(cfg.data.data_dir, max_stories=cfg.data.num_train + cfg.data.num_val)
templ = sum(1 for s in stories if s.startswith("Once upon a time, ") and " walked into the " in s and "Together, they" in s)
print(f"total={len(stories)} template={templ} ({100*templ/len(stories):.1f}%) real={len(stories)-templ}")
for word in ["lion", "Suddenly", "Lily"]:
    n = sum(1 for s in stories if word in s)
    print(f"  stories containing {word!r}: {n}")
real_lily = sum(1 for s in stories if "Lily" in s and "Together, they" not in s)
print(f"  real (non-template) stories containing 'Lily': {real_lily}")


@torch.no_grad()
def slot0_dist(prefix_ids: list[int], belt_tick: int | None = None) -> torch.Tensor:
    p = torch.tensor([prefix_ids], device=device)
    kv = model.prefill_prefix(p)
    belt = model.init_empty_belt(1, tick=p.size(1) if belt_tick is None else belt_tick, device=device, deterministic=True)
    out = model.forward_belt(belt, kv_cache=kv)
    return F.softmax(out.slot0_logits.float(), dim=-1)[0]


def top(dist: torch.Tensor, k: int = 6) -> str:
    pr, ix = dist.topk(k)
    return ", ".join(f"{tok.id_to_piece(int(i))!r}:{float(p):.2f}" for p, i in zip(pr, ix))


print("\n=== 3. Prefix sensitivity of Slot-0 (cold belt) ===")
prompts = {
    "lion (user)": prompt,
    "lion (no trailing space)": prompt.rstrip(),
    "real-style lion": "Once upon a time, there was a big lion named Leo. Leo liked to",
    "template": "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Together, they carefully",
}
dists = {}
for name, text in prompts.items():
    d = slot0_dist(tok.encode(text))
    dists[name] = d
    print(f"{name:28s} -> {top(d)}")

print("\n=== 4. Does Slot 0 depend on the prefix at all? (same length, shuffled prefix) ===")
base_ids = tok.encode(prompts["real-style lion"])
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(base_ids) - 1, generator=g) + 1
shuffled = [base_ids[0]] + [base_ids[i] for i in perm.tolist()]
d_true = slot0_dist(base_ids)
d_shuf = slot0_dist(shuffled)
d_empty = slot0_dist([tok.bos_id], belt_tick=len(base_ids))
kl = lambda a, b: float((a * (a.clamp_min(1e-9).log() - b.clamp_min(1e-9).log())).sum())
print(f"true    -> {top(d_true)}")
print(f"shuffled-> {top(d_shuf)}")
print(f"BOS-only-> {top(d_empty)}")
print(f"KL(true || shuffled) = {kl(d_true, d_shuf):.3f}   KL(true || BOS-only) = {kl(d_true, d_empty):.3f}")

print("\n=== 5. Teacher-forced next-token accuracy on REAL vs TEMPLATE val stories (Slot 0, cold belt) ===")
for kind in ["real", "template"]:
    correct = total = 0
    for i in range(len(val_ds)):
        seq = val_ds[i]["input_ids"].tolist()
        text = tok.decode(seq)
        is_t = "Together, they" in text and " walked into the " in text
        if (kind == "template") != is_t:
            continue
        L = next((j for j, t in enumerate(seq) if t == tok.pad_id), len(seq))
        for t in range(10, min(L - 1, 60), 7):
            d = slot0_dist(seq[:t])
            correct += int(int(d.argmax()) == seq[t])
            total += 1
        if total > 300:
            break
    print(f"{kind:8s}: slot0 acc = {correct/max(1,total):.3f} over {total} positions")
