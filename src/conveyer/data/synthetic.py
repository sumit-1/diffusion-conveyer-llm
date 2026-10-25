from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

from conveyer.config import ConveyerConfig


class SyntheticTokenizer:
    """
    Compact deterministic symbolic + word-piece vocabulary (128 tokens) for synthetic lookahead planning tasks.
    """

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3  # '=>' separator between prompt and output

        special = ["<PAD>", "<BOS>", "<EOS>", "=>", "|", ":", ",", ";", "(", ")", "[", "]", "->", "+", "=", "?"]
        task_tags = ["[PATH]", "[REV]", "[PLAN]", "[STORY]", "START", "GOAL", "EDGES", "SEQ", "SUM", "END", "STEP", "OK"]
        nodes = [f"N{i}" for i in range(32)]
        numbers = [f"{i}" for i in range(32)]
        words = [
            "the", "little", "robot", "walked", "across", "bridge", "to", "find", "golden", "key",
            "locked", "door", "opened", "quietly", "inside", "shiny", "crystal", "glowed", "bright", "blue",
            "cat", "chased", "mouse", "under", "wooden", "table", "until", "both", "fell", "asleep",
            "wise", "owl", "watched", "stars", "shine", "above", "dark", "forest",
        ]
        all_pieces = special + task_tags + nodes + numbers + words
        while len(all_pieces) < vocab_size:
            all_pieces.append(f"TOK_{len(all_pieces)}")
        self.pieces = all_pieces[:vocab_size]
        self.piece_to_id = {p: i for i, p in enumerate(self.pieces)}

    def id_to_piece(self, token_id: int) -> str:
        if 0 <= token_id < len(self.pieces):
            return self.pieces[token_id]
        return f"<{token_id}>"

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = [self.bos_id]
        for raw in text.strip().split():
            if raw in self.piece_to_id:
                tokens.append(self.piece_to_id[raw])
            else:
                # Fallback hash into word range
                idx = 64 + (hash(raw) % max(1, self.vocab_size - 64))
                tokens.append(idx)
        return tokens

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            out.append(self.id_to_piece(idx))
        return " ".join(out)


def _make_path_planning_sample(
    rng: np.random.Generator,
    tok: SyntheticTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Task 0 (Lookahead Graph Pathfinding - `[PATH]`):
    A fork graph where START branches into two paths of length 6:
      - One branch leads to a DEAD_END.
      - The other branch leads to the GOAL node.
    The edges are listed in random order, and the target trajectory is:
      `=> N_start N_step1 N_step2 N_step3 N_step4 N_step5 N_goal <EOS> ...`
    Crucially, we also require the model to emit the GOAL-conditioned path in reverse-verified
    or lookahead form where choosing `N_step1` at Position 0 requires chaining backwards from `GOAL`
    across the 6-hop chain. On the 64-slot conveyor belt, `N_goal` at the end of the path is anchored
    while `N_step1 .. N_step5` crystallize simultaneously!
    """
    node_pool = rng.choice(32, size=12, replace=False)
    start_node = int(node_pool[0])
    good_chain = [int(x) for x in node_pool[1:6]]  # 5 intermediate hops
    goal_node = int(node_pool[6])
    bad_chain = [int(x) for x in node_pool[7:12]]  # 5 dead-end hops

    edges: list[tuple[int, int]] = []
    # Branch A (reaches goal_node)
    edges.append((start_node, good_chain[0]))
    for i in range(len(good_chain) - 1):
        edges.append((good_chain[i], good_chain[i + 1]))
    edges.append((good_chain[-1], goal_node))

    # Branch B (dead-end branch from start_node)
    edges.append((start_node, bad_chain[0]))
    for i in range(len(bad_chain) - 1):
        edges.append((bad_chain[i], bad_chain[i + 1]))

    # Shuffle edge order so local greedy heuristics fail unless the model traces backward from GOAL
    perm = rng.permutation(len(edges))
    shuffled_edges = [edges[i] for i in perm]

    prompt_pieces = ["<BOS>", "[PATH]", "START", f"N{start_node}", "GOAL", f"N{goal_node}", "EDGES"]
    for u, v in shuffled_edges:
        prompt_pieces.extend([f"N{u}", "->", f"N{v}", ";"])
    prompt_pieces.append("=>")

    full_path = [start_node] + good_chain + [goal_node]
    # Emit the path twice separated by '|' and ending with GOAL confirmation so the full 64-slot belt is active
    target_pieces: list[str] = []
    for n in full_path:
        target_pieces.append(f"N{n}")
    target_pieces.append("|")
    for n in reversed(full_path):
        target_pieces.append(f"N{n}")
    target_pieces.extend(["OK", f"N{goal_node}", "<EOS>"])

    prompt_ids = [tok.piece_to_id[p] for p in prompt_pieces]
    target_ids = [tok.piece_to_id[p] for p in target_pieces]
    full_ids = prompt_ids + target_ids

    out = torch.full((seq_len,), tok.pad_id, dtype=torch.long)
    L = min(seq_len, len(full_ids))
    out[:L] = torch.tensor(full_ids[:L], dtype=torch.long)
    return out, len(prompt_ids), 0


def _make_reverse_plan_sample(
    rng: np.random.Generator,
    tok: SyntheticTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Task 1 (Delayed-Constraint / Reverse Block Planning - `[REV]`):
    The prompt gives a sequence of 16 numbers `a_0 .. a_15`.
    The output must emit pairs `(a_i + a_{15-i}) mod 32` followed by the reversed sequence `a_15 .. a_0`.
    """
    length = 12
    nums = [int(x) for x in rng.integers(0, 16, size=length)]
    prompt_pieces = ["<BOS>", "[REV]", "SEQ"] + [f"{x}" for x in nums] + ["=>"]

    sums = [(nums[i] + nums[length - 1 - i]) % 32 for i in range(length)]
    rev = list(reversed(nums))
    target_pieces = (
        ["SUM"]
        + [f"{s}" for s in sums]
        + ["|"]
        + [f"{r}" for r in rev]
        + ["OK", "<EOS>"]
    )

    prompt_ids = [tok.piece_to_id[p] for p in prompt_pieces]
    target_ids = [tok.piece_to_id[p] for p in target_pieces]
    full_ids = prompt_ids + target_ids

    out = torch.full((seq_len,), tok.pad_id, dtype=torch.long)
    L = min(seq_len, len(full_ids))
    out[:L] = torch.tensor(full_ids[:L], dtype=torch.long)
    return out, len(prompt_ids), 1


def _make_target_anchored_chain_sample(
    rng: np.random.Generator,
    tok: SyntheticTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Task 2 (Target-Anchored Arithmetic / Story Planning - `[PLAN]`):
    Prompt specifies a target final value `END = T` and a sequence of deltas `d_1 .. d_10`.
    The model must choose the unique initial value `x_0` at Position 0 such that after applying
    `x_{k} = (x_{k-1} + d_k) mod 32` for 10 steps, the final token at Position 10 equals `T`!
    Then it emits the full trajectory `x_0, x_1, ..., x_10 = T`.
    """
    steps = 10
    deltas = [int(x) for x in rng.integers(1, 8, size=steps)]
    target_end = int(rng.integers(0, 32))
    total_delta = sum(deltas) % 32
    x0 = (target_end - total_delta) % 32

    traj = [x0]
    for d in deltas:
        traj.append((traj[-1] + d) % 32)
    assert traj[-1] == target_end

    prompt_pieces = (
        ["<BOS>", "[PLAN]", "END", f"{target_end}", "STEP"]
        + [f"+ {d}" for d in deltas]
        # flatten "+ d" into two tokens
    )
    flat_prompt: list[str] = ["<BOS>", "[PLAN]", "END", f"{target_end}", "STEP"]
    for d in deltas:
        flat_prompt.extend(["+", f"{d}"])
    flat_prompt.append("=>")

    target_pieces = [f"{val}" for val in traj] + ["=", f"{target_end}", "OK", "<EOS>"]

    prompt_ids = [tok.piece_to_id[p] for p in flat_prompt]
    target_ids = [tok.piece_to_id[p] for p in target_pieces]
    full_ids = prompt_ids + target_ids

    out = torch.full((seq_len,), tok.pad_id, dtype=torch.long)
    L = min(seq_len, len(full_ids))
    out[:L] = torch.tensor(full_ids[:L], dtype=torch.long)
    return out, len(prompt_ids), 2


class SyntheticLookaheadDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        task: str = "all",
        vocab_size: int = 128,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.task = task
        self.tokenizer = SyntheticTokenizer(vocab_size=vocab_size)

        rng = np.random.default_rng(seed)
        self.samples: list[dict[str, Any]] = []

        for i in range(num_samples):
            if task == "path":
                t_choice = 0
            elif task == "reverse":
                t_choice = 1
            elif task == "constraint":
                t_choice = 2
            else:
                t_choice = i % 3

            if t_choice == 0:
                seq, plen, tid = _make_path_planning_sample(rng, self.tokenizer, seq_len)
            elif t_choice == 1:
                seq, plen, tid = _make_reverse_plan_sample(rng, self.tokenizer, seq_len)
            else:
                seq, plen, tid = _make_target_anchored_chain_sample(rng, self.tokenizer, seq_len)

            self.samples.append(
                {
                    "input_ids": seq,
                    "prompt_len": plen,
                    "task_id": tid,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def build_synthetic_datasets(
    cfg: ConveyerConfig,
) -> tuple[SyntheticLookaheadDataset, SyntheticLookaheadDataset, SyntheticTokenizer]:
    train_ds = SyntheticLookaheadDataset(
        num_samples=cfg.data.num_train,
        seq_len=cfg.data.seq_len,
        task=cfg.data.task,
        vocab_size=cfg.data.vocab_size,
        seed=cfg.train.seed,
    )
    val_ds = SyntheticLookaheadDataset(
        num_samples=cfg.data.num_val,
        seq_len=cfg.data.seq_len,
        task=cfg.data.task,
        vocab_size=cfg.data.vocab_size,
        seed=cfg.train.seed + 10000,
    )
    return train_ds, val_ds, train_ds.tokenizer
