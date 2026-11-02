from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Optional
import urllib.request
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset

from conveyer.config import ConveyerConfig

TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
TINYSTORIES_TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
)


class TiktokenCompactTokenizer:
    """
    Backed by OpenAI `tiktoken` (`gpt2` BPE encoding), mapped into a compact vocabulary
    of size `vocab_size` (e.g. 4096) covering the most frequent real-TinyStories BPE subwords
    plus all 256 single-byte BPE tokens so ANY arbitrary English string encodes with ZERO `<UNK>`.
    """

    def __init__(
        self,
        gpt2_to_compact: dict[int, int],
        compact_to_gpt2: list[int],
        source: str = "real_tinystories_v2",
    ) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.gpt2_to_compact = gpt2_to_compact
        self.compact_to_gpt2 = compact_to_gpt2
        self.source = source
        self.vocab_size = len(compact_to_gpt2)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        # Prebuild byte-level fallback table so any rare BPE token decomposes into exact byte BPE tokens
        self.byte_to_compact: dict[int, int] = {}
        for b in range(256):
            gid = self.enc.encode_single_token(bytes([b]))
            if gid in self.gpt2_to_compact:
                self.byte_to_compact[b] = self.gpt2_to_compact[gid]

        # Prebuild fast array lookup for GPT-2 token IDs (0..50256) -> compact ID (-1 if rare)
        self._lut = np.full(self.enc.n_vocab, -1, dtype=np.int32)
        for gid, cid in self.gpt2_to_compact.items():
            if 0 <= gid < self.enc.n_vocab:
                self._lut[gid] = cid

        # Prebuild boolean mask of English content words (nouns, verbs, adjectives, names) for the Thought Reservoir
        stopwords = {
            "the", "and", "a", "to", "was", "in", "on", "of", "for", "with", "that", "is", "at", "be", "are",
            "as", "by", "an", "or", "from", "it", "this", "were", "had", "have", "has", "did", "so", "up", "out",
            "he", "she", "his", "her", "him", "they", "their", "them", "you", "your", "we", "our", "its", "but",
            "not", "very", "too", "all", "one", "there", "when", "then", "what", "who", "how", "can", "will",
            "would", "could", "should", "about", "into", "over", "after", "before", "once", "upon", "time",
        }
        self.content_vocab_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for cid in range(260, self.vocab_size):
            piece = self.id_to_piece(cid).strip().lower()
            if len(piece) >= 2 and piece.isalpha() and piece not in stopwords:
                self.content_vocab_mask[cid] = True

    @classmethod
    def build_from_texts(cls, texts: list[str], vocab_size: int = 4096) -> "TiktokenCompactTokenizer":
        enc = tiktoken.get_encoding("gpt2")
        # Special tokens occupy 0..3: <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3
        compact_to_gpt2: list[int] = [-1, -2, -3, -4]
        seen_gpt2: set[int] = set()

        # 1. Always include all 256 single-byte BPE tokens so 100% of UTF-8 strings are representable
        for b in range(256):
            gid = enc.encode_single_token(bytes([b]))
            if gid not in seen_gpt2 and len(compact_to_gpt2) < vocab_size:
                seen_gpt2.add(gid)
                compact_to_gpt2.append(gid)

        # 2. Count BPE token frequencies across real TinyStories
        counts: Counter[int] = Counter()
        sample = texts[:50000]
        for batch_list in enc.encode_ordinary_batch(sample, num_threads=8):
            counts.update(batch_list)

        for gid, _ in counts.most_common():
            if len(compact_to_gpt2) >= vocab_size:
                break
            if gid not in seen_gpt2:
                seen_gpt2.add(gid)
                compact_to_gpt2.append(gid)

        # Pad up to vocab_size with sequential gpt2 token IDs if needed
        next_gid = 0
        while len(compact_to_gpt2) < vocab_size:
            if next_gid not in seen_gpt2:
                seen_gpt2.add(next_gid)
                compact_to_gpt2.append(next_gid)
            next_gid += 1

        gpt2_to_compact = {gid: cid for cid, gid in enumerate(compact_to_gpt2) if gid >= 0}
        return cls(
            gpt2_to_compact=gpt2_to_compact,
            compact_to_gpt2=compact_to_gpt2,
            source="real_tinystories_v2",
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "compact_to_gpt2": self.compact_to_gpt2,
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TiktokenCompactTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        compact_to_gpt2: list[int] = data["compact_to_gpt2"]
        source: str = data.get("source", "legacy")
        gpt2_to_compact = {gid: cid for cid, gid in enumerate(compact_to_gpt2) if gid >= 0}
        return cls(gpt2_to_compact=gpt2_to_compact, compact_to_gpt2=compact_to_gpt2, source=source)

    def id_to_piece(self, token_id: int) -> str:
        if token_id == self.pad_id:
            return "<PAD>"
        if token_id == self.bos_id:
            return "<BOS>"
        if token_id == self.eos_id:
            return "<EOS>"
        if token_id == self.unk_id or not (0 <= token_id < len(self.compact_to_gpt2)):
            return "<UNK>"
        gid = self.compact_to_gpt2[token_id]
        if gid < 0:
            return "<SPEC>"
        raw = self.enc.decode_single_token_bytes(gid).decode("utf-8", errors="replace")
        return raw

    def _map_gids_to_compact(self, gids: list[int], max_tokens: Optional[int] = None) -> list[int]:
        ids: list[int] = [self.bos_id]
        gpt2_map = self.gpt2_to_compact
        byte_map = self.byte_to_compact
        dec_single = self.enc.decode_single_token_bytes
        unk = self.unk_id
        for gid in gids:
            cid = gpt2_map.get(gid)
            if cid is not None:
                ids.append(cid)
            else:
                for b in dec_single(gid):
                    ids.append(byte_map.get(b, unk))
            if max_tokens is not None and len(ids) >= max_tokens:
                break
        return ids

    def encode(self, text: str) -> list[int]:
        gids = self.enc.encode_ordinary(text)
        return self._map_gids_to_compact(gids)

    def encode_batch_into_matrix(
        self,
        texts: list[str],
        out_matrix: np.ndarray,
        out_lens: np.ndarray,
        start_row: int,
        seq_len: int,
    ) -> int:
        """
        Multi-threaded Rust BPE batch encoding into `out_matrix[start_row : start_row + written]` (uint16).
        Truncates input text to ~1400 chars before BPE since we only need `seq_len` (e.g. 160) tokens.
        """
        char_cap = max(seq_len * 8, 1024)
        clipped = [t[:char_cap] if len(t) > char_cap else t for t in texts]
        batch_gids = self.enc.encode_ordinary_batch(clipped, num_threads=8)

        row = start_row
        max_rows = out_matrix.shape[0]
        eos = self.eos_id
        pad = self.pad_id
        max_body = seq_len - 1

        for gids in batch_gids:
            if row >= max_rows:
                break
            ids = self._map_gids_to_compact(gids, max_tokens=max_body)
            if len(ids) < 16:
                continue
            if len(ids) < seq_len:
                ids.append(eos)
            L = min(seq_len, len(ids))
            out_matrix[row, :L] = ids[:L]
            if L < seq_len:
                out_matrix[row, L:] = pad
            out_lens[row] = L
            row += 1
        return row - start_row

    def decode(self, ids: list[int]) -> str:
        gids: list[int] = []
        for cid in ids:
            if cid in (self.pad_id, self.bos_id, self.unk_id):
                continue
            if cid == self.eos_id:
                break
            if 0 <= cid < len(self.compact_to_gpt2):
                gid = self.compact_to_gpt2[cid]
                if gid >= 0:
                    gids.append(gid)
        return self.enc.decode(gids)


def _download_with_resume(url: str, dest: Path, min_expected_bytes: int = 1_000_000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing >= min_expected_bytes:
        return dest
    total_size = 0
    try:
        req_head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_head, timeout=15) as r:
            total_size = int(r.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    if total_size > 0 and existing >= total_size:
        return dest

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing > 0 and total_size > existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    else:
        existing = 0

    print(
        f"[TinyStories] Downloading {dest.name} "
        f"({existing / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB)...",
        flush=True,
    )
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_log = t0
    downloaded = existing
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
        while True:
            chunk = r.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 5.0:
                dt = now - t0
                speed = (downloaded - existing) / (1024 * 1024) / max(dt, 1e-3)
                pct = 100.0 * downloaded / max(total_size, 1)
                print(
                    f"  [{dest.name}] {downloaded / (1024 * 1024):.1f} / "
                    f"{total_size / (1024 * 1024):.1f} MB ({pct:.1f}%) at {speed:.1f} MB/s",
                    flush=True,
                )
                last_log = now
    return dest


def iter_clean_real_stories(txt_path: Path, max_stories: Optional[int] = None) -> Iterator[str]:
    """
    Streams clean real TinyStories from a `<|endoftext|>`-separated raw text file without
    loading the entire 2.12 GB string list into memory at once.
    Filters out byte-0 mid-sentence fragments and any synthetic templates.
    """
    yielded = 0
    buf = ""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split("<|endoftext|>")
            buf = parts[-1]
            for raw in parts[:-1]:
                s = " ".join(raw.strip().split())
                if len(s) < 80:
                    continue
                if not (s[0].isupper() or s[0] in "\"'"):
                    continue
                if "walked into the " in s and "Together, they" in s:
                    continue
                yield s
                yielded += 1
                if max_stories is not None and yielded >= max_stories:
                    return
    s = " ".join(buf.strip().split())
    if len(s) >= 80 and (s[0].isupper() or s[0] in "\"'"):
        if not ("walked into the " in s and "Together, they" in s):
            if max_stories is None or yielded < max_stories:
                yield s


def ensure_raw_tinystories_files(data_dir: str | Path, need_full_train: bool = True) -> tuple[Path, Path]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    val_path = _download_with_resume(TINYSTORIES_VALID_URL, root / "TinyStoriesV2-GPT4-valid.txt", min_expected_bytes=20_000_000)
    train_path = root / "TinyStoriesV2-GPT4-train.txt"
    if need_full_train or not train_path.exists():
        _download_with_resume(TINYSTORIES_TRAIN_URL, train_path, min_expected_bytes=2_000_000_000)
    return train_path, val_path


def load_or_prepare_tinystories(data_dir: str | Path, max_stories: int = 42000) -> list[str]:
    """
    Returns up to `max_stories` 100% REAL TinyStories strings (zero templates).
    """
    root = Path(data_dir)
    train_path, val_path = ensure_raw_tinystories_files(root, need_full_train=(max_stories > 25000))
    source_path = train_path if train_path.exists() and train_path.stat().st_size > 20_000_000 else val_path
    stories = list(iter_clean_real_stories(source_path, max_stories=max_stories))
    return stories


def _build_or_load_memmap_split(
    split_name: str,
    raw_txt_path: Path,
    tokenizer: TiktokenCompactTokenizer,
    data_dir: Path,
    target_stories: int,
    seq_len: int,
) -> tuple[np.memmap, np.ndarray]:
    """
    Pre-tokenizes `target_stories` real TinyStories from `raw_txt_path` into a compact disk-backed
    `uint16` numpy memmap of shape `(N, seq_len)` + `uint16` lengths array `(N,)`.
    Loading an existing memmap takes < 10ms and uses near-zero RAM.
    """
    prefix = data_dir / f"real_{split_name}_v5_n{target_stories}_s{seq_len}_v{tokenizer.vocab_size}"
    tokens_bin = Path(str(prefix) + ".tokens.bin")
    lens_bin = Path(str(prefix) + ".lens.bin")
    meta_json = Path(str(prefix) + ".meta.json")

    if tokens_bin.exists() and lens_bin.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        n_rows = int(meta["num_rows"])
        if n_rows > 0 and tokens_bin.stat().st_size == n_rows * seq_len * 2:
            mm = np.memmap(tokens_bin, dtype=np.uint16, mode="r", shape=(n_rows, seq_len))
            lens = np.fromfile(lens_bin, dtype=np.uint16)
            return mm, lens

    print(
        f"[TinyStories] Building binary token cache for '{split_name}' "
        f"(target={target_stories:,} stories, seq_len={seq_len})...",
        flush=True,
    )
    tmp_tokens = Path(str(tokens_bin) + ".tmp")
    tmp_lens = Path(str(lens_bin) + ".tmp")

    mm_write = np.memmap(tmp_tokens, dtype=np.uint16, mode="w+", shape=(target_stories, seq_len))
    lens_arr = np.zeros(target_stories, dtype=np.uint16)

    t0 = time.time()
    last_log = t0
    row = 0
    batch_texts: list[str] = []
    batch_chunk_size = 50000

    for story in iter_clean_real_stories(raw_txt_path, max_stories=int(target_stories * 1.05) + 2000):
        batch_texts.append(story)
        if len(batch_texts) >= batch_chunk_size:
            added = tokenizer.encode_batch_into_matrix(batch_texts, mm_write, lens_arr, row, seq_len)
            row += added
            batch_texts.clear()
            now = time.time()
            if now - last_log >= 4.0:
                dt = now - t0
                rate = row / max(dt, 1e-3)
                print(
                    f"  [{split_name}] Tokenized {row:,} / {target_stories:,} stories "
                    f"({100.0 * row / max(target_stories, 1):.1f}%) at {rate:,.0f} stories/s",
                    flush=True,
                )
                last_log = now
            if row >= target_stories:
                break

    if batch_texts and row < target_stories:
        added = tokenizer.encode_batch_into_matrix(batch_texts, mm_write, lens_arr, row, seq_len)
        row += added
        batch_texts.clear()

    mm_write.flush()
    del mm_write

    # If actual valid stories in file < target_stories, truncate file to exact `row * seq_len * 2` bytes
    if row < target_stories:
        with open(tmp_tokens, "r+b") as f:
            f.truncate(row * seq_len * 2)
        lens_arr = lens_arr[:row]

    lens_arr.tofile(tmp_lens)
    os.replace(tmp_tokens, tokens_bin)
    os.replace(tmp_lens, lens_bin)
    meta_json.write_text(
        json.dumps({"num_rows": int(row), "seq_len": int(seq_len), "vocab_size": int(tokenizer.vocab_size)}),
        encoding="utf-8",
    )
    dt = time.time() - t0
    print(f"[TinyStories] Saved '{split_name}' binary cache: {row:,} real stories in {dt:.1f}s", flush=True)

    mm = np.memmap(tokens_bin, dtype=np.uint16, mode="r", shape=(row, seq_len))
    return mm, lens_arr


class TinyStoriesDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        stories: Optional[list[str]] = None,
        tokenizer: Optional[TiktokenCompactTokenizer] = None,
        seq_len: int = 160,
        prompt_len: int = 28,
        randomize_prompt_len: bool = False,
        token_matrix: Optional[np.ndarray] = None,
        valid_lens: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.randomize_prompt_len = randomize_prompt_len

        if token_matrix is not None and valid_lens is not None:
            self.token_matrix = token_matrix
            self.valid_lens = valid_lens
        else:
            assert stories is not None and tokenizer is not None
            n_cap = len(stories)
            mat = np.full((n_cap, seq_len), tokenizer.pad_id, dtype=np.uint16)
            lens = np.zeros(n_cap, dtype=np.uint16)
            written = tokenizer.encode_batch_into_matrix(stories, mat, lens, 0, seq_len)
            self.token_matrix = mat[:written]
            self.valid_lens = lens[:written]

    def __len__(self) -> int:
        return int(self.token_matrix.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = torch.from_numpy(np.asarray(self.token_matrix[idx], dtype=np.int64))
        L = int(self.valid_lens[idx])
        if not self.randomize_prompt_len:
            p_len = max(4, min(self.prompt_len, L - 8))
        else:
            low = 6
            high = max(low + 1, min(64, L - 8))
            p_len = int(torch.randint(low, high, (1,)).item())
        return {
            "input_ids": row,
            "prompt_len": p_len,
            "task_id": 0,
        }


def build_tinystories_datasets(
    cfg: ConveyerConfig,
    infer_only: bool = False,
) -> tuple[TinyStoriesDataset, TinyStoriesDataset, TiktokenCompactTokenizer]:
    data_dir = Path(cfg.data.data_dir)
    repo_root = Path(__file__).resolve().parents[3]
    if (
        not data_dir.is_absolute()
        and not (data_dir / f"tiktoken_compact_{cfg.data.vocab_size}.json").exists()
        and (repo_root / data_dir / f"tiktoken_compact_{cfg.data.vocab_size}.json").exists()
    ):
        data_dir = repo_root / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / f"tiktoken_compact_{cfg.data.vocab_size}.json"
    val_sample_path = data_dir / "val_sample_256.txt"

    if infer_only and tok_path.exists():
        tokenizer = TiktokenCompactTokenizer.load(tok_path)
        cfg.data.vocab_size = tokenizer.vocab_size
        cfg.model.vocab_size = tokenizer.vocab_size
        if val_sample_path.exists():
            sample_lines = [l.strip() for l in val_sample_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            _, val_txt = ensure_raw_tinystories_files(data_dir, need_full_train=False)
            sample_lines = list(iter_clean_real_stories(val_txt, max_stories=256))
        val_ds = TinyStoriesDataset(
            stories=sample_lines,
            tokenizer=tokenizer,
            seq_len=cfg.data.seq_len,
            prompt_len=28,
            randomize_prompt_len=False,
        )
        return val_ds, val_ds, tokenizer

    train_txt, val_txt = ensure_raw_tinystories_files(data_dir, need_full_train=True)

    tokenizer_opt: Optional[TiktokenCompactTokenizer] = None
    if tok_path.exists():
        loaded = TiktokenCompactTokenizer.load(tok_path)
        if loaded.source == "real_tinystories_v2":
            tokenizer_opt = loaded

    if tokenizer_opt is None:
        print("[TinyStories] Building compact 4096 BPE vocabulary from 50,000 real stories...", flush=True)
        sample_stories = list(iter_clean_real_stories(train_txt, max_stories=50000))
        tokenizer_opt = TiktokenCompactTokenizer.build_from_texts(sample_stories, vocab_size=cfg.data.vocab_size)
        tokenizer_opt.save(tok_path)

    tokenizer = tokenizer_opt
    cfg.data.vocab_size = tokenizer.vocab_size
    cfg.model.vocab_size = tokenizer.vocab_size

    train_mm, train_lens = _build_or_load_memmap_split(
        split_name="train",
        raw_txt_path=train_txt,
        tokenizer=tokenizer,
        data_dir=data_dir,
        target_stories=cfg.data.num_train,
        seq_len=cfg.data.seq_len,
    )
    val_mm, val_lens = _build_or_load_memmap_split(
        split_name="val",
        raw_txt_path=val_txt,
        tokenizer=tokenizer,
        data_dir=data_dir,
        target_stories=cfg.data.num_val,
        seq_len=cfg.data.seq_len,
    )

    train_ds = TinyStoriesDataset(
        tokenizer=tokenizer,
        seq_len=cfg.data.seq_len,
        randomize_prompt_len=True,
        token_matrix=train_mm,
        valid_lens=train_lens,
    )
    val_ds = TinyStoriesDataset(
        tokenizer=tokenizer,
        seq_len=cfg.data.seq_len,
        prompt_len=28,
        randomize_prompt_len=False,
        token_matrix=val_mm,
        valid_lens=val_lens,
    )
    return train_ds, val_ds, tokenizer
