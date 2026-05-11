"""Dataset loaders + Dirichlet / IID partitioning for FL simulation.

Supported datasets:
  * ``synthetic`` — synthetic text-token classification, offline-safe default.
  * ``sent140``   — Sentiment140 sentiment classification. Falls back to a
                    HuggingFace mirror when an offline cache is missing; if
                    that fails too, we generate a Sentiment140-shaped synthetic
                    corpus so the pipeline still runs end-to-end.
  * ``shakespeare`` — character-level next-character prediction.
  * ``agnews``   — 4-way news topic classification.

Every loader returns a list of (train_dataset, test_dataset) tuples — one per
client — plus a single global test set used for centralized evaluation.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


@dataclass
class TokenizedSample:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    label: int


class TextClassificationDataset(Dataset):
    """In-memory tokenized text classification dataset."""

    def __init__(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        assert input_ids.shape == attention_mask.shape
        assert input_ids.shape[0] == labels.shape[0]
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Synthetic text classification
# ---------------------------------------------------------------------------


def _build_synthetic(
    *,
    num_classes: int,
    samples_per_class: int,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate label-correlated random token sequences.

    Per-class "signature" tokens are scattered throughout the sequence so the
    signal survives the random projection of a frozen untrained encoder. With
    a real pretrained backbone you can use a weaker signature; we err on the
    side of teachable data so ablation runs converge in a handful of rounds.
    """
    n = num_classes * samples_per_class
    half = max(1, seq_len // 2)
    input_ids = rng.integers(low=1, high=vocab_size, size=(n, seq_len), dtype=np.int64)
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    # Reserve `signature_len` non-overlapping token-IDs per class. We splatter
    # them across the first half of the sequence so a frozen random encoder
    # still has multiple chances to preserve the class-conditional statistic.
    signature_len = max(4, seq_len // 4)
    class_sigs = []
    used = set()
    for _ in range(num_classes):
        sig: list[int] = []
        while len(sig) < signature_len:
            t = int(rng.integers(low=1, high=vocab_size))
            if t not in used:
                sig.append(t)
                used.add(t)
        class_sigs.append(np.asarray(sig, dtype=np.int64))

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        positions = rng.choice(half, size=signature_len, replace=False)
        for s_i, pos in enumerate(positions):
            input_ids[idx, pos] = class_sigs[c][s_i]

    perm = rng.permutation(n)
    input_ids, labels = input_ids[perm], labels[perm]
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Tokenizer abstraction
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Trivial whitespace+char tokenizer for offline runs."""

    def __init__(self, vocab_size: int = 256, max_len: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        # Reserve 0 = PAD, 1 = UNK
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._alphabet = string.printable
        self._lookup = {c: (i + 2) % vocab_size for i, c in enumerate(self._alphabet)}

    def encode(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        ids = [self._lookup.get(ch, self.unk_token_id) for ch in text[: self.max_len]]
        if len(ids) < self.max_len:
            ids = ids + [self.pad_token_id] * (self.max_len - len(ids))
        ids = np.asarray(ids, dtype=np.int64)
        mask = (ids != self.pad_token_id).astype(np.int64)
        return ids, mask

    def encode_batch(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        ids, masks = [], []
        for t in texts:
            i, m = self.encode(t)
            ids.append(i)
            masks.append(m)
        return np.stack(ids), np.stack(masks)


def _make_hf_tokenizer(backbone: str, max_len: int):
    """Try to load a HF tokenizer; return None on failure."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        log.warning("transformers unavailable: %s", exc)
        return None
    try:
        tok = AutoTokenizer.from_pretrained(backbone)
        tok.model_max_length = max_len
        return tok
    except Exception as exc:  # pragma: no cover
        log.warning("Could not load HF tokenizer for %s: %s", backbone, exc)
        return None


def _tokenize_with_hf(tokenizer, texts: Sequence[str], max_len: int):
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# Sent140 / AG News / Shakespeare loaders (best-effort)
# ---------------------------------------------------------------------------

_NEG_VOCAB = [
    "awful", "terrible", "hate", "boring", "worst", "broken", "disappointed",
    "horrible", "annoyed", "regret", "ugh", "bad", "useless", "sad", "angry",
    "fail", "trash", "lame", "miserable", "pain",
]
_POS_VOCAB = [
    "great", "love", "amazing", "fantastic", "happy", "wonderful", "perfect",
    "excellent", "joy", "thrilled", "awesome", "best", "delighted", "smile",
    "yay", "lol", "haha", "good", "favorite", "win",
]


_POS_TEMPLATES = [
    "I {pos} this movie! It was absolutely {pos}.",
    "This is the {pos} thing I have ever seen. So {pos}.",
    "Such a {pos} day today, I feel {pos} and {pos}.",
    "Honestly, {pos}. A {pos} experience all around.",
    "What a {pos} surprise — completely {pos}.",
    "Truly {pos}. I'd recommend it to anyone, {pos} stuff.",
    "Feeling {pos} after watching, totally {pos}.",
]
_NEG_TEMPLATES = [
    "I {neg} this movie. It was absolutely {neg}.",
    "This is the {neg} thing I have ever seen. So {neg}.",
    "Such a {neg} day today, I feel {neg} and {neg}.",
    "Honestly, {neg}. A {neg} experience all around.",
    "What a {neg} surprise — completely {neg}.",
    "Truly {neg}. I'd warn anyone away from it, {neg} stuff.",
    "Feeling {neg} after watching, totally {neg}.",
]


def _synthesize_sent140(
    *, num_samples: int, rng: np.random.Generator
) -> Tuple[List[str], np.ndarray]:
    """Generate sentiment-labelled sentences that are at least mildly
    recognisable to a pretrained BERT. We pick from a small bank of templates
    and fill them in with `_POS_VOCAB` / `_NEG_VOCAB` words."""
    texts, labels = [], []
    for _ in range(num_samples):
        lbl = int(rng.integers(0, 2))
        if lbl == 1:
            tpl = _POS_TEMPLATES[int(rng.integers(0, len(_POS_TEMPLATES)))]
            text = tpl.format(
                pos=str(rng.choice(_POS_VOCAB)),
            )
            # Replace remaining {pos} occurrences with fresh draws
            while "{pos}" in text:
                text = text.replace("{pos}", str(rng.choice(_POS_VOCAB)), 1)
        else:
            tpl = _NEG_TEMPLATES[int(rng.integers(0, len(_NEG_TEMPLATES)))]
            text = tpl.format(neg=str(rng.choice(_NEG_VOCAB)))
            while "{neg}" in text:
                text = text.replace("{neg}", str(rng.choice(_NEG_VOCAB)), 1)
        texts.append(text)
        labels.append(lbl)
    return texts, np.asarray(labels, dtype=np.int64)


def _try_load_sent140_hf(num_samples: int, rng: np.random.Generator):
    """Attempt to pull Sent140 from HuggingFace `datasets`. Return None on failure."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    candidates = [
        ("sentiment140",),
        ("stanfordnlp/sentiment140",),
    ]
    for args in candidates:
        try:
            ds = load_dataset(*args, split=f"train[:{num_samples * 2}]")
            texts = list(ds["text"])
            raw_labels = np.asarray(ds["sentiment"], dtype=np.int64)
            # Sent140 uses {0, 4}; coerce to {0, 1}.
            labels = (raw_labels > 0).astype(np.int64)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            keep = min(num_samples // 2, len(pos_idx), len(neg_idx))
            if keep == 0:
                return None
            idx = np.concatenate([pos_idx[:keep], neg_idx[:keep]])
            rng.shuffle(idx)
            return [texts[i] for i in idx], labels[idx]
        except Exception as exc:
            log.warning("Sent140 HF load failed for %s: %s", args, exc)
    return None


def _try_load_agnews_hf(num_samples: int, rng: np.random.Generator):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("ag_news", split=f"train[:{num_samples}]")
        texts = list(ds["text"])
        labels = np.asarray(ds["label"], dtype=np.int64)
        return texts, labels
    except Exception as exc:
        log.warning("AG News load failed: %s", exc)
        return None


def _synthesize_shakespeare(
    *, num_samples: int, seq_len: int, rng: np.random.Generator
) -> Tuple[List[str], np.ndarray]:
    """Bigram-Markov synthetic Shakespeare: next char depends on previous char.

    LEAF's real Shakespeare benchmark gives each client one role's lines, and
    next-character prediction is signal-bearing because of strong bigram
    statistics. Our fallback Markov chain reproduces that statistic so the
    pipeline can learn end-to-end without downloading LEAF.
    """
    alphabet = string.ascii_lowercase + " "
    n_letters = len(alphabet)
    # Build a sticky transition matrix: each letter prefers a small subset of
    # successors. This is what the model can actually pick up.
    trans = rng.dirichlet([0.1] * n_letters, size=n_letters)

    def _label(ch: str) -> int:
        return 26 if ch == " " else (ord(ch) - ord("a"))

    texts, labels = [], []
    for _ in range(num_samples):
        # Random starting char.
        cur = int(rng.integers(0, n_letters))
        history = [alphabet[cur]]
        for _ in range(seq_len - 1):
            cur = int(rng.choice(n_letters, p=trans[cur]))
            history.append(alphabet[cur])
        nxt = int(rng.choice(n_letters, p=trans[cur]))
        texts.append("".join(history))
        labels.append(_label(alphabet[nxt]))
    return texts, np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def iid_partition(
    num_samples: int, num_clients: int, rng: np.random.Generator
) -> List[np.ndarray]:
    idx = rng.permutation(num_samples)
    return [a for a in np.array_split(idx, num_clients)]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_size: int = 4,
) -> List[np.ndarray]:
    """Dirichlet label-skew partitioning (Yurochkin et al. 2019 / Hsu et al. 2019)."""
    num_classes = int(labels.max()) + 1
    n = labels.shape[0]
    while True:
        idx_per_client: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            cls_idx = np.where(labels == c)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)
            # ensure no client is empty by clipping early splits
            proportions = np.array(
                [
                    p * (len(client) < n / num_clients)
                    for p, client in zip(proportions, idx_per_client)
                ]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, split_points)
            for i, s in enumerate(splits):
                idx_per_client[i].extend(s.tolist())
        sizes = [len(c) for c in idx_per_client]
        if min(sizes) >= min_size:
            break
    return [np.asarray(c, dtype=np.int64) for c in idx_per_client]


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------


@dataclass
class FederatedData:
    client_train: List[TextClassificationDataset]
    client_test: List[TextClassificationDataset]
    global_test: TextClassificationDataset
    num_classes: int
    vocab_size: int


def _split_train_test(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, ...]:
    n = labels.shape[0]
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        input_ids[train_idx],
        attention_mask[train_idx],
        labels[train_idx],
        input_ids[test_idx],
        attention_mask[test_idx],
        labels[test_idx],
    )


def _make_subset_dataset(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> TextClassificationDataset:
    return TextClassificationDataset(
        input_ids[indices],
        attention_mask[indices],
        labels[indices],
    )


def load_federated_data(
    cfg: DataConfig,
    *,
    num_clients: int,
    backbone: str,
    seed: int,
) -> FederatedData:
    rng = np.random.default_rng(seed)
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.dataset.lower()
    use_hf_tokenizer = backbone not in {"tiny_fallback", "char"}

    # ---------------- 1. raw text/label arrays ----------------
    total = num_clients * cfg.num_samples_per_client
    texts: Optional[List[str]] = None
    labels: Optional[np.ndarray] = None
    num_classes = cfg.num_classes

    if dataset_name == "synthetic":
        ids, mask, labels = _build_synthetic(
            num_classes=num_classes,
            samples_per_class=max(8, total // max(num_classes, 1)),
            vocab_size=4096,
            seq_len=cfg.max_seq_len,
            rng=rng,
        )
        vocab_size = 4096

    else:
        if dataset_name == "sent140":
            res = _try_load_sent140_hf(total, rng)
            if res is None:
                log.warning("Falling back to synthesized Sent140-like corpus.")
                texts, labels = _synthesize_sent140(num_samples=total, rng=rng)
            else:
                texts, labels = res
            num_classes = 2
        elif dataset_name == "agnews":
            res = _try_load_agnews_hf(total, rng)
            if res is None:
                log.warning("Falling back to synthesized AG-News-like corpus.")
                texts, labels = _synthesize_sent140(num_samples=total, rng=rng)
                num_classes = 2
            else:
                texts, labels = res
                num_classes = 4
        elif dataset_name == "shakespeare":
            texts, labels = _synthesize_shakespeare(
                num_samples=total, seq_len=cfg.max_seq_len, rng=rng
            )
            num_classes = 27
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

        if use_hf_tokenizer:
            tok = _make_hf_tokenizer(backbone, cfg.max_seq_len)
            if tok is None:
                log.warning("HF tokenizer unavailable; using char tokenizer.")
                tok = CharTokenizer(vocab_size=256, max_len=cfg.max_seq_len)
                ids, mask = tok.encode_batch(texts)
                vocab_size = tok.vocab_size
            else:
                ids, mask = _tokenize_with_hf(tok, texts, cfg.max_seq_len)
                vocab_size = tok.vocab_size
        else:
            tok = CharTokenizer(vocab_size=256, max_len=cfg.max_seq_len)
            ids, mask = tok.encode_batch(texts)
            vocab_size = tok.vocab_size

    assert labels is not None

    # ---------------- 2. partition across clients ----------------
    if cfg.partition == "iid":
        client_idx = iid_partition(len(labels), num_clients, rng)
    elif cfg.partition == "dirichlet":
        client_idx = dirichlet_partition(
            labels, num_clients, cfg.dirichlet_alpha, rng
        )
    else:
        raise ValueError(f"Unknown partition: {cfg.partition!r}")

    # ---------------- 3. per-client train/test + global test ----------------
    client_train, client_test = [], []
    global_test_ids, global_test_mask, global_test_labels = [], [], []
    for idx in client_idx:
        tr_ids, tr_mask, tr_y, te_ids, te_mask, te_y = _split_train_test(
            ids[idx], mask[idx], labels[idx], cfg.test_size, rng
        )
        client_train.append(TextClassificationDataset(tr_ids, tr_mask, tr_y))
        client_test.append(TextClassificationDataset(te_ids, te_mask, te_y))
        global_test_ids.append(te_ids)
        global_test_mask.append(te_mask)
        global_test_labels.append(te_y)

    global_test = TextClassificationDataset(
        np.concatenate(global_test_ids, axis=0),
        np.concatenate(global_test_mask, axis=0),
        np.concatenate(global_test_labels, axis=0),
    )

    log.info(
        "Loaded %s with %d clients (partition=%s, α=%s). Sizes: %s",
        dataset_name,
        num_clients,
        cfg.partition,
        cfg.dirichlet_alpha if cfg.partition == "dirichlet" else "n/a",
        [len(d) for d in client_train],
    )

    return FederatedData(
        client_train=client_train,
        client_test=client_test,
        global_test=global_test,
        num_classes=int(num_classes),
        vocab_size=int(vocab_size),
    )
