"""
train_utils.py
---------------
Shared helpers used by every experiment script: reproducible seeding,
a generic evaluation pass (perplexity/val-loss), and a wall-clock timer.
Kept separate from model.py / data.py so each experiment script stays
short and readable.
"""

import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from .data import PAD


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lm_batch_loss(logits, input_ids):
    """Next-token prediction loss with left-shift; PAD ignored."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_targets.reshape(-1),
        ignore_index=PAD, reduction="mean",
    )
    return loss, shift_targets


@torch.no_grad()
def evaluate(model, dataset, batch_size=32, device="cpu"):
    """Token-weighted cross-entropy over the full validation split.

    Weighting by token count rather than averaging per-batch means matters
    because the last batch is usually short and sequences vary in length; a
    mean of batch means silently over-weights those examples.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    val = dataset.val
    for i in range(0, len(val), batch_size):
        batch = val[i:i + batch_size]
        input_ids, attn_mask = dataset.collate(batch)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        logits = model(input_ids, attn_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        B, T, V = shift_logits.shape
        loss_sum = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_targets.reshape(-1),
            ignore_index=PAD, reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += int((shift_targets != PAD).sum().item())
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    ppl = float(np.exp(min(mean_loss, 20)))
    return mean_loss, ppl


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
