"""
model.py
--------
A minimal causal Transformer language model, small enough to train on CPU
in seconds, plus a from-scratch LoRA (Hu et al., 2021) linear layer used
for the parameter-efficient adaptation arm of CG-PED.

No HuggingFace / PEFT dependency: LoRA is implemented directly (~20 lines)
so the whole pipeline is self-contained and auditable.

The adapters are wrapped onto the module that actually owns each projection,
and `assert_lora_live` checks by forward/backward pass that every adapter
receives gradient. Both exist because an earlier version of this file aliased
the attention projections onto the enclosing Block; the wrapper was installed
on the alias while the forward pass called the original, so the query, value
and output adapters silently received no gradient for the whole of training.
Nothing in the loss curves or the trainable-parameter count revealed it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# LoRA linear layer
# ----------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    Wraps a frozen base nn.Linear with a trainable low-rank update:
        y = W0 x + b0 + (alpha / r) * B (A x)
    W0, b0 are frozen; only A, B are trainable.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # start at identity (no-op) update
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.drop(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def apply_lora(model: nn.Module, r: int = 4, alpha: int = 8, targets=LORA_TARGETS):
    """
    In-place: replace every target nn.Linear with a LoRALinear wrapper.

    Walks the module tree and rebinds the attribute on the layer's ACTUAL
    parent, so the wrapper sits on the path the forward pass takes. Rebinding
    an attribute on some other module that merely holds a reference to the
    same nn.Linear produces a wrapper that is never called and never receives
    gradient; `assert_lora_live` below is the regression test for exactly
    that failure mode.
    """
    to_wrap = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if name in targets and isinstance(child, nn.Linear):
                to_wrap.append((parent, name, child))
    for parent, name, child in to_wrap:
        setattr(parent, name, LoRALinear(child, r=r, alpha=alpha))
    return model


def lora_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def assert_lora_live(model: nn.Module, input_ids, attn_mask=None):
    """
    Verify that every LoRA adapter in `model` is actually on the forward path.

    Runs one forward/backward pass and checks that each adapter's B matrix
    receives a gradient. An adapter that is wrapped but never called reports
    `grad is None`, which is silent during training -- the run completes, the
    trainable-parameter count still looks right, and the arm quietly trains
    with frozen base weights and no adaptation at all.

    Returns (n_adapters, n_live). Raises AssertionError if any adapter is dead.
    """
    mods = lora_modules(model)
    assert mods, "assert_lora_live called on a model with no LoRA adapters"
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(input_ids, attn_mask)
    V = logits.shape[-1]
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V), input_ids[:, 1:].reshape(-1)
    )
    loss.backward()

    dead = [n for n, m in mods if m.lora_B.grad is None]
    n_live = len(mods) - len(dead)
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    assert not dead, (
        f"{len(dead)} of {len(mods)} LoRA adapters received no gradient and are "
        f"therefore not on the forward path: {dead}"
    )
    return len(mods), n_live


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------
# Tiny causal Transformer
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True = valid
            pad = (~key_padding_mask).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)
            att = att.masked_fill(pad, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)  # rows that are fully masked -> 0
        att = self.drop(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        x = x + self.drop(h)
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                 max_len=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, V) logits
