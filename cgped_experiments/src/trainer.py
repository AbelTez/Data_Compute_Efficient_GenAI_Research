"""
trainer.py
----------
A single, configurable training loop shared by every cell of the factorial,
so cells differ ONLY in the flags being tested (curriculum ordering, LoRA,
distillation) and nothing else -- same model size, same optimizer, same
number of gradient steps, same batch size, same seeds. This is what makes
the comparison meaningful.
"""

import random
from dataclasses import dataclass

import torch
import torch.optim as optim

from .data import curriculum_order, random_order, pacing_schedule
from .model import TinyTransformerLM, apply_lora, trainable_param_count, total_param_count
from .distillation import kd_loss
from .train_utils import lm_batch_loss, evaluate, set_seed, Timer


@dataclass
class RunConfig:
    name: str
    seed: int = 0
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_len: int = 64
    dropout: float = 0.1
    batch_size: int = 32
    n_steps: int = 300
    lr: float = 3e-3
    eval_every: int = 15
    use_curriculum: bool = False
    use_lora: bool = False
    lora_r: int = 4
    lora_alpha: int = 8
    use_distillation: bool = False
    kd_temperature: float = 2.0
    kd_alpha: float = 0.5
    teacher_d_model: int = 128
    teacher_n_layers: int = 4
    teacher_n_heads: int = 8
    teacher_d_ff: int = 256
    teacher_steps: int = 250
    thresholds: tuple = (3.6, 3.55, 3.5)


def build_model(cfg: RunConfig, vocab_size, d_model=None, n_layers=None,
                 n_heads=None, d_ff=None, max_len=None):
    return TinyTransformerLM(
        vocab_size,
        d_model=d_model or cfg.d_model,
        n_layers=n_layers or cfg.n_layers,
        n_heads=n_heads or cfg.n_heads,
        d_ff=d_ff or cfg.d_ff,
        max_len=max(max_len or 0, cfg.max_len),
        dropout=cfg.dropout,
    )


def _sample_batch(dataset, pool_indices, batch_size, rng):
    idx = [rng.choice(pool_indices) for _ in range(batch_size)]
    batch = [dataset.train[i] for i in idx]
    return dataset.collate(batch, max_len=dataset.cfg.max_len)


def train_teacher(dataset, cfg: RunConfig, device="cpu"):
    """Train a larger, plain (no curriculum/LoRA/KD) teacher model once per
    seed; shared by every distillation cell in that row."""
    set_seed(cfg.seed + 999)
    teacher = build_model(
        cfg, dataset.cfg.vocab_size,
        d_model=cfg.teacher_d_model, n_layers=cfg.teacher_n_layers,
        n_heads=cfg.teacher_n_heads, d_ff=cfg.teacher_d_ff,
        max_len=dataset.cfg.max_len,
    ).to(device)
    opt = optim.AdamW(teacher.parameters(), lr=cfg.lr)
    order = random_order(dataset.train, seed=cfg.seed)
    rng = random.Random(cfg.seed)
    for step in range(cfg.teacher_steps):
        input_ids, attn_mask = _sample_batch(dataset, order, cfg.batch_size, rng)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        logits = teacher(input_ids, attn_mask)
        loss, _ = lm_batch_loss(logits, input_ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    val_loss, val_ppl = evaluate(teacher, dataset, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, {"val_loss": val_loss, "val_ppl": val_ppl}


def run_training(dataset, cfg: RunConfig, teacher=None, device="cpu", return_model=False):
    """
    Runs one training configuration end-to-end and returns a dict with the
    loss history (for plotting) and summary metrics (for the results table).
    If return_model=True, also returns the trained nn.Module as a second
    return value (useful for downstream qualitative sampling without
    having to re-run training a second time).
    """
    set_seed(cfg.seed)
    model = build_model(cfg, dataset.cfg.vocab_size, max_len=dataset.cfg.max_len).to(device)

    if cfg.use_lora:
        apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
        model.to(device)

    trainable = trainable_param_count(model)
    total = total_param_count(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=cfg.lr)

    if cfg.use_curriculum:
        order = curriculum_order(dataset.train, seed=cfg.seed)
        pools = pacing_schedule(order, cfg.n_steps)
    else:
        order = random_order(dataset.train, seed=cfg.seed)
        pools = [order] * cfg.n_steps  # full pool available every step

    rng = random.Random(cfg.seed)
    history = []  # (step, train_loss, val_loss, val_ppl)
    steps_to_threshold = {}
    thresholds = list(cfg.thresholds)

    t0_timer = Timer()
    with t0_timer:
        for step in range(cfg.n_steps):
            pool = pools[step]
            input_ids, attn_mask = _sample_batch(dataset, pool, cfg.batch_size, rng)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

            if cfg.use_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids, attn_mask)
                student_logits = model(input_ids, attn_mask)
                shift_student = student_logits[:, :-1, :].contiguous()
                shift_teacher = teacher_logits[:, :-1, :].contiguous()
                shift_targets = input_ids[:, 1:].contiguous()
                loss, ce, kd = kd_loss(
                    shift_student, shift_teacher, shift_targets,
                    temperature=cfg.kd_temperature, alpha=cfg.kd_alpha,
                )
            else:
                logits = model(input_ids, attn_mask)
                loss, _ = lm_batch_loss(logits, input_ids)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.eval_every == 0 or step == cfg.n_steps - 1:
                val_loss, val_ppl = evaluate(model, dataset, device=device)
                history.append(
                    {"step": step, "train_loss": float(loss.item()),
                     "val_loss": val_loss, "val_ppl": val_ppl}
                )
                for th in thresholds:
                    if th not in steps_to_threshold and val_loss <= th:
                        steps_to_threshold[th] = step

    final_val_loss, final_val_ppl = evaluate(model, dataset, device=device)

    result = {
        "name": cfg.name,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_sec": round(t0_timer.elapsed, 4),
        "history": history,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "steps_to_threshold": steps_to_threshold,
    }
    if return_model:
        return result, model
    return result
