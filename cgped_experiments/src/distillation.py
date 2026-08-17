"""
distillation.py
----------------
Standard response-based knowledge distillation (Hinton et al., 2015)
between a frozen teacher LM and a student LM, combined with the usual
hard-label cross-entropy term.
"""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, targets, pad_id=0,
            temperature: float = 2.0, alpha: float = 0.5):
    """
    student_logits, teacher_logits: (B, T, V)
    targets: (B, T) int64, shifted-by-one language modeling targets
    alpha: weight on hard-label CE; (1 - alpha) on soft KD term.
    """
    B, T, V = student_logits.shape
    mask = (targets != pad_id).float()

    ce = F.cross_entropy(
        student_logits.reshape(-1, V), targets.reshape(-1),
        ignore_index=pad_id, reduction="mean",
    )

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)  # (B,T)
    kd = (kd * mask).sum() / mask.sum().clamp_min(1.0)
    kd = kd * (temperature ** 2)

    total = alpha * ce + (1 - alpha) * kd
    return total, ce.detach(), kd.detach()
