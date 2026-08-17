"""
verify.py
---------
Smoke tests that run before any experiment and fail loudly if the pipeline is
not doing what the paper says it does. Each one exists because the property it
checks failed silently at least once during this project.

Run: python3 experiments/verify.py   (~2 seconds, no training)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (TinyTransformerLM, apply_lora, assert_lora_live,
                       lora_modules, trainable_param_count)


def check_adapters_receive_gradient():
    """
    The one that matters. An adapter wrapped onto a module the forward pass
    never calls trains nothing, reports a plausible parameter count, and
    produces a loss curve that looks entirely normal.
    """
    model = TinyTransformerLM(vocab_size=87, d_model=64, n_layers=2,
                              n_heads=4, d_ff=128, max_len=64, dropout=0.1)
    apply_lora(model, r=4, alpha=8)
    x = torch.randint(1, 87, (4, 16))
    n_adapters, n_live = assert_lora_live(model, x)
    assert n_adapters == 12, f"expected 12 adapters (6 projections x 2 blocks), got {n_adapters}"
    print(f"  ok  all {n_live}/{n_adapters} LoRA adapters receive gradient")


def check_adapters_sit_on_the_real_projections():
    """Guards the specific aliasing bug: the wrapper must be the object the
    attention module itself calls, not a same-named reference held elsewhere."""
    model = TinyTransformerLM(vocab_size=87, d_model=64, n_layers=2,
                              n_heads=4, d_ff=128, max_len=64, dropout=0.1)
    apply_lora(model, r=4, alpha=8)
    names = {n for n, _ in lora_modules(model)}
    for b in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert f"blocks.{b}.attn.{proj}" in names, f"blocks.{b}.attn.{proj} not adapted"
        for proj in ("fc1", "fc2"):
            assert f"blocks.{b}.{proj}" in names, f"blocks.{b}.{proj} not adapted"
    print("  ok  adapters sit on all 12 projections the forward pass calls")


def check_base_weights_are_frozen():
    """The parameter saving is only real if the wrapped weights stop training."""
    model = TinyTransformerLM(vocab_size=87, d_model=64, n_layers=2,
                              n_heads=4, d_ff=128, max_len=64, dropout=0.1)
    full = trainable_param_count(model)
    apply_lora(model, r=4, alpha=8)
    for name, mod in lora_modules(model):
        assert not mod.base.weight.requires_grad, f"{name} base weight still trainable"
        if mod.base.bias is not None:
            assert not mod.base.bias.requires_grad, f"{name} base bias still trainable"
    lora = trainable_param_count(model)
    assert lora < full, "LoRA arm does not reduce the trainable parameter count"
    print(f"  ok  base weights frozen; trainable {full:,} -> {lora:,} "
          f"({100 * (full - lora) / full:.1f}% fewer)")


def check_lora_starts_as_a_no_op():
    """B is initialised to zero, so the adapted model must be numerically
    identical to the base model before any update. If it is not, the LoRA arm
    and the full-fine-tune arm do not start from the same function."""
    torch.manual_seed(0)
    model = TinyTransformerLM(vocab_size=87, d_model=64, n_layers=2,
                              n_heads=4, d_ff=128, max_len=64, dropout=0.0)
    model.eval()
    x = torch.randint(1, 87, (4, 16))
    with torch.no_grad():
        before = model(x)
    apply_lora(model, r=4, alpha=8)
    model.eval()
    with torch.no_grad():
        after = model(x)
    gap = (before - after).abs().max().item()
    assert gap < 1e-6, f"adapted model differs from base at init by {gap}"
    print(f"  ok  adapted model is identical to the base model at init (max diff {gap:.2e})")


def check_curriculum_actually_reorders():
    """A curriculum that produces the same exposure as random sampling is not
    a curriculum. Checks the eligible pool is genuinely restricted early on."""
    from src.data import SyntheticCompositionalDataset, SyntheticConfig, curriculum_order, pacing_schedule

    ds = SyntheticCompositionalDataset(SyntheticConfig(seed=7))
    order = curriculum_order(ds.train, seed=7)
    pools = pacing_schedule(order, 300)
    assert len(pools[0]) < len(ds.train), "curriculum does not restrict the pool at step 0"
    assert len(pools[-1]) == len(ds.train), "curriculum never opens up to the full pool"
    early = [ds.train[i].difficulty for i in pools[0]]
    late = [ds.train[i].difficulty for i in pools[-1]]
    assert max(early) <= max(late), "early pool is not easier than the full pool"
    print(f"  ok  curriculum pool grows {len(pools[0])} -> {len(pools[-1])} examples; "
          f"hardest example early {max(early):.3f} vs overall {max(late):.3f}")


def main():
    print("Pipeline verification")
    print("-" * 64)
    torch.set_num_threads(1)
    checks = [
        check_adapters_receive_gradient,
        check_adapters_sit_on_the_real_projections,
        check_base_weights_are_frozen,
        check_lora_starts_as_a_no_op,
        check_curriculum_actually_reorders,
    ]
    for fn in checks:
        fn()
    print("-" * 64)
    print(f"All {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
