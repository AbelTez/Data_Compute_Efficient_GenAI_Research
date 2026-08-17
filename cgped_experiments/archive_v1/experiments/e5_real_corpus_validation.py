"""
E5 — Real-Corpus Validation: Naive Baseline vs. Full CG-PED on Genuine Text
=============================================================================
E1-E4 use the fully controllable Synthetic Compositional benchmark (SC-LM)
to isolate CG-PED's mechanism under exact experimental control. E5 asks a
complementary question: does the same finding hold on a real, non-synthetic
English text corpus, with a difficulty score derived from genuine corpus
statistics rather than a synthetic ground truth?

The corpus is "Tiny Shakespeare" (~1.1MB of public-domain play text, a
standard small-scale char-level language-modeling benchmark), fetched from
a public GitHub mirror. This is real text: no part of the sequences,
vocabulary, or difficulty scores is synthetically generated. Difficulty is
0.5*norm(distinct-character count) + 0.5*norm(mean word rarity under the
corpus's own empirical frequency distribution) -- the same generic
two-axis definition used in E1-E4, computed from real corpus statistics
instead of a synthetic generator's ground truth.

Alongside the quantitative metrics used in E1-E4, this experiment also
generates qualitative text samples from both models, since for a real
corpus a qualitative reading of generated continuations is itself
informal evidence of learned character-level structure, not just a loss
number.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.data import RealTextConfig, RealTextDataset
from src.trainer import RunConfig, run_training, train_teacher
from src.plotting import plot_two_curves, bar_compare

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cache", "tinyshakespeare.txt")


@torch.no_grad()
def generate_sample(model, dataset, prompt: str, n_chars: int = 120,
                     temperature: float = 0.8, device="cpu"):
    model.eval()
    ids = [1] + [dataset.stoi.get(ch, 1) for ch in prompt]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n_chars):
        if ids.shape[1] >= dataset.cfg.max_len:
            break
        logits = model(ids)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    model.train()
    return dataset.decode(ids[0].tolist())


def main():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Real corpus not found at {CORPUS_PATH}. Download it first, e.g.:\n"
            f"  curl -s https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            f"data/tinyshakespeare/input.txt -o {CORPUS_PATH}"
        )

    cfg_data = RealTextConfig(corpus_path=CORPUS_PATH, chunk_len=64, max_examples=1500)
    ds = RealTextDataset(cfg_data)

    teacher_cfg = RunConfig(name="E5_teacher", seed=7, teacher_steps=250, lr=3e-3)
    teacher, teacher_info = train_teacher(ds, teacher_cfg)

    shared = dict(n_steps=300, eval_every=10, seed=7, lr=4e-3,
                  thresholds=(3.6, 3.0, 2.7, 2.6))

    rc_naive = RunConfig(
        name="E5_naive_baseline",
        use_curriculum=False, use_lora=False, use_distillation=False,
        **shared,
    )
    rc_cgped = RunConfig(
        name="E5_full_cgped",
        use_curriculum=True, use_lora=True, lora_r=4, lora_alpha=8,
        use_distillation=True, kd_temperature=2.0, kd_alpha=0.5,
        **shared,
    )

    res_naive, model_naive = run_training(ds, rc_naive, return_model=True)
    res_cgped, model_cgped = run_training(ds, rc_cgped, teacher=teacher, return_model=True)

    prompt = "ROMEO:\n"
    sample_naive = generate_sample(model_naive, ds, prompt)
    sample_cgped = generate_sample(model_cgped, ds, prompt)

    out = {
        "teacher": teacher_info,
        "naive_baseline": res_naive,
        "full_cgped": res_cgped,
        "qualitative_samples": {
            "prompt": prompt,
            "naive": sample_naive,
            "cgped": sample_cgped,
        },
        "corpus": {
            "name": "Tiny Shakespeare (public domain)",
            "source": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "vocab_size": ds.cfg.vocab_size,
            "n_train_examples": len(ds.train),
            "n_val_examples": len(ds.val),
            "chunk_len_chars": cfg_data.chunk_len,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e5_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_two_curves(
        res_naive["history"], res_cgped["history"],
        "Naive baseline", "Full CG-PED",
        "E5: Validation Loss vs. Training Step (Real Text Corpus)",
        "Validation loss (char-level next-token CE)",
        os.path.join(FIG_DIR, "e5_loss_curves.png"),
    )
    bar_compare(
        ["Naive", "CG-PED"],
        [res_naive["trainable_params"], res_cgped["trainable_params"]],
        "E5: Trainable Parameters (Real Text Corpus)", "# trainable params",
        os.path.join(FIG_DIR, "e5_trainable_params.png"),
    )

    param_reduction = 1 - res_cgped["trainable_params"] / res_naive["trainable_params"]
    print("=== E5: Real-Corpus Validation (Tiny Shakespeare) ===")
    print(f"Corpus -> {len(ds.train)} train / {len(ds.val)} val chunks, "
          f"vocab_size={ds.cfg.vocab_size}")
    print(f"Naive  -> trainable={res_naive['trainable_params']}  "
          f"final_val_loss={res_naive['final_val_loss']:.4f}  "
          f"val_ppl={res_naive['final_val_ppl']:.2f}")
    print(f"CG-PED -> trainable={res_cgped['trainable_params']}  "
          f"final_val_loss={res_cgped['final_val_loss']:.4f}  "
          f"val_ppl={res_cgped['final_val_ppl']:.2f}")
    print(f"Trainable-parameter reduction: {param_reduction*100:.1f}%")
    print(f"\nQualitative sample (prompt={prompt!r}):")
    print(f"  Naive : {sample_naive!r}")
    print(f"  CG-PED: {sample_cgped!r}")

    # ------------------------------------------------------------------
    # Catch-up analysis (mirrors E4b): does CG-PED's quality gap close
    # given more steps, while it keeps its 65% trainable-parameter saving
    # throughout? Honest reporting either way.
    # ------------------------------------------------------------------
    extended_cfg = RunConfig(
        name="E5_full_cgped_extended",
        use_curriculum=True, use_lora=True, lora_r=4, lora_alpha=8,
        use_distillation=True, kd_temperature=2.0, kd_alpha=0.5,
        n_steps=600, eval_every=10, seed=7, lr=4e-3,
        thresholds=(3.6, 3.0, 2.7, 2.6),
    )
    res_cgped_ext = run_training(ds, extended_cfg, teacher=teacher)
    target = res_naive["final_val_loss"]
    catch_up_step = None
    for h in res_cgped_ext["history"]:
        if h["val_loss"] <= target:
            catch_up_step = h["step"]
            break
    closest = min(res_cgped_ext["history"], key=lambda h: h["val_loss"])
    out["full_cgped_extended"] = res_cgped_ext
    out["catch_up_step"] = catch_up_step
    out["closest_approach"] = closest
    with open(os.path.join(RESULTS_DIR, "e5_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot_two_curves(
        res_naive["history"], res_cgped_ext["history"],
        "Naive baseline (300 steps)", "Full CG-PED (extended budget)",
        "E5b: CG-PED Catch-Up Analysis (Real Text Corpus)",
        "Validation loss (char-level next-token CE)",
        os.path.join(FIG_DIR, "e5b_catchup_curves.png"),
    )

    gap_pct = (closest["val_loss"] - target) / target * 100
    if catch_up_step is not None:
        print(f"\nCatch-up analysis: CG-PED reaches naive's 300-step quality "
              f"(val_loss<={target:.4f}) at step {catch_up_step}, "
              f"while using {param_reduction*100:.0f}% fewer trainable parameters throughout.")
    else:
        print(f"\nCatch-up analysis: within a 600-step extended budget, CG-PED's closest "
              f"approach to naive's 300-step quality is val_loss={closest['val_loss']:.4f} "
              f"at step {closest['step']} (naive target={target:.4f}, a {gap_pct:.1f}% gap), "
              f"while using {param_reduction*100:.0f}% fewer trainable parameters throughout.")
    return out


if __name__ == "__main__":
    main()
