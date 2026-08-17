# CG-PED — experiment project

**Do curriculum learning, LoRA, and knowledge distillation compose?**

This is the runnable codebase behind `../CG-PED_Research_Paper.docx`. It
implements CG-PED — a small generative language model trained with a
structure- and rarity-informed curriculum, LoRA adapters, and knowledge
distillation from a larger teacher — and tests one question:

> Three techniques each reduce the cost of training a generative model in a
> different way. Do their benefits add up when you use all three at once?

They add up almost exactly — and the sum is negative, because only one of the
three effects is large and it is a cost. The paper is organised around that
result and around explaining each component's price. It is a negative result
about the method, reported as one rather than reframed.

## Quick start

```bash
pip install -r requirements.txt
python3 experiments/verify.py     # pipeline self-checks, ~2 seconds
python3 experiments/run_all.py    # everything, ~35-45 min on one CPU core
```

No GPU, no Hugging Face, no model hub. The only network access is a one-time
1.1 MB download of the public-domain corpus, checked against a recorded
SHA-256.

## Design

A single **2³ factorial**: curriculum (C) × LoRA (L) × distillation (D), all
eight combinations, three seeds each, on two benchmarks. Cells are named by
which factors are on — `---` is the naive baseline, `CLD` is full CG-PED.

| | Data axis | Adaptation | Loss |
|---|---|---|---|
| `---` | random order | full fine-tune | cross-entropy |
| `C--` | competence-paced curriculum | full fine-tune | cross-entropy |
| `-L-` | random order | LoRA r=4 | cross-entropy |
| `--D` | random order | full fine-tune | CE + KD |
| `CL-` `C-D` `-LD` | the three pairs | | |
| `CLD` | curriculum | LoRA r=4 | CE + KD |

A factorial rather than a chain of pairwise ablations, because the research
question is about the **interaction** between the factors, and an interaction
cannot be read off one-factor-at-a-time comparisons. With all eight cells you
get each factor's main effect *and* you can compare what the full composition
achieves against what the individual effects predict it should achieve. The
gap between those two numbers is the paper's headline.

Everything else is held identical across cells: architecture, optimizer,
learning rate, batch size, step budget, and the seed for each row.

## Benchmarks

- **SC-LM (synthetic)** — sequences built by composing a root token with a
  variable number of affixes, roots drawn from a Zipfian distribution. Every
  example has a known, noise-free ground-truth difficulty, which is what makes
  it useful for isolating a curriculum's mechanism. It is a controlled testbed,
  not a stand-in for any real language.
- **Tiny Shakespeare (real)** — ~1.1 MB of public-domain play text at the
  character level. Difficulty comes from the corpus's own word-frequency
  statistics, not from a generator. Included because a curriculum validated
  only on data whose difficulty you defined yourself proves very little.

## Files

```
src/
  data.py          both benchmarks + the shared difficulty scorer
  model.py         TinyTransformerLM, LoRALinear, assert_lora_live
  distillation.py  the combined CE + KD loss
  trainer.py       one configurable training loop used by every cell
  train_utils.py   seeding, token-weighted evaluation, timing
  plotting.py      matplotlib helpers (Agg backend)
experiments/
  verify.py        pipeline self-checks — run this first
  run_factorial.py the main study
  run_budget.py    extended-budget and equal-wall-clock follow-up
  make_figures.py  all five figures, from the JSON
  make_paper.py    regenerates the .docx from the JSON
  run_all.py       the whole thing, in order
results/           factorial.json, budget.json
figures/           the five figures in the paper
archive_v1/        the earlier five-study version, kept for provenance
```

## Verification

`verify.py` runs five checks before any training happens. Each exists because
the property it tests failed silently at least once here:

1. every LoRA adapter receives gradient
2. adapters sit on the projections the forward pass actually calls
3. wrapped base weights are frozen, and the trainable count really drops
4. the adapted model is numerically identical to the base model at init
5. the curriculum genuinely restricts the early pool

Check 1 is the important one. In the first version of this project the
attention projections were aliased onto the enclosing `Block`, so the wrapper
was installed on the alias while the forward pass called the original. The
query, value and output adapters received no gradient for the whole of
training. Nothing revealed it: the runs completed, the loss curves looked
normal, and the trainable-parameter count still looked right. The paper's
Section 8 discusses this; the fix generalises well beyond this project.

## Honest scope

CPU-only, from-scratch models of ~80K parameters, a few thousand training
examples, 300–900 steps, minutes per run. Every number reported is a real
measurement from an executed run — none are projected or copied from other
papers — but they establish how the mechanism behaves at this scale, not how
it would behave at deployment scale. Three seeds per cell is enough to see
which differences survive seed noise and which do not; it is not enough for a
formal significance test, and the paper does not claim one.

## Reproducing

`run_all.py` regenerates every number and figure from scratch. Seeds are fixed
(7, 17, 27), `torch.set_num_threads(1)` keeps CPU reduction order stable, and
the corpus is checksummed. Results reproduce to about three decimal places on
comparable hardware; they are **not** bit-identical across different PyTorch
builds or CPU instruction sets, and the paper says so rather than claiming
exact determinism.
