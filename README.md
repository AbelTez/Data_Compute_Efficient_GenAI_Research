# CG-PED — final deliverable

**Do curriculum learning, LoRA, and knowledge distillation compose?**
A factorial test of three efficiency techniques for training small generative
language models — and why composing them does not help.

Research area: *Data and Compute-Efficient Generative AI*
Author: Abel Tezare, Software Engineering Student, Addis Ababa University

```
CG-PED_Research_Paper.docx    the paper
cgped_experiments/            the runnable project
    README.md                 project documentation
    src/                      implementation
    experiments/              the study, the figures, and this paper's generator
    results/                  factorial.json, budget.json
    figures/                  the five figures in the paper
    data_cache/               the public-domain corpus (checksummed)
    archive_v1/               the earlier five-study version, kept for provenance
```

## The point of the research, in one paragraph

Three techniques each cut the cost of training a generative model in a
different way, and each is normally studied alone. A practitioner under a
budget has no reason to pick just one — so the implicit assumption is that
using all three compounds their benefits. This project tests that assumption
with a full 2³ factorial: all eight combinations of curriculum, LoRA and
distillation, three seeds each, at a fixed step budget, on a synthetic
benchmark and on real text.

**They compose almost perfectly** — the full combination lands within 0.0003
and 0.0008 nats of what its three separate effects predict, about a tenth of
the seed-to-seed spread. **And that is exactly why it loses.** Because nothing
compounds, the composition is worth the plain sum of its parts, and only one
of the three effects is large: LoRA costs +0.027 nats on synthetic data and
+0.278 on real text at from-scratch scale, 28× the seed noise. Distillation
helps, but an order of magnitude less, and only in proportion to how much
better the teacher is than the student. The curriculum never clears its own
noise floor and changes sign between the two benchmarks. Added up, full CG-PED
trails the naive baseline by 0.015 and 0.254 nats while saving 72–74% of
trainable parameters.

The contribution is the per-component price list that makes all of that
readable, the additivity result that makes it predictive (if the components
add, single-factor runs are enough to predict any combination), an
equal-wall-clock comparison much less flattering than the equal-step one
usually reported, and a silent implementation failure (paper, Section 8) that
produced a complete but fictitious set of LoRA results before it was caught.
The study ends by recommending a configuration that is *not* its own method.

## Start here

1. **Read the paper.** The Summary table on page 1 is the entire result. If
   you read nothing else, read that. After it: the question is stated in
   Section 1, the evidence is Tables 2 and 3 with Figure 3, and Section 6
   explains the mechanism behind each number.
2. **Check it yourself.** `cd cgped_experiments && pip install -r
   requirements.txt && python3 experiments/verify.py` runs the five pipeline
   self-checks in about two seconds without training anything.
3. **Reproduce it.** `python3 experiments/run_all.py` regenerates every number
   and figure from scratch in 35–45 minutes on one CPU core. No GPU, no model
   hub. The paper itself is regenerated from the results JSON by
   `experiments/make_paper.py`, so no number in it is typed by hand.

## What this does and does not show

It shows that at this scale — ~80K-parameter from-scratch models, a few
thousand examples, a few hundred steps — the three techniques act
independently, that LoRA's cost dominates and grows tenfold moving from
synthetic to real data, that distillation's value tracks the teacher–student
gap rather than the technique, and that the parameter saving is real and
unaffected by any of it. Every number is a measurement from an executed run;
none is projected or copied from another paper.

It does not show how any of this behaves at deployment scale, and it does not
evaluate LoRA under the conditions LoRA was designed for, since the student
here is trained from random initialisation rather than adapted from a
pretrained checkpoint. Both limitations are stated in the paper's Section 7
rather than left for a reader to find.

## A note on the result being negative

A research contribution is judged by whether it is validated and cleanly
reasoned, not by whether the method wins. A negative
result that is measured cleanly, exceeds its own noise floor, and comes with a
mechanism is more useful to the next person than a marginal positive one that
a change of seed would erase — and this project has an example of the latter
in its own history, which is why Section 8 exists.
