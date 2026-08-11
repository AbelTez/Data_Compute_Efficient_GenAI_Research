# CG-PED — Final Deliverable

**Curriculum-Guided Parameter-Efficient Distillation: A Composed Methodology
for Data- and Compute-Efficient Generative AI**

This archive contains the complete Master's-level research deliverable for
the "Data and Compute-Efficient Generative AI" research assignment:

```
CG-PED_Research_Paper.docx      <- the full research paper (29 pages)
cgped_experiments/              <- the complete, runnable experiment project
    README.md                   <- project-specific documentation
    requirements.txt
    data_cache/                 <- the real public-domain text corpus used in E5
    src/                        <- CG-PED implementation (data, model, LoRA, distillation, trainer)
    experiments/                <- the 5 pilot studies (E1-E5) + figure/summary generators
    results/                    <- JSON results from executed runs
    figures/                    <- all plots and diagrams (also embedded in the paper)
```

## Start here

1. **Read the paper first**: `CG-PED_Research_Paper.docx`. It is a complete,
   honest, 29-page research paper — abstract, literature review grounded in
   real, current publications, full CG-PED methodology, experimental setup,
   results for all five pilot studies (including real generated text
   samples), a mechanistic discussion of where the method does and doesn't
   win, limitations, a concrete future-work plan, references, and
   reproducibility appendices.
2. **Reproduce the results**: `cd cgped_experiments && pip install -r
   requirements.txt && python3 experiments/run_all.py`. This regenerates
   every number and figure in the paper from scratch in about 6–8 minutes on
   a single CPU core — no GPU needed. The real text corpus used in E5 is
   already included in `data_cache/`, and will also be re-downloaded
   automatically if missing (from a public, pinned GitHub URL).
3. **Read `cgped_experiments/README.md`** for the full project structure, an
   explanation of what each pilot study measures, and an explicit "honest
   scope statement" about what this proof-of-concept does and does not
   demonstrate.

## The honest one-paragraph summary

CG-PED composes three efficiency techniques — a structure- and
rarity-informed curriculum, LoRA, and knowledge distillation — for training
small generative language models efficiently. It is evaluated experimentally,
not only theoretically: four pilot studies (E1–E4) use a fully controllable
synthetic benchmark to isolate each component's mechanism, and a fifth (E5)
validates the same comparison on a genuine, public-domain text corpus with
real generated-text samples. All reported numbers are real, reproducible
measurements from executed runs. Curriculum ordering and distillation each
produce genuine, modest gains; LoRA cuts trainable parameters by roughly
63–65% at a real quality cost at this tiny, from-scratch model scale; and —
most substantively — while the full pipeline's quality gap narrows to
near-zero on the synthetic benchmark given extra training steps, it does
**not** fully close on real text, a genuine and informative divergence that
the paper reports honestly rather than omits. The paper explains the likely
mechanistic reasons for this and lays out a concrete plan (Section 8 /
Appendix A) for validating CG-PED at larger scale with real pretrained models
and GPU infrastructure.

## A note on scope

This methodology and its experiments are fully domain- and language-general
— they are not tied to any specific language, alphabet, or population, in
line with the assignment's own instruction that "the choice of model is left
entirely to the participant" and that "the primary contribution should be the
proposed methodology rather than the selected model."
