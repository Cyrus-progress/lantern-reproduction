# An Independent Reproduction and Robustness Study of LANTERN

*A machine-learning pipeline for predicting lipid-nanoparticle transfection efficiency.*

## Abstract

We independently reproduce the headline result of LANTERN (Mehradfar et al.,
[arXiv:2507.03209](https://arxiv.org/abs/2507.03209)) — an MLP that predicts lipid-nanoparticle
(LNP) transfection efficiency from molecular structure — rebuilding the featurization, baselines,
and model from the paper's Methods and validating every feature against the authors' released
artifacts. On the authors' exact random split we recover **R² = 0.812 ± 0.011** (paper: 0.816),
matching all four reported metrics within seed variance. We then extend the study three ways the
paper does not: (i) a **Murcko scaffold split** showing the MLP collapses to R² ≈ 0.49 while a
plain 5-nearest-neighbour model becomes both the most accurate and the most stable predictor on
novel scaffolds; (ii) **stronger and alternative models** — LightGBM and a GROVER pretrained-GNN
representation — none of which overturn that conclusion; and (iii) a **leak-free re-run** that
quantifies the small inflation from the authors' joint MinMax scaling. The takeaway: LANTERN's
in-distribution result is real and reproducible, but in-distribution leaderboards mis-rank models
for prospective, novel-scaffold lipid design.

## 1. Introduction

Lipid nanoparticles deliver mRNA into cells; the ionizable-lipid structure dominates delivery
performance ("transfection efficiency"). LANTERN reports that classical molecular representations
with shallow models rival a purpose-built pretrained graph neural network (AGILE) on a refined
1,100-compound dataset. Because this is a strong claim, we test it by rebuilding the pipeline from
scratch — using the authors' materials only as an answer key — and then probe how far the result
generalizes.

## 2. Data and featurization

**Dataset.** The refined AGILE set: 1,100 (SMILES, transfection-efficiency) pairs, target on a
roughly log-scaled range ≈ [−2.3, 16]. We use the authors' exact train/val/test index splits
(880/110/110) for three schemes: `random`, `Murcko_scaffold`, and `scaffold_balanced`.

**Features (2,258-dim).** Each molecule is mapped to `[circular(2048) ‖ expert(210)]`:
- *Circular*: a count-based Morgan fingerprint. Reproducing the answer key required matching
  DeepChem's `CircularFingerprint(radius=1024, is_counts_based=True, chiral=True)` — an effectively
  **unbounded** radius (not the textbook radius 2), which populates ~420 bins per molecule vs ~56.
  Our RDKit reimplementation matches **1,052 / 1,100** molecules exactly; the remaining 48 differ
  only by version-dependent hash reshuffling (total counts invariant, max per-bin |Δ| = 3).
- *Expert*: DeepChem's `RDKitDescriptors` — the current 217 RDKit descriptors, ASCII-sorted, minus
  7 newer additions, with `Ipc` in average mode. We match **209 / 210** columns exactly (the lone
  `NumHAcceptors` drift is ≤1 per molecule, r = 0.93, from an RDKit version change).

A third representation, **GROVER** (4,800-dim pretrained-GNN embeddings), is available for the
dataset molecules and used in the ablation below.

## 3. Models and protocol

Five models behind one `fit/predict` interface: **kNN** (k=5), **Random Forest** (100 trees),
**SVR** (RBF), **LightGBM** (400 trees), and the paper's **MLP** (7 layers,
2258→200→300→500→500→300→200→1, ReLU, Adam lr 2e-4, MSE, ≤100 epochs with best-validation-epoch
selection). Metrics: R², RMSE, MAE, Pearson r, computed on inverse-transformed targets. Neural
results are reported as mean ± std over 10 seeds. AGILE is not re-run; its published numbers are
cited.

**The scaling leak.** The authors fit a single `MinMaxScaler` on `[X | y]` over *all* rows,
including the test rows and the target column. We replicate this for comparability (`leak=True`)
and also provide an honest path (`leak=False`) that fits the scaler on train rows only.

## 4. Results

### 4.1 Headline reproduction (random split)

The MLP reproduces within seed variance on every metric:

| Metric | This reproduction (10 seeds) | Paper | Verdict |
|---|---|---|---|
| R² | **0.8123 ± 0.0107** | 0.8161 | within noise |
| RMSE | 1.4450 ± 0.0414 | 1.4308 | within noise |
| MAE | 1.0978 ± 0.0293 | 1.1003 | within noise |
| Pearson r | 0.9023 ± 0.0064 | 0.9053 | within noise |

Baselines match too (kNN 0.618 vs 0.615; SVR 0.731 vs 0.729), and kNN (0.62) far exceeds the cited
AGILE (0.27) — the paper's central claim reproduces.

### 4.2 Generalization: the scaffold split

Under the Murcko scaffold split (novel test scaffolds), the ranking **inverts**:

| Model | Random R² | Scaffold R² | Δ |
|---|---|---|---|
| kNN | 0.6178 | **0.5946** | −0.02 |
| RF | 0.7256 | 0.4453 | −0.28 |
| SVR | 0.7310 | 0.3162 | −0.41 |
| MLP | 0.8123 | 0.4868 | −0.33 |

Three findings: the MLP loses ~40% of its skill on novel scaffolds; the best in-distribution model
(MLP) is the *worst* choice out-of-distribution while the simplest (kNN) is best; and the MLP's
seed variance inflates ~5× (σ 0.011 → 0.055) — it becomes unreliable, not merely worse.

### 4.3 Stronger models and feature representations

Test R² (10-seed mean, leaked scaling) across all three splits:

| Model | Random | Murcko scaffold | Scaffold-balanced |
|---|---|---|---|
| kNN | 0.618 | **0.595** | −0.193 |
| RF | 0.726 | 0.445 | 0.273 |
| SVR | 0.731 | 0.316 | 0.137 |
| LightGBM | 0.760 | 0.450 | **0.454** |
| MLP | **0.812** | 0.487 | 0.176 |
| MLP · GROVER only | 0.520 | 0.226 | −0.230 |
| MLP · circ+exp+GROVER | 0.762 | 0.254 | 0.097 |

Three points. **(a) LightGBM** is a strong new baseline (second on random) and the *most consistent*
model on the two hard splits (≈0.45 on both). **(b) GROVER underperforms**: the pretrained-GNN
embedding alone reaches only 0.52 on random, and *adding* it to circular+expert **dilutes** the MLP
(0.812 → 0.762). This reinforces the paper's thesis that classical features suffice here.
**(c) The best out-of-distribution model is split-dependent** — MLP on random, kNN on Murcko
scaffold, LightGBM on scaffold-balanced. Critically, kNN's much-touted OOD robustness is *not*
universal: on scaffold-balanced it drops below zero (worse than predicting the mean) while LightGBM
leads. A boosted-tree is the most reliable single choice across both novel-scaffold regimes.
(Figure: `results/stronger_models_R2.png`.)

### 4.4 The data leak, quantified

On the random split — the clean, apples-to-apples comparison — the replicated MinMax leak barely
moves scores and changes no ranking:

| Model | Leaked R² | Honest R² | Inflation |
|---|---|---|---|
| kNN | 0.6178 | 0.6226 | −0.0048 |
| RF | 0.7256 | 0.7243 | +0.0013 |
| SVR | 0.7310 | 0.7292 | +0.0018 |
| LightGBM | 0.7595 | 0.7561 | +0.0034 |
| MLP | 0.8123 | 0.8047 | +0.0076 |

The MLP is inflated most (+0.008 R², ~1%); trees/SVR by ≤0.003; kNN is marginally *lower* leaked.
So the leak we flagged is real but negligible and ranking-preserving. On the scaffold splits the
honest protocol is not directly comparable: fitting the scaler on train rows only leaves
novel-scaffold *test* features outside the scaled range, so the flexible models (SVR, MLP)
extrapolate and become numerically unstable (the MLP's R² diverges) — a symptom of the distribution
shift itself, not of the leak. (Figure: `results/leak_effect.png`.)

## 5. Discussion and limitations

The reproduction confirms LANTERN's in-distribution result and its "shallow beats pretrained GNN"
thesis. But the scaffold-split analysis shows that this leaderboard does not transfer to the task
that matters — predicting genuinely novel lipids — where a local nearest-neighbour estimator is a
stronger and more stable prior. Limitations: we do not re-run AGILE (cited); the replicated MinMax
leak mildly inflates absolute scores (quantified in §4.4); and one of 210 descriptors carries a
sub-unit version drift.

## 6. Reproducibility

Pinned `requirements.txt`; a fast deterministic `pytest` suite and GitHub Actions CI; and
one-command reproduction via `python run_all.py` (or `python run_matrix.py` for the full matrix).
DeepChem is not required — the fingerprint is reproduced with RDKit and validated. An interactive
demo (`serve/`, deployable on a free Hugging Face Space) serves exact predictions for arbitrary
SMILES.

## Verification

- Core: `python run_all.py` regenerates the §4.1–4.2 tables and the robustness figure.
- Matrix: `python run_matrix.py && python plot_matrix.py` regenerates §4.3–4.4.
- Tests: `pytest -q` (featurization fidelity, deterministic-model reproduction, split integrity).
