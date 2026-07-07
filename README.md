# LANTERN — independent reproduction

An independent, from-scratch reproduction of the key result of
[LANTERN](https://arxiv.org/abs/2507.03209) (Mehradfar et al.), which predicts lipid
nanoparticle (LNP) transfection efficiency from molecular structure.

The essential data (the 1,100-molecule `AGILE.csv` and the exact train/val/test split
files, ~90KB) is vendored under `data/`, credited to the authors'
[LANTERN](https://github.com/AsalMehradfar/LANTERN) repository (MIT). All featurization,
models, and training here are re-implemented from the paper's Methods. The upstream repo
is used **only as an answer key** and is optional — clone it to `./LANTERN` if you want
to run `--validate` (checks our features against their precomputed fingerprint pickles)
or `--source answerkey` (trains on their pickles instead of ours).

**Result:** the headline MLP reproduces at **R² = 0.8123 ± 0.0107** (paper: 0.8161), and
a Murcko-scaffold generalization probe shows the result is in-distribution only. See
[FINDINGS.md](FINDINGS.md).

## Explaining it

Two plain-language companions to the code:

- **Interactive explainer** — **live at
  [cyrus-progress.github.io/lantern-reproduction](https://cyrus-progress.github.io/lantern-reproduction/)**
  (source: [`website/index.html`](website/index.html)). A self-contained page with a
  High School / Undergrad / PhD depth toggle, an interactive predicted-vs-true scatter,
  a random-vs-scaffold results chart, and dark mode. Also opens directly in any browser
  (no build or server needed).
- **[WALKTHROUGH.md](WALKTHROUGH.md)** — a step-by-step, tenth-grade-level narrative of the
  entire project, plus a glossary.

## Layout

```
featurize.py          SMILES -> (count Morgan 2048, expert RDKit 210, GROVER 4800); --build/--validate
models.py             kNN / RF / SVR / LightGBM baselines + 7-layer PyTorch MLP, one fit/predict API
train.py              assemble features, MinMax preprocessing (leak / no-leak), split, train, score
evaluate.py           R2 / RMSE / MAE / Pearson r, and the predicted-vs-true scatter plot
plot_robustness.py    Phase-2 grouped bar chart (R2 mean+/-std, random vs scaffold)
run_matrix.py         full model x split x leak/no-leak matrix -> results/stronger_models_matrix.csv
plot_matrix.py        figures for the stronger-models matrix + the data-leak effect
run_all.py            one-command reproduction (features -> models -> robustness figure)
export_model.py       freeze the MLP for the demo backend -> serve/artifacts/ (numpy)
serve/                demo backend (FastAPI on a Hugging Face Space); see serve/README.md
tests/                pytest suite (fast, deterministic; run in CI)
data/                 vendored: AGILE.csv + train/val/test split files (from upstream)
features/             cached feature matrices (git-ignored; built by featurize.py)
results/              metrics CSVs + figures
LANTERN/              (optional, git-ignored) upstream clone; only for --validate / answerkey / GROVER
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned; or: make install
```

DeepChem (used by the authors for the count Morgan fingerprint) is **not** required —
we reproduce it with RDKit's Morgan generator and validate the match. See FINDINGS §3.

## Run

```bash
source .venv/bin/activate

# 1. Featurization: build the feature cache from the vendored data/AGILE.csv
python featurize.py --build          # writes features/{circular,expert,target}.npy
# (optional) prove the features match the upstream answer-key pickles:
#   git clone https://github.com/AsalMehradfar/LANTERN
#   python featurize.py --validate   # 1052/1100 circular exact, 209/210 expert exact

# 2. Reproduce the headline table (random split), all models + MLP scatter
python train.py --model all --split random --seed 0

# 3. Generalization probe (Murcko scaffold split)
python train.py --model all --split scaffold --seed 0

# 4. Robustness: 10 seeds per model, each split, then the bar chart
python train.py --model all --split random   --seeds 10
python train.py --model all --split scaffold --seeds 10
python plot_robustness.py
```

Useful flags: `--model {knn,rf,svr,lgbm,mlp,all}`, `--split {random,scaffold,scaffold_balanced}`,
`--features circular,expert,grover` (ablate blocks), `--no-leak` (fit the scaler on train rows
only — the honest protocol), `--source {computed,answerkey}`, `--seeds N` (mean±std).

### Extended experiments

```bash
python run_all.py           # one-command core reproduction (features -> models -> figure)
python run_matrix.py        # full matrix: every model x 3 splits x leak/no-leak, 10 seeds
python plot_matrix.py       # figures for the matrix + the leak-inflation effect
pytest -q                   # fast test suite (also runs in CI)
```

Adds **LightGBM** and a **GROVER-representation** model (MLP on the 4800-dim pretrained-GNN
embeddings), the previously-unused **`scaffold_balanced`** split, and a **no-leak** re-run that
quantifies how much the authors' MinMax leak inflates scores.

### Interactive demo

`serve/` is a tiny FastAPI backend (deployable free on a Hugging Face Space) that predicts a
pasted SMILES using the exact training featurization. `python export_model.py` freezes the model
into `serve/artifacts/`; the website's "Try it" section calls the Space. See
[serve/README.md](serve/README.md).

## Key results

Test R² (10-seed means, leaked scaling to match the paper):

| Model | Random | Murcko scaffold | Scaffold-balanced |
|---|---|---|---|
| kNN | 0.618 | **0.595** | −0.193 |
| RF | 0.726 | 0.445 | 0.273 |
| SVR | 0.731 | 0.316 | 0.137 |
| LightGBM | 0.760 | 0.450 | **0.454** |
| **MLP** | **0.812** | 0.487 | 0.176 |
| AGILE (cited) | 0.266 | 0.006 | — |

The best model **depends on the split**: MLP in-distribution (random), kNN on Murcko scaffold,
LightGBM on scaffold-balanced — where kNN collapses below zero. So kNN's OOD robustness is *not*
universal; LightGBM is the most consistent across both novel-scaffold splits. GROVER
(pretrained-GNN) embeddings underperform and even dilute the classical features. The MinMax leak
inflates the MLP by ~0.008 R² (≤0.003 for others) — negligible and ranking-preserving. Full
discussion in [REPORT.md](REPORT.md) and [FINDINGS.md](FINDINGS.md).

## Notes / deviations

- Fingerprint is unbounded-radius count Morgan (DeepChem `CircularFingerprint(1024, ...)`),
  **not** radius 2 as some summaries state. Validated to exact/count-preserving match.
- Preprocessing replicates the authors' `MinMaxScaler` fit on `[X | y]` over all rows
  (a mild, intentional-for-comparability target leak).
- AGILE is not re-run (pretrained GNN); its numbers are cited.
