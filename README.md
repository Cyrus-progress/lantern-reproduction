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

## Layout

```
featurize.py          SMILES -> (count Morgan 2048, expert RDKit 210); --validate, --build
models.py             kNN / RF / SVR baselines + 7-layer PyTorch MLP, one fit/predict API
train.py              assemble features, MinMax preprocessing, split, train, score vs paper
evaluate.py           R2 / RMSE / MAE / Pearson r, and the predicted-vs-true scatter plot
plot_robustness.py    Phase-2 grouped bar chart (R2 mean+/-std, random vs scaffold)
data/                 vendored: AGILE.csv + train/val/test split files (from upstream)
features/             cached feature matrices (git-ignored; built by featurize.py)
results/              metrics CSVs + figures
LANTERN/              (optional, git-ignored) upstream clone; only for --validate / answerkey
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy==1.26.4 pandas scikit-learn scipy matplotlib rdkit torch
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

Useful flags: `--model {knn,rf,svr,mlp,all}`, `--split {random,scaffold,scaffold_balanced}`,
`--features circular,expert` (ablate blocks), `--source {computed,answerkey}` (train on our
features or the authors' pickles), `--seeds N` (multi-seed mean±std).

## Key results

| Model | Random R² | Scaffold R² |
|---|---|---|
| kNN | 0.6178 | **0.5946** (best OOD) |
| RF  | 0.7256 | 0.4453 |
| SVR | 0.7310 | 0.3162 |
| **MLP** | **0.8123** (best ID) | 0.4868 |
| AGILE (cited) | 0.2655 | 0.0057 |

Random-split values are 10-seed means. The best in-distribution model (MLP) is *not* the
best on novel scaffolds (kNN) — full discussion in [FINDINGS.md](FINDINGS.md).

## Notes / deviations

- Fingerprint is unbounded-radius count Morgan (DeepChem `CircularFingerprint(1024, ...)`),
  **not** radius 2 as some summaries state. Validated to exact/count-preserving match.
- Preprocessing replicates the authors' `MinMaxScaler` fit on `[X | y]` over all rows
  (a mild, intentional-for-comparability target leak).
- AGILE is not re-run (pretrained GNN); its numbers are cited.
