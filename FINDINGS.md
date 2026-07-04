# FINDINGS — Independent reproduction of LANTERN

**Question.** Can the LANTERN paper's headline result — an MLP that predicts lipid
nanoparticle transfection efficiency at **R² ≈ 0.82** — be reproduced by rebuilding
the pipeline from scratch? And does it survive a harder generalization test?

**Answer.** Yes, it reproduces within noise. And no, it does not generalize to novel
chemical scaffolds — where the simplest model (kNN) becomes both the best *and* the
most stable.

Everything below uses the authors' **exact** train/val/test index splits, so the
numbers are directly comparable. Features were computed by our own `featurize.py`
(not their code) and validated against their answer-key pickles.

---

## 1. Did the headline result reproduce? (random split)

MLP on [count Morgan 2048 + expert RDKit 210], mean ± std over 10 seeds:

| Metric | This reproduction | Paper | Within noise? |
|---|---|---|---|
| **R²** | **0.8123 ± 0.0107** | 0.8161 | ✅ yes |
| RMSE | 1.4450 ± 0.0414 | 1.4308 | ✅ yes |
| MAE | 1.0978 ± 0.0293 | 1.1003 | ✅ yes |
| Pearson r | 0.9023 ± 0.0064 | 0.9053 | ✅ yes |

The paper's 0.8161 lies inside our ±1σ band (0.8016–0.8230). **Acceptance criterion met.**

Full model table, random split (single seed 0 for baselines, which are deterministic
except RF):

| Model | This repro R² | Paper R² |
|---|---|---|
| kNN | 0.6178 | 0.6148 |
| RF  | 0.7328 | 0.7169 |
| SVR | 0.7310 | 0.7285 |
| **MLP** | **0.7990** (0.8123 over 10 seeds) | **0.8161** |
| AGILE (cited, not re-run) | — | 0.2655 |

The paper's central claim reproduces: **even kNN (0.62) far outperforms the heavyweight
AGILE graph neural net (0.27)** on this refined dataset. Simple structural features +
a small model beat the pretrained GNN.

---

## 2. What the scaffold split revealed (the generalization probe)

Re-running everything under the **Murcko scaffold split** (test molecules have
structurally novel cores, absent from training) — mean ± std over 10 seeds:

| Model | Random R² | Scaffold R² | Drop (Δ) |
|---|---|---|---|
| kNN | 0.6178 ± 0.0000 | **0.5946 ± 0.0000** | **+0.0233** |
| RF  | 0.7256 ± 0.0077 | 0.4453 ± 0.0209 | +0.2803 |
| SVR | 0.7310 ± 0.0000 | 0.3162 ± 0.0000 | +0.4147 |
| MLP | 0.8123 ± 0.0107 | 0.4868 ± 0.0552 | +0.3255 |
| AGILE (cited) | 0.2655 | 0.0057 | +0.2598 |

Single-seed scaffold numbers match the paper closely (kNN 0.5946 vs 0.5919; SVR 0.3162
vs 0.3157; RF 0.4434 vs 0.4747; MLP 0.4904 vs 0.4532), so the reproduction extends to
the second split too.

**Three findings, in order of importance:**

1. **The headline model does not generalize.** The MLP collapses from R² 0.81 → 0.49
   the moment the test set contains genuinely new scaffolds. Most of its apparent skill
   on the random split was recognizing near-duplicates of training molecules, not
   learning transferable structure–function rules.

2. **The best in-distribution model is the worst choice out-of-distribution — and vice
   versa.** On the random split the ranking is MLP > SVR > RF > kNN. On the scaffold
   split it *inverts* to **kNN > MLP > RF > SVR**. kNN barely moves (Δ = 0.02) because
   "find the most similar known lipids and average them" degrades gracefully when the
   test molecules are new; the flexible models overfit the training distribution and
   fall hardest. **Picking a model on a random split would pick exactly the wrong model
   for real-world deployment on novel lipids.**

3. **Variance blow-up (not reported in the paper).** Beyond the mean collapsing, the
   MLP's *run-to-run* instability grows **5×**: R² std goes from ±0.0107 (random) to
   ±0.0552 (scaffold). So under distribution shift the neural net is not just worse on
   average — it's unpredictable, swinging ~0.11 in R² between random seeds. The
   deterministic kNN/SVR have zero seed variance, and kNN is therefore the single most
   *reliable* model for novel chemistry. This is our robustness contribution on top of
   the paper.

Figure: `results/robustness_R2_random_vs_scaffold.png` (grouped bars, error bars = seed std).

---

## 3. Featurization reproduction (the hard part)

We rebuilt both feature blocks independently and validated against the answer-key
pickles (`python featurize.py --validate`):

- **Circular (count Morgan, 2048-dim): 1052/1100 molecules match exactly.** The
  remaining 48 have *identical total counts* merely redistributed across bit indices
  (max per-bit |diff| = 3) — a cosmetic Morgan-hashing change between RDKit versions.
  **Key correction to the project brief:** the fingerprint is **not radius 2**. The
  authors' code uses DeepChem `CircularFingerprint(1024, is_counts_based=True,
  chiral=True)` — i.e. radius **1024** (effectively unbounded; it saturates near
  radius 50 for these lipids), which is why each molecule lights up ~420 bits, not ~56.
  We reproduce it with RDKit's Morgan generator at a large radius with chirality on.

- **Expert (RDKit descriptors, 210-dim): 209/210 columns match exactly.** These are
  DeepChem's `RDKitDescriptors` in ASCII-sorted name order — the full current set of
  217 minus the 7 that newer RDKit added (`NumAmideBonds`, `NumAtomStereoCenters`,
  `NumBridgeheadAtoms`, `NumHeterocycles`, `NumSpiroAtoms`,
  `NumUnspecifiedAtomStereoCenters`, `Phi`), with `Ipc` computed as `avg=True`. The one
  imperfect column, `NumHAcceptors`, differs by ±1 on some molecules (r = 0.93) due to
  an RDKit definition change we cannot undo without the authors' exact RDKit build.
  One of 210 features, off by ±1, after scaling — negligible, and the matching baseline
  scores confirm it.

The featurization was validated *before* any modeling: baseline R² landing on the paper
(kNN 0.618 vs 0.615, SVR 0.731 vs 0.729) is independent confirmation the features are right.

---

## 4. Caveats and honesty notes

- **Mild target leak, replicated on purpose.** The authors fit a single
  `MinMaxScaler(feature_range=(-1,1))` on `hstack([X, y])` over **all 1100 rows,
  including test rows and the target column**. This leaks test-set feature/target range
  into scaling. We replicate it exactly so our numbers are comparable to theirs; it
  inflates all models' absolute scores slightly but does not change the qualitative
  conclusions or the between-model comparison.
- **AGILE was not re-run** (per constraint) — it is a pretrained GNN. Its R² = 0.2655
  (random) / 0.0057 (scaffold) are cited as fixed numbers.
- **"Early stopping"** in the paper is keep-best-validation-epoch (restore the weights
  from the lowest val-loss epoch) over a full 100-epoch run, not patience-based halting.

---

## 5. Bottom line

The LANTERN headline (**R² ≈ 0.82, MLP, random split**) is **real and independently
reproducible** — same result from scratch-built features and training code, on the
authors' own splits. But the scaffold-split probe shows that headline is an
*in-distribution* number: on truly novel lipid scaffolds the MLP loses ~40% of its R²
and becomes unstable, while a plain 5-nearest-neighbor model is both the most accurate
and the most reliable. For prospective design of new lipids, the paper's ranking of
models is misleading; kNN is the honest baseline to beat.
