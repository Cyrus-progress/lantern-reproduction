"""
Fast, deterministic regression tests for the LANTERN reproduction.

Designed to run in CI without the upstream LANTERN clone: featurization is built
from the vendored data/AGILE.csv, and the metric checks use only the deterministic
models (kNN / SVR / RF). The answer-key fidelity test skips gracefully when the
(large, un-vendored) fingerprint pickles are absent.

Run:  pytest -q
"""
import os

import numpy as np
import pytest

import featurize
import train


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
def test_featurize_shapes_and_determinism():
    smiles, _ = featurize.load_smiles_and_target()
    sample = smiles[:8]
    circ1, exp1 = featurize.featurize_smiles(sample)
    circ2, exp2 = featurize.featurize_smiles(sample)
    assert circ1.shape == (8, 2048)
    assert exp1.shape == (8, 210)
    # featurization must be deterministic
    assert np.array_equal(circ1, circ2)
    assert np.array_equal(exp1, exp2)


def test_circular_known_stats():
    """Molecule 0's count-Morgan total is version-invariant (see FINDINGS)."""
    smiles, _ = featurize.load_smiles_and_target()
    circ, _ = featurize.featurize_smiles(smiles[:1])
    assert circ[0].sum() == 633            # total count is preserved across RDKit versions
    assert 400 <= int((circ[0] > 0).sum()) <= 440


def test_expert_name_list():
    names = featurize.EXPERT_NAMES
    assert len(names) == 210
    assert names == sorted(names)          # ASCII-sorted (DeepChem ordering)
    assert featurize._EXPERT_EXCLUDE.isdisjoint(names)  # the 7 newer descriptors excluded


def test_featurize_combined_width():
    smiles, _ = featurize.load_smiles_and_target()
    X = featurize.featurize_combined(smiles[:4])
    assert X.shape == (4, 2258)            # what the models train on / the demo serves


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", ["random", "scaffold", "scaffold_balanced"])
def test_split_integrity(split):
    tr, va, te = train.load_split(split)
    assert len(tr) == 880 and len(va) == 110 and len(te) == 110
    allidx = np.concatenate([tr, va, te])
    assert len(set(allidx.tolist())) == 1100          # disjoint
    assert set(allidx.tolist()) == set(range(1100))   # covers every row


# --------------------------------------------------------------------------- #
# Pipeline reproduction (deterministic models only -> stable in CI)
# --------------------------------------------------------------------------- #
def test_knn_reproduces_random():
    m = train.run_experiment("knn", "random")
    assert abs(m["R2"] - 0.6178) < 0.03


def test_svr_reproduces_random():
    m = train.run_experiment("svr", "random")
    assert abs(m["R2"] - 0.7310) < 0.03


def test_rf_seeded_determinism():
    a = train.run_experiment("rf", "random", seed=0)
    b = train.run_experiment("rf", "random", seed=0)
    assert a["R2"] == b["R2"]              # same seed -> identical result


def test_noleak_path_runs():
    leaked = train.run_experiment("knn", "random", leak=True)
    honest = train.run_experiment("knn", "random", leak=False)
    for m in (leaked, honest):
        assert np.isfinite(m["R2"]) and -1.0 < m["R2"] < 1.0


# --------------------------------------------------------------------------- #
# Full answer-key fidelity (skips when the upstream pickles are absent)
# --------------------------------------------------------------------------- #
def test_matches_answer_key_if_present():
    key = os.path.join(featurize.ANSWER_KEY_DIR, "circular.pkl")
    if not os.path.isfile(key):
        pytest.skip("answer-key pickles not present (upstream LANTERN clone absent)")
    import pickle
    smiles, _ = featurize.load_smiles_and_target()
    circ, _ = featurize.featurize_smiles(smiles)
    ref = pickle.load(open(key, "rb"))
    exact = sum(np.array_equal(circ[i], np.asarray(ref[s], float))
                for i, s in enumerate(smiles))
    assert exact >= 1000                  # >=1000/1100 exact (rest are count-preserving)
