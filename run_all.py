"""
run_all.py -- one-command reproduction of the core LANTERN results + figures.

Rebuilds features from the vendored data, runs every model on the random and
Murcko-scaffold splits at 10 seeds, and regenerates the robustness figure. This
reproduces the headline result and the generalization probe end-to-end.

For the extended "stronger models" comparison (LightGBM + GROVER + all 3 splits +
no-leak), run  python run_matrix.py  instead (slower).

Usage:  python run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print("\n>>> " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    run("featurize.py", "--build")
    for split in ["random", "scaffold"]:
        run("train.py", "--model", "all", "--split", split, "--seeds", "10")
    run("plot_robustness.py")
    print("\nDone. Tables + figures are in results/. "
          "See FINDINGS.md and REPORT.md for the writeup.")


if __name__ == "__main__":
    main()
