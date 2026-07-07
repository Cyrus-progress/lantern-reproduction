# LANTERN reproduction -- common tasks.
# Assumes an activated venv (see requirements.txt).

.PHONY: help install features validate reproduce matrix figures test clean

help:
	@echo "make install    - install pinned dependencies"
	@echo "make features    - build the feature cache from data/AGILE.csv"
	@echo "make validate    - check featurization against the answer key (needs ./LANTERN)"
	@echo "make reproduce   - core results + robustness figure (random + scaffold, 10 seeds)"
	@echo "make matrix      - full stronger-models matrix (all models x 3 splits x leak/no-leak)"
	@echo "make test        - run the fast test suite"
	@echo "make clean       - remove the feature cache"

install:
	pip install -r requirements.txt

features:
	python featurize.py --build

validate:
	python featurize.py --validate

reproduce:
	python run_all.py

matrix:
	python run_matrix.py

figures:
	python plot_robustness.py

test:
	pytest -q

clean:
	rm -rf features/*.npy
