# Presence of a root conftest.py makes pytest add the repo root to sys.path,
# so tests can `import featurize` / `import train` regardless of how pytest is
# invoked (plain `pytest` in CI vs `python -m pytest` locally).
