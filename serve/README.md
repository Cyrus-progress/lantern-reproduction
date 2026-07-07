---
title: LANTERN Transfection Predictor
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# LANTERN transfection-efficiency predictor (demo backend)

A tiny FastAPI service behind the [LANTERN, Reproduced](https://cyrus-progress.github.io/lantern-reproduction/)
explainer. Paste a lipid SMILES → predicted transfection efficiency, an
applicability/reliability badge (distance to the nearest known lipid), the
nearest known lipids, and a 2D structure.

Predictions reuse the project's `featurize.py`, so featurization is identical to
training. Inference is pure numpy (the frozen MLP weights) — no torch/sklearn on
the server.

`POST /predict  {"smiles": "..."}` →
`{score, reliability, nearest_distance, domain_threshold, neighbors[], svg}`

## Deploy to a free Hugging Face Space

From the repo root, build the self-contained bundle (copies `featurize.py` in and
writes `serve/artifacts/`):

```bash
python export_model.py          # trains + freezes the model into serve/artifacts/
```

Then create a **Docker** Space and push the contents of `serve/`:

```bash
# one-time: create the Space at https://huggingface.co/new-space (SDK: Docker)
cd serve
git init && git remote add origin https://huggingface.co/spaces/<you>/lantern-predictor
git add -A && git commit -m "LANTERN demo backend" && git push -u origin main
```

The Space builds the Dockerfile and serves on port 7860. Copy its URL
(`https://<you>-lantern-predictor.hf.space`) into the website's `API_BASE`
constant (see the "Try it" section in `website/index.html`).

## Run locally

```bash
cd serve && pip install -r requirements.txt
uvicorn app:app --port 7860
curl -s localhost:7860/predict -H 'content-type: application/json' \
  -d '{"smiles":"CCCCCCCC/C=C\\CCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCC/C=C\\CCCCCCCC)NCCCN(C)CCCN"}'
```
