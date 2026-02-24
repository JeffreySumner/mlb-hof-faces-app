# ⚾ Hall of Fame Face Predictor App

Streamlit app that predicts whether a player "looks like" a Hall of Famer based on their Baseball Reference headshot.

> **Disclaimer**: This is an educational/satirical project. A player's appearance has nothing to do with baseball ability or Hall of Fame worthiness.

## What this app now does

- Search players by `bbrefid`, Lahman `playerid`, player name, and optional team filter
- Fetch live headshots from Baseball Reference (with local image cache)
- Pull the trained Keras model artifact from Hugging Face and cache locally
- Apply the same inference preprocessing used in the blog pipeline (`32x32`, grayscale, flattened)
- Show probabilities and threshold-based prediction

## Project structure

```text
mlb-hof-faces-app/
├── app.py
├── config.py
├── pyproject.toml
├── services/
│   ├── model_loader.py
│   ├── player_lookup.py
│   └── bbref_image.py
├── inference/
│   ├── preprocess.py
│   └── predict.py
├── model/
│   └── train_model.py
└── data/
```

## Quick start

1. Install dependencies:

```bash
uv sync
```

2. Run the app:

```bash
uv run streamlit run app.py
```

3. Open:

- `http://localhost:8501`

## Model artifact location

By default the app expects model artifact in Hugging Face dataset repo:

- Repo: `rpy-ai/mlb-hof-faces`
- File: `artifacts/mlb_hof_model.keras`

Override with env vars if needed:

- `HOF_MODEL_REPO_ID`
- `HOF_MODEL_FILENAME`

## Deployment notes (Posit Connect Cloud)

Streamlit is supported. For deployment:

1. Export requirements if needed by your deploy flow:

```bash
uv pip compile pyproject.toml -o requirements.txt
```

2. Set environment variables (if using non-default HF artifact path).
3. Deploy app from repo root.

## Data sources

- Player metadata: Lahman database via `pylahman`
- Player images: Baseball Reference
- Model artifact: Hugging Face dataset storage

