# World Cup Predictor

Local-first Streamlit app for predicting football match outcomes from Kaggle data.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Prepare Kaggle CSVs and point `data_sources.toml` at them:

- `data/matches.csv` must include:
  - `date`
  - `home_team`
  - `away_team`
  - `home_score`
  - `away_score`
- `data/rankings.csv` is optional and should include:
  - `date`
  - `team`
  - `rank`

Then run:

```bash
python -m src.train --config data_sources.toml --output artifacts
```

## Run the app

```bash
streamlit run app.py
```

## Streamlit Cloud

- Deploy this repository directly to Streamlit Cloud using `streamlit_app.py` as the entrypoint.
- The app will create fallback demo artifacts automatically if `artifacts/` is empty.
- To use your own Kaggle-trained models, set `MODEL_DIR` to the artifact folder.
