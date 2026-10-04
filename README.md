# NAAD AI backend: validation branch

The repair branch replaces synthetic audio and misleading fallback scores with a **text-only research baseline interface**. It does not include trained E-DAIC weights, a tokenizer, or the E-DAIC data.

## Important use restriction

E-DAIC's published end-user licence allows research use only and forbids commercial use and redistribution. Do not use E-DAIC, its transcripts, features, or models trained on it commercially unless the rights holder grants written permission. Keep participant data and derived transcripts out of Git. The service is not a diagnosis, PHQ-8 questionnaire, or crisis assessment.

## Local research workflow

1. Obtain and sign the current E-DAIC EULA. Download the official labels and extract the licensed participant archives into a local, private directory. Never commit those files.
2. Install training dependencies: `python -m pip install -r requirements-training.txt`.
3. Align actual transcript rows to official participant split labels:

   `python -m src.etl_pipeline --data-dir /private/edaic/extracted --labels-dir /private/edaic/labels --output data/processed/edaic_text.csv`

4. Train the text-only participant-level PHQ-score baseline:

   `python -m src.train_multimodal --dataset data/processed/edaic_text.csv --output-dir models`

   The test split is not used for model selection. The script writes the selected model and metrics locally; both are ignored by Git. The model marks itself eligible only if it beats the training-median baseline on the supplied development split. This is an engineering gate, not clinical validation.
5. Install test requirements and run tests:

   `python -m pip install -r requirements-test.txt`

   `python -m pytest -q`

## Serving warning

The API expects a local trusted scikit-learn joblib model and checksum manifest. Keep `NAAD_RESEARCH_USE_ONLY=1` unset in production. For a local, authorized research evaluation only, after reviewing the EULA and model provenance, set `NAAD_RESEARCH_USE_ONLY=1` and configure `NAAD_MODEL_PATH`, `NAAD_MODEL_MANIFEST`, and `CORS_ALLOW_ORIGINS`.

Joblib deserialization can execute code. Never load a model artifact from an untrusted source. Do not commit model files or manifests to this public repository. There is currently no approved trained model on this branch; prediction should return unavailable until one is trained, reviewed, and supplied through a private research deployment.

## Scope

The first model estimates participant-level PHQ-8 score from interview text. It does not provide five-way emotion classification, diagnose depression, assess immediate self-harm risk, or support longitudinal personalization. Add real voice features only after specifying and testing identical training/serving preprocessing.
