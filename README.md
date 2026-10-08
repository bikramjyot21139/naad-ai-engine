# NAAD AI backend — research validation branch

This branch deliberately replaces synthetic audio inputs and misleading fallback scores with a fail-closed, **text-only research API**. It does not contain E-DAIC data or trained weights. Without an authorized, compatible model, `/api/health` and `/api/predict` return `503`; this is intentional, not a routing fix candidate.

## E-DAIC licence and scope

The published E-DAIC end-user licence permits research use and forbids commercial use and redistribution. Do not use its transcripts, features, labels, or trained derivatives commercially without written authorization. Keep raw data, derived transcripts, model files, and manifests out of public Git. The baseline estimates an interview-level PHQ-8 research target only; it is not a questionnaire, diagnosis, emotion classifier, or crisis assessment.

## Colab / local research workflow

1. Obtain and sign the current E-DAIC licence. Download and extract the authorized participant archives privately. Do not commit data.
2. Install training packages: `python -m pip install -r requirements-training.txt`.
3. Build aligned participant-level data from the official split labels and participant transcript directories:

   `python -m src.etl_pipeline --data-dir /private/edaic/extracted --labels-dir /private/edaic/labels --output data/processed/edaic_text.csv`

   ETL requires a recognized transcript text column and a speaker column with identifiable participant turns. If the actual dataset uses different encodings, delimiters, headers, or speaker tags, stop and explicitly verify/map that format first; do not silently mix interviewer turns into participant text.
4. Install tests and run them before training:

   `python -m pip install -r requirements-test.txt`

   `python -m pytest -q`

5. Train the fixed TF-IDF + Ridge baseline:

   `python -m src.train_multimodal --dataset data/processed/edaic_text.csv --output-dir models`

   The script requires non-empty, participant-unique train/dev/test sets, keeps participant splits disjoint, uses only train for fitting, and never uses test for model selection. It compares dev MAE with a train-median baseline. A dev improvement is only a small engineering screening gate.
6. The model and manifest are local research artifacts. The manifest deliberately writes `deployable: false`, even if the screening gate passes, so training cannot accidentally enable a hosted API. Do not edit this flag to `true` yourself or upload artifacts to public GitHub.

## Serving / deployment warning

The API needs a compatible trusted scikit-learn joblib artifact and manifest. Keep `NAAD_RESEARCH_USE_ONLY=1` unset on public or commercial services. This repository has no approved trained model; predictions remain unavailable until rights, validation, artifact provenance, privacy, and the serving environment have been separately reviewed. Vercel is not automatically configured by running the training notebook.

Joblib uses pickle internally and can execute code during loading. Never load model artifacts from untrusted sources. Store research artifacts privately and only use them in a properly restricted research deployment.

## Future work

The text baseline does not support short chat messages, longitudinal baselines, voice emotion, or acute-risk detection. Any real audio model requires verified E-DAIC audio labels, documented preprocessing shared by training and inference, participant-disjoint evaluation, independent safety validation, and an authorized deployment pathway.
