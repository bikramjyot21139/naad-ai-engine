import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.feature_extraction import clean_transcript, extract_acoustic_features

RAW_DATA_DIR = Path("/content/naad-ai-root/data")
PROCESSED_DIR = RAW_DATA_DIR / "processed"

def run_etl():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    audio_files = list(RAW_DATA_DIR.rglob("*.wav"))
    
    aligned_dataset = []
    acoustic_matrices = []

    for idx, audio_path in enumerate(tqdm(audio_files[:1500], desc="Extracting Features")):
        features = extract_acoustic_features(str(audio_path))
        if features is None:
            continue

        aligned_dataset.append({
            "sample_id": f"SAM_{idx:05d}",
            "dataset_origin": audio_path.parent.name,
            "audio_file": audio_path.name,
            "transcript": "Vocal recording sample",
            "emotion_label": "neutral",
            "depression_score": np.nan
        })
        acoustic_matrices.append(features)

    master_df = pd.DataFrame(aligned_dataset)
    master_df.to_csv(PROCESSED_DIR / "master_dataset.csv", index=False)
    if len(acoustic_matrices) > 0:
        np.save(PROCESSED_DIR / "acoustic_features_matrix.npy", np.vstack(acoustic_matrices))

if __name__ == "__main__":
    run_etl()
