import re
import numpy as np
import librosa

def clean_transcript(text: str) -> str:
    """Standardizes transcript text by stripping noise tags and timestamps."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_acoustic_features(audio_path: str, target_sr=16000, duration=10.0) -> np.ndarray:
    """Extracts a 32-dimensional acoustic feature vector (MFCCs, F0 pitch, Spectral & RMS Energy)."""
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
        if len(y) < sr * 0.5:
            return None
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_max = float(np.max(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
        
        spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_roll = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        
        vector = np.concatenate([
            mfcc_mean, 
            mfcc_std, 
            [f0_mean, f0_max, f0_std, spec_cent, spec_roll, rms_energy]
        ]).astype(np.float32)
        
        return vector
    except Exception:
        return None
