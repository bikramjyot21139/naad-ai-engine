from pathlib import Path

import joblib


def export_model(model, export_path: str | Path) -> Path:
    """Persist the scikit-learn research baseline as a trusted local artifact.

    This model is not ONNX: its TF-IDF vectorizers and Ridge estimator are packaged
    together in the fitted pipeline and loaded by the restricted research API.
    """
    path = Path(export_path)
    if path.suffix != ".joblib":
        raise ValueError("The text baseline is serialized as .joblib, not ONNX.")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
