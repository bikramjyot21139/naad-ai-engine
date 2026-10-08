from pathlib import Path

import pytest

from src.train_multimodal import train_baseline


def make_rows():
    rows = []
    for index in range(30):
        split = "train" if index < 20 else "dev" if index < 26 else "test"
        # Enough varied examples per split to exercise both TF-IDF vocabularies.
        rows.append({
            "participant_id": 1000 + index,
            "split": split,
            "transcript": f"participant statement number {index} describing varied stress sleep routine and day {index}",
            "phq8_score": float(index % 13),
        })
    return rows


def test_training_uses_fixed_split_and_never_marks_artifact_deployable(tmp_path):
    import pandas as pd

    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(make_rows()).to_csv(dataset, index=False)
    output_dir = tmp_path / "models"
    manifest = train_baseline(dataset, output_dir)

    assert manifest["n_train_participants"] == 20
    assert manifest["n_dev_participants"] == 6
    assert manifest["n_test_participants"] == 4
    assert manifest["test_split_used_for_selection"] is False
    assert manifest["deployable"] is False
    assert (output_dir / "phq8_text_baseline.joblib").is_file()
    assert (output_dir / "phq8_text_baseline.manifest.json").is_file()


def test_training_rejects_missing_required_data(tmp_path):
    import pandas as pd

    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([{
        "participant_id": 1,
        "split": "train",
        "transcript": "",
        "phq8_score": 2,
    }]).to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="Empty participant transcripts"):
        train_baseline(dataset, tmp_path / "models")
