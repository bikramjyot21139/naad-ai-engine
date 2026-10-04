from pathlib import Path

import pandas as pd
import pytest

from src.etl_pipeline import build_dataset


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_aligns_participant_text_to_official_split_and_target(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    for split in ("train", "dev", "test"):
        records = [{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}] if split == "train" else []
        write_csv(labels_dir / f"{split}_split.csv", records)
    folder = data_dir / "302_P"
    write_csv(folder / "302_Transcript.csv", [
        {"speaker": "Ellie", "value": "How are you?"},
        {"speaker": "Participant", "value": " I feel tired. "},
    ])
    result = build_dataset(data_dir, labels_dir)
    assert len(result) == 1
    assert result.loc[0, "participant_id"] == 302
    assert result.loc[0, "split"] == "train"
    assert result.loc[0, "phq8_score"] == 4
    assert result.loc[0, "transcript"] == "I feel tired."
    assert result.loc[0, "transcript_speaker_filter"] == "participant_turns"


def test_rejects_participant_duplicated_across_official_splits(tmp_path):
    labels_dir = tmp_path / "labels"
    for split in ("train", "dev", "test"):
        records = [{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}] if split in ("train", "dev") else []
        write_csv(labels_dir / f"{split}_split.csv", records)
    with pytest.raises(ValueError, match="more than once"):
        build_dataset(tmp_path / "data", labels_dir)


def test_requires_real_transcript(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    for split in ("train", "dev", "test"):
        records = [{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}] if split == "train" else []
        write_csv(labels_dir / f"{split}_split.csv", records)
    (data_dir / "302_P").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="transcript"):
        build_dataset(data_dir, labels_dir)
