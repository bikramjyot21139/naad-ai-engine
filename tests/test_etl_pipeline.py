from pathlib import Path

import pandas as pd
import pytest

from src.etl_pipeline import build_dataset


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Participant_ID", "PHQ_Binary", "PHQ_Score"] if path.name.endswith("_split.csv") else ["speaker", "value"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def write_labels(labels_dir: Path, train_rows=None, dev_rows=None, test_rows=None):
    for split, rows in (("train", train_rows or []), ("dev", dev_rows or []), ("test", test_rows or [])):
        write_csv(labels_dir / f"{split}_split.csv", rows)


def test_aligns_participant_text_to_official_split_and_target(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    write_labels(labels_dir, train_rows=[{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}])
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
    row = {"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}
    write_labels(labels_dir, train_rows=[row], dev_rows=[row])
    with pytest.raises(ValueError, match="more than once"):
        build_dataset(tmp_path / "data", labels_dir)


def test_requires_real_transcript(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    write_labels(labels_dir, train_rows=[{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}])
    (data_dir / "302_P").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="transcript"):
        build_dataset(data_dir, labels_dir)


def test_refuses_to_include_interviewer_when_speaker_mapping_is_unknown(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    write_labels(labels_dir, train_rows=[{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}])
    folder = data_dir / "302_P"
    write_csv(folder / "302_Transcript.csv", [
        {"speaker": "Ellie", "value": "Interviewer speech"},
        {"speaker": "Speaker 2", "value": "Participant response"},
    ])
    with pytest.raises(ValueError, match="Could not identify participant turns"):
        build_dataset(data_dir, labels_dir)


def test_requires_speaker_column_to_avoid_mixed_transcripts(tmp_path):
    data_dir, labels_dir = tmp_path / "data", tmp_path / "labels"
    write_labels(labels_dir, train_rows=[{"Participant_ID": 302, "PHQ_Binary": 0, "PHQ_Score": 4}])
    folder = data_dir / "302_P"
    path = folder / "302_Transcript.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"value": "Unattributed speech"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No speaker column"):
        build_dataset(data_dir, labels_dir)
