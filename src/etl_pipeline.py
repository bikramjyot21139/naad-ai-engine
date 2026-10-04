import argparse
import re
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
TEXT_COLUMNS = ("value", "text", "transcript", "utterance", "content")


def participant_id(path: Path):
    match = re.search(r"(?<!\\d)(\\d{3})(?!\\d)", path.name)
    return int(match.group(1)) if match else None


def read_label_splits(labels_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = labels_dir / f"{split}_split.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Official split file is missing: {path}")
        frame = pd.read_csv(path)
        if not {"Participant_ID", "PHQ_Score"}.issubset(frame.columns):
            raise ValueError(f"{path} must include Participant_ID and PHQ_Score")
        frame = frame.copy()
        frame["Participant_ID"] = pd.to_numeric(frame["Participant_ID"], errors="raise").astype(int)
        frame["PHQ_Score"] = pd.to_numeric(frame["PHQ_Score"], errors="raise")
        frame["split"] = split
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    if labels["Participant_ID"].duplicated().any():
        raise ValueError("Participant appears more than once across official splits")
    if not labels["PHQ_Score"].between(0, 24).all():
        raise ValueError("PHQ_Score has a value outside 0–24")
    return labels


def _find_transcript(directory: Path, pid: int) -> Path:
    matches = [p for p in directory.glob("*.csv") if "transcript" in p.name.lower() and participant_id(p) == pid]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one participant transcript for {pid}; found {len(matches)}")
    return matches[0]


def _extract_text(path: Path):
    frame = pd.read_csv(path)
    normalized_columns = {str(c).strip().lower(): c for c in frame.columns}
    text_column = next((normalized_columns[name] for name in TEXT_COLUMNS if name in normalized_columns), None)
    if text_column is None:
        raise ValueError(f"No known text column in {path}: {list(frame.columns)}")

    text = frame[text_column].fillna("").astype(str)
    speaker_column = normalized_columns.get("speaker")
    method = "all_turns"
    if speaker_column is not None:
        speaker = frame[speaker_column].fillna("").astype(str).str.strip().str.lower()
        interviewer = speaker.str.contains(r"ellie|interviewer|wizard|agent", regex=True)
        participant = speaker.str.contains(r"participant|subject|user", regex=True)
        if participant.any():
            text = text.loc[participant]
            method = "participant_turns"
        elif interviewer.any() and (~interviewer).any():
            text = text.loc[~interviewer]
            method = "non_interviewer_turns"

    utterances = [re.sub(r"\\s+", " ", value).strip() for value in text.tolist()]
    return " ".join(value for value in utterances if value), method


def build_dataset(data_dir: Path, labels_dir: Path) -> pd.DataFrame:
    labels = read_label_splits(labels_dir)
    rows = []
    for record in labels.to_dict(orient="records"):
        pid = int(record["Participant_ID"])
        directories = [p for p in data_dir.glob("*_P") if participant_id(p) == pid]
        if len(directories) != 1:
            raise FileNotFoundError(
                f"Expected one extracted participant directory for {pid}; found {len(directories)}. "
                "Extract licensed E-DAIC archives locally before running ETL."
            )
        transcript_path = _find_transcript(directories[0], pid)
        transcript, speaker_filter = _extract_text(transcript_path)
        if not transcript:
            raise ValueError(f"Transcript is empty for participant {pid}")
        rows.append({
            "participant_id": pid,
            "split": record["split"],
            "transcript": transcript,
            "phq8_score": float(record["PHQ_Score"]),
            "phq_binary_official": int(record["PHQ_Binary"]) if pd.notna(record.get("PHQ_Binary")) else None,
            "transcript_speaker_filter": speaker_filter,
            "transcript_file": transcript_path.name,
        })
    result = pd.DataFrame(rows).sort_values(["split", "participant_id"]).reset_index(drop=True)
    if result["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant after transcript/label alignment")
    return result


def run_etl(data_dir: Path, labels_dir: Path, output_path: Path) -> pd.DataFrame:
    result = build_dataset(data_dir, labels_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Aligned {len(result)} participants; split counts: {result['split'].value_counts().to_dict()}")
    print(f"Wrote research metadata to {output_path}. Never commit transcripts or derived data to Git.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Align local E-DAIC transcripts to official participant splits and PHQ labels.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted *_P participant folders")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Official train/dev/test split CSVs")
    parser.add_argument("--output", type=Path, default=Path("data/processed/edaic_text.csv"))
    args = parser.parse_args()
    run_etl(args.data_dir, args.labels_dir, args.output)


if __name__ == "__main__":
    main()
