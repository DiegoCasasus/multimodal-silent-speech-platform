from pathlib import Path
import argparse
import collections
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    args = parser.parse_args()

    root = Path(args.processed_root)
    files = sorted(root.rglob("*_preprocessed.npz"))

    participants = collections.Counter()
    labels = collections.Counter()
    shapes = collections.Counter()
    skipped = []

    for path in files:
        try:
            data = np.load(path, allow_pickle=True)

            if "participant_id" in data.files:
                participant_id = str(data["participant_id"])
            else:
                participant_id = "unknown"
            participants[participant_id] += 1

            if "labels" in data.files:
                labels.update([str(x) for x in data["labels"].tolist()])

            for modality in ["eeg", "emg", "imu"]:
                if modality in data.files:
                    shapes[(modality, tuple(data[modality].shape[1:]))] += 1

            data.close()

        except Exception as exc:
            skipped.append((str(path), str(exc)))

    print("=" * 80)
    print("Processed dataset audit")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Processed NPZ files: {len(files)}")

    print("\nParticipants:")
    for key, value in participants.items():
        print(f"  {key}: {value}")

    print("\nLabels:")
    for key, value in labels.items():
        print(f"  {key}: {value}")

    print("\nShapes:")
    for key, value in shapes.items():
        print(f"  {key}: {value}")

    if skipped:
        print("\nSkipped / unreadable files:")
        for path, error in skipped[:20]:
            print(f"  {path}: {error}")
    else:
        print("\nSkipped / unreadable files: none")


if __name__ == "__main__":
    main()