#!/usr/bin/env python3
"""Migrate legacy LibriSpeech manifests to the current wav2vec training schema."""

import argparse
import json
import os
import shutil
from pathlib import Path


SPLIT_FILENAMES = {
    "train": "train.json",
    "dev_clean": "dev_clean.json",
    "dev_other": "dev_other.json",
    "test_clean": "test_clean.json",
    "test_other": "test_other.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=Path("data/all/manifests"),
        help="Directory containing train/dev/test JSONL manifests and speaker_id_mapping.json.",
    )
    parser.add_argument(
        "--phys_cache_root",
        type=Path,
        default=None,
        help="Root for prosody physics cache paths. Defaults to <manifest_dir>/phys_cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_dir = args.manifest_dir
    phys_cache_root = args.phys_cache_root or manifest_dir / "phys_cache"

    with open(manifest_dir / "speaker_id_mapping.json", encoding="utf-8") as f:
        spk_map = json.load(f)
    spk2idx = {int(k): int(v) for k, v in spk_map["spk2idx"].items()}
    print(f"loaded spk2idx: {len(spk2idx)} speakers")

    for split_name, filename in SPLIT_FILENAMES.items():
        in_path = manifest_dir / filename
        bak_path = manifest_dir / f"{filename}.bak"
        tmp_path = manifest_dir / f"{filename}.new"
        if not in_path.exists():
            print(f"skip {filename}: missing")
            continue

        count = 0
        with open(in_path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin):
                old = json.loads(line)
                manifest_id = line_idx + 1
                spk_id = int(old.get("speaker", old.get("spk_id", -1)))
                new = {
                    "manifest_id": int(manifest_id),
                    "index": int(line_idx),
                    "utterance_key": f"{split_name}-{manifest_id:09d}",
                    "audio_filepath": old["audio_filepath"],
                    "duration": float(old["duration"]),
                    "text": old["text"],
                    "full_id": str(old.get("id", old.get("full_id", ""))),
                    "spk_id": spk_id,
                    "spk_idx": int(spk2idx.get(spk_id, -1)),
                    "prosody_physics_filepath": str(phys_cache_root / split_name / f"{manifest_id}.npy"),
                }
                fout.write(json.dumps(new, ensure_ascii=False) + "\n")
                count += 1

        if not bak_path.exists():
            shutil.copy(in_path, bak_path)
        os.replace(tmp_path, in_path)
        print(f"{split_name}: {count} entries migrated (backup: {bak_path.name})")


if __name__ == "__main__":
    main()
