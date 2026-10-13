"""Restore original input videos and captions under the established showcase names."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent
CASES = {
    "dynamic_0B7ds6NmVBQl_60": ("BiAudio", "0B7ds6NmVBQl_60"),
    "dynamic_I_M0f1c59oc_115": ("BiAudio", "I_M0f1c59oc_115"),
    "dynamic_TKPZf_P5_18_60": ("BiAudio", "TKPZf_P5_18_60"),
    "dynamic_Wc_gzO9i38k_463": ("BiAudio", "Wc_gzO9i38k_463"),
    "dynamic_f2kvR5I8s3c_20": ("BiAudio", "f2kvR5I8s3c_20"),
    "fixed_camera_dynamic_obj_tAiVUt5vE34l_80": ("BiAudio", "tAiVUt5vE34l_80"),
    "fixed_solo1_cello_Lq3671NhNZk_clip_6": ("MUSIC", "solo1_cello_Lq3671NhNZk_clip_6"),
    "invisible_9X2wM6HD_og_313": ("BiAudio", "9X2wM6HD_og_313"),
    "invisible_eE5oue9yeFQ_354": ("BiAudio", "eE5oue9yeFQ_354"),
    "multi_GmtUEtUCX8o_137": ("BiAudio", "GmtUEtUCX8o_137"),
    "multi_PuekW6d_0yEl_80": ("BiAudio", "PuekW6d_0yEl_80"),
    "multi_duet_clarinet_acoustic_guitar_SPXyBDbGt4s_clip_4":
        ("MUSIC", "duet_clarinet_acoustic_guitar_SPXyBDbGt4s_clip_4"),
    "multi_dynamic_530swnPWJrQ_17": ("BiAudio", "530swnPWJrQ_17"),
    "multi_solo2_congas_A22AWJ6zFBs_clip_11": ("MUSIC", "solo2_congas_A22AWJ6zFBs_clip_11"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="Directory containing DATA/ (default: parent of this repository).")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    target = ROOT / "assets/input"
    records = []
    for name, (dataset, original) in CASES.items():
        source = args.source_root / "DATA" / dataset / "video/test" / f"{original}.mp4"
        caption_file = args.source_root / "DATA" / dataset / "caption" / (
            "test_valid.tsv" if dataset == "BiAudio" else "test.tsv")
        with caption_file.open() as stream:
            labels = {row[0]: row[1] for row in csv.reader(stream, delimiter="\t") if len(row) >= 2}
        caption = labels[original].strip()
        if not source.is_file() or not caption:
            raise ValueError(f"Missing original input or caption: {source}")
        records.append({"name": name, "dataset": dataset, "original_id": original,
                        "source_video": source.relative_to(args.source_root).as_posix(),
                        "caption_source": caption_file.relative_to(args.source_root).as_posix(),
                        "caption": caption, "source_sha256": sha256(source)})
    target.mkdir(parents=True, exist_ok=True)
    # Check all destinations before changing any existing files.
    for row in records:
        dest = target / f"{row['name']}.mp4"
        text = dest.with_suffix(".txt")
        if dest.exists() and sha256(dest) != row["source_sha256"]:
            raise ValueError(f"Refusing to overwrite a different input: {dest}")
        if text.exists() and text.read_text().strip() != row["caption"]:
            raise ValueError(f"Refusing to overwrite edited caption: {text}")
    for row in records:
        dest = target / f"{row['name']}.mp4"
        if not dest.exists():
            shutil.copy2(args.source_root / row["source_video"], dest)
        dest.with_suffix(".txt").write_text(row["caption"] + "\n", encoding="utf-8")
        if sha256(dest) != row["source_sha256"]:
            raise ValueError(f"Copy verification failed: {dest}")
    (target / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    old_files = [ROOT / "assets" / f"{name}.mp4" for name in CASES]
    old_files = [path for path in old_files if path.is_file()]
    if old_files:
        backup = ROOT / "output/assets_showcase_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup.mkdir(parents=True)
        for path in old_files:
            shutil.move(str(path), backup / path.name)
        print(f"Original showcase files preserved: {backup}")
    (ROOT / "assets/output").mkdir(exist_ok=True)
    print(f"Prepared and hash-verified {len(records)} video/text pairs: {target}")


if __name__ == "__main__":
    main()
