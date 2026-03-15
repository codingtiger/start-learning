"""
Prepare Kaggle CIFAR-10 competition data for local training.

Expected raw files from Kaggle:
    train.7z
    test.7z
    trainLabels.csv
    sampleSubmission.csv

Output layout:
    data/cifar-10/
        train/
        test/
        trainLabels.csv
        sampleSubmission.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

import py7zr

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
TARGET_DIR = BASE_DIR / "data" / "cifar-10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and organize Kaggle CIFAR-10 data.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing train.7z/test.7z/trainLabels.csv/sampleSubmission.csv",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=TARGET_DIR,
        help="Directory to place extracted train/test folders and csv files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing extracted data before preparing again",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    return parser.parse_args()


def ensure_source_files(source_dir: Path) -> dict[str, Path]:
    bundle_archive = source_dir / "cifar-10.zip"
    required_files = {
        "train_archive": source_dir / "train.7z",
        "test_archive": source_dir / "test.7z",
        "train_labels": source_dir / "trainLabels.csv",
        "sample_submission": source_dir / "sampleSubmission.csv",
    }
    if bundle_archive.exists():
        required_files["bundle_archive"] = bundle_archive
        return required_files

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Kaggle files:\n" + "\n".join(missing)
        )
    return required_files


def extract_bundle_archive(bundle_path: Path, source_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] extract bundle {bundle_path} -> {source_dir}")
        return

    with ZipFile(bundle_path, mode="r") as archive:
        archive.extractall(path=source_dir)


def reset_target_dir(target_dir: Path, force: bool, dry_run: bool) -> None:
    if not target_dir.exists():
        return

    if not force:
        return

    if dry_run:
        print(f"[dry-run] remove existing directory: {target_dir}")
        return

    shutil.rmtree(target_dir)


def extract_archive(archive_path: Path, output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] extract {archive_path} -> {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=output_dir)


def flatten_nested_dir(output_dir: Path, dry_run: bool) -> None:
    nested_dir = output_dir / output_dir.name
    if not nested_dir.exists() or not nested_dir.is_dir():
        return

    if dry_run:
        print(f"[dry-run] flatten nested directory {nested_dir} -> {output_dir}")
        return

    for item in nested_dir.iterdir():
        shutil.move(str(item), output_dir / item.name)
    nested_dir.rmdir()


def copy_csv(source_path: Path, target_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] copy {source_path} -> {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def main() -> None:
    args = parse_args()
    files = ensure_source_files(args.source_dir)

    if "bundle_archive" in files:
        extract_bundle_archive(files["bundle_archive"], args.source_dir, args.dry_run)
        files = ensure_source_files(args.source_dir)

    if args.target_dir.exists() and not args.force:
        train_dir = args.target_dir / "train"
        test_dir = args.target_dir / "test"
        if train_dir.exists() and test_dir.exists():
            print(f"Target already prepared: {args.target_dir}")
            print("Use --force to rebuild.")
            return

    reset_target_dir(args.target_dir, force=args.force, dry_run=args.dry_run)

    print(f"Source dir: {args.source_dir}")
    print(f"Target dir: {args.target_dir}")

    train_output_dir = args.target_dir / "train"
    test_output_dir = args.target_dir / "test"

    extract_archive(files["train_archive"], train_output_dir, args.dry_run)
    extract_archive(files["test_archive"], test_output_dir, args.dry_run)
    flatten_nested_dir(train_output_dir, args.dry_run)
    flatten_nested_dir(test_output_dir, args.dry_run)
    copy_csv(files["train_labels"], args.target_dir / "trainLabels.csv", args.dry_run)
    copy_csv(
        files["sample_submission"],
        args.target_dir / "sampleSubmission.csv",
        args.dry_run,
    )

    print("CIFAR-10 data prepared successfully.")


if __name__ == "__main__":
    main()
