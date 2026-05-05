#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_IN_KAGGLE_ZIP_NAMES = {"\\", "#", "[", "]", "'", "&"}


def _slugify_component(s: str) -> str:
    s = s.replace("&", " and ")
    s = s.replace("#", " sharp ")
    s = s.replace("'", "")
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("\\", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^A-Za-z0-9._ -]+", "_", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.strip("._- ")
    return s or "file"


def _safe_relpath_for_zip(rel: Path) -> str:
    # Kaggle rejects '\' inside ZIP entry names. ZIP spec expects '/'.
    return rel.as_posix().replace("\\", "/")


@dataclass(frozen=True)
class Item:
    src: Path
    dst: Path
    zip_name: str


def build_plan(input_dir: Path, output_dir: Path) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    warnings: list[str] = []

    used_names: set[str] = set()
    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".mid", ".midi"}:
            continue

        base = _slugify_component(src.stem)
        ext = src.suffix.lower()
        name = f"{base}{ext}"
        if name in used_names:
            i = 2
            while f"{base}__{i}{ext}" in used_names:
                i += 1
            name = f"{base}__{i}{ext}"
        used_names.add(name)

        dst = output_dir / name
        zip_name = _safe_relpath_for_zip(Path(output_dir.name) / name)

        if any(ch in zip_name for ch in FORBIDDEN_IN_KAGGLE_ZIP_NAMES):
            warnings.append(f"still has forbidden char after sanitize: {zip_name!r}")

        items.append(Item(src=src, dst=dst, zip_name=zip_name))

    return items, warnings


def copy_items(items: list[Item], *, dry_run: bool) -> None:
    for it in items:
        if dry_run:
            continue
        it.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it.src, it.dst)


def write_zip(items: list[Item], zip_path: Path, *, output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for it in items:
            # Read from output_dir to ensure ZIP name matches sanitized filename.
            sanitized_file = it.dst
            if not sanitized_file.exists():
                raise FileNotFoundError(sanitized_file)
            zf.write(sanitized_file, arcname=it.zip_name)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Prepare Kaggle-friendly MIDI dataset: sanitize names and create a ZIP without forbidden characters."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/midi_raw"),
        help="Folder with raw MIDIs (recursive). Default: dataset/midi_raw",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/midi_raw_kaggle"),
        help="Output folder (flat, sanitized). Default: dataset/midi_raw_kaggle",
    )
    p.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("midiraw_kaggle.zip"),
        help="Where to write the ZIP. Default: midiraw_kaggle.zip",
    )
    p.add_argument("--no-zip", action="store_true", help="Only create output folder, do not zip.")
    p.add_argument("--dry-run", action="store_true", help="Do not copy or zip, only print stats.")
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output folder before copying (safe: only deletes the output path).",
    )
    args = p.parse_args(argv)

    input_dir: Path = args.input
    output_dir: Path = args.output
    zip_path: Path = args.zip_path

    if not input_dir.exists():
        print(f"ERROR: input folder not found: {str(input_dir)!r}", file=sys.stderr)
        return 2

    if args.clean_output and output_dir.exists() and output_dir.is_dir():
        if args.dry_run:
            pass
        else:
            shutil.rmtree(output_dir)

    items, warnings = build_plan(input_dir, output_dir)
    print(f"Found {len(items)} MIDI files in {input_dir}")
    print(f"Will write sanitized copy to {output_dir} (flat)")
    if not args.no_zip:
        print(f"Will write Kaggle-friendly ZIP to {zip_path}")

    if warnings:
        print("WARNINGS:")
        for w in warnings[:50]:
            print(f"- {w}")
        if len(warnings) > 50:
            print(f"... and {len(warnings) - 50} more")

    if not items:
        print("No .mid/.midi files found. Nothing to do.")
        return 0

    copy_items(items, dry_run=args.dry_run)
    if not args.no_zip:
        write_zip(items, zip_path, output_dir=output_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry-run complete.")
    else:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

