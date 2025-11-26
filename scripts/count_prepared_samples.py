#!/usr/bin/env python3
"""Count tampered (fake) and real samples per dataset and split.

Scans a `prepared` root for `manifest.parquet` files (one per prepared
dataset) and an optional `combined_manifest.parquet` and prints a concise
breakdown of counts by split and label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


MANIFEST_FILENAME = "manifest.parquet"
COMBINED_MANIFEST = "combined_manifest.parquet"


def load_manifest(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to read manifest {path}: {exc}", file=sys.stderr)
        return None


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # Ensure required columns exist
    if "split" not in df.columns or "label" not in df.columns:
        raise ValueError("Manifest missing required 'split' or 'label' columns")
    table = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    # Add totals column
    table["_total"] = table.sum(axis=1)
    return table


def process_prepared_root(prepared_root: Path) -> None:
    if not prepared_root.exists():
        print(f"Prepared root '{prepared_root}' does not exist.")
        return

    # First, check for a combined manifest at the prepared root
    combined_path = prepared_root / COMBINED_MANIFEST
    if combined_path.exists():
        df = load_manifest(combined_path)
        if df is not None:
            print(f"== Combined manifest: {combined_path} ==")
            table = summarize_df(df)
            if table.empty:
                print("(combined manifest is empty)\n")
            else:
                print(table.to_string())
                print()

    # Next, scan immediate subdirectories for per-dataset manifests
    for child in sorted(prepared_root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.exists():
            # skip directories without a manifest.parquet
            continue
        df = load_manifest(manifest_path)
        dataset_name = child.name
        print(f"== Dataset: {dataset_name} (manifest: {manifest_path}) ==")
        if df is None:
            print("(failed to load manifest)\n")
            continue
        if df.empty:
            print("(manifest is empty)\n")
            continue
        try:
            table = summarize_df(df)
        except ValueError as exc:
            print(f"Invalid manifest columns: {exc}\n")
            continue
        print(table.to_string())
        # Also print overall totals per label
        totals = df["label"].value_counts().to_dict()
        print(f"Totals: {totals}\n")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Count prepared samples per dataset/split")
    p.add_argument("--prepared-root", default="prepared", help="Path to prepared root (default: prepared)")
    args = p.parse_args(argv)

    prepared_root = Path(args.prepared_root)
    process_prepared_root(prepared_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
