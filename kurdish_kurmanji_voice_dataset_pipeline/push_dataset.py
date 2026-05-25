#!/usr/bin/env python3
"""
Push the Kurdish Kurmanji voice dataset to HuggingFace Hub.

Usage:
    python -m kurdish_kurmanji_voice_dataset_pipeline.push_dataset --repo muzaffercky/azadiya-welat-kurdish-kurmanji-voice
    python -m kurdish_kurmanji_voice_dataset_pipeline.push_dataset --repo ... --manifest dataset/speaker_clustering_manifest.json
    python -m kurdish_kurmanji_voice_dataset_pipeline.push_dataset --repo ... --private
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from datasets import Audio, Dataset
from dotenv import load_dotenv, dotenv_values, find_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from kurdish_kurmanji_voice_dataset_pipeline.load_dataset_from_local import load_dataset

load_dotenv()

DEFAULT_MANIFEST = "dataset/speaker_clustering_manifest.json"

COLUMNS = {
    "id": "id",
    "text": "text",
    "audio_file": "audio",
    "duration": "duration",
    "speaker_id": "speaker_id",
    "align_score": "align_score",
    "word_count": "word_count",
    "text_source_url": "text_source_url",
    "audio_source_url": "audio_source_url",
    "dns_mos": "dns_mos",
    "enhanced": "is_enhanced",
}

logger = logging.getLogger(__name__)


def _check_repo_does_not_exist(repo_id: str, token: str) -> None:
    api = HfApi()
    try:
        api.dataset_info(repo_id=repo_id, token=token)
        raise RuntimeError(
            f"Dataset '{repo_id}' already exists on HuggingFace. "
            "Delete it first or choose a different name."
        )
    except RepositoryNotFoundError:
        pass


def _build_dataset(manifest: Path) -> Dataset:
    raw = load_dataset(str(manifest))

    missing = [src for src in COLUMNS if src not in raw.column_names]
    if missing:
        raise ValueError(f"Manifest is missing expected columns: {missing}")

    ds = raw.select_columns(list(COLUMNS.keys()))
    ds = ds.rename_columns(COLUMNS)
    ds = ds.cast_column("audio", Audio())
    return ds


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. username/dataset-name)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
        help="Path to manifest JSON file",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create as private dataset"
    )
    args = parser.parse_args()

    token = os.environ.get("MY_HF_TOKEN")
    if not token:
        logger.error("MY_HF_TOKEN env variable not set")
        sys.exit(1)

    logger.info("Checking if '%s' already exists...", args.repo)
    _check_repo_does_not_exist(args.repo, token)

    logger.info("Loading manifest from %s", args.manifest)
    ds = _build_dataset(args.manifest)
    logger.info("%d segments ready to push", len(ds))
    logger.info("Columns: %s", ds.column_names)

    logger.info("Pushing to %s (private=%s)...", args.repo, args.private)
    ds.push_to_hub(args.repo, token=token, private=args.private)
    logger.info("Done! https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
