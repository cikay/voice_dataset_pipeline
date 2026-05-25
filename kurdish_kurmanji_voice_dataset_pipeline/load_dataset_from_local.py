import json
import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_dataset(manifest_file: str) -> Dataset:
    """Load the local segmented dataset.

    The ``audio`` column is kept as a plain file-path string — no HF Audio cast.
    Audio is loaded on demand in the training Dataset class using soundfile,
    which avoids pulling in torchcodec entirely.

    Args:
        manifest_file: Path to the manifest file containing segment metadata.
    """
    segments = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))

    logger.info("📋 %d segments loaded from %s", len(segments), manifest_file)

    return Dataset.from_list(segments)
