import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from .enhancement.stage import EnhancementStage
from .filter.stage import FilterStage
from .segmentation.stage import SegmentationStage
from .speaker_clustering.stage import SpeakerClusteringStage
from .quality_score.stage import QualityScoreStage
from .stats.stage import StatsStage
from .stage import PipelineStage

logger = logging.getLogger(__name__)

STAGE_REGISTRY: dict[str, type[PipelineStage]] = {
    "enhancement": EnhancementStage,
    "segmentation": SegmentationStage,
    "filter": FilterStage,
    "quality_score": QualityScoreStage,
    "speaker_clustering": SpeakerClusteringStage,
    "stats": StatsStage,
}


def setup_logging(log_file: Path | None = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)


def run_pipeline(config_path: Path = Path("configs/config.yml"), from_stage: str | None = None) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stages_order: list[str] = config.get("stages", [])

    if from_stage:
        if from_stage not in stages_order:
            raise ValueError(f"Unknown stage '{from_stage}'. Available: {stages_order}")
        stages_order = stages_order[stages_order.index(from_stage):]

    full_order = list(STAGE_REGISTRY.keys())
    prev_manifest = None
    for idx, stage_name in enumerate(stages_order, 1):
        stage_config = {**config.get(stage_name, {})}
        if prev_manifest is not None:
            stage_config["input_manifest"] = prev_manifest
        elif stage_name in full_order:
            full_idx = full_order.index(stage_name)
            if full_idx > 0:
                predecessor = full_order[full_idx - 1]
                pred_manifest = config.get(predecessor, {}).get("output_manifest")
                if pred_manifest:
                    stage_config["input_manifest"] = pred_manifest
        stage_cls = STAGE_REGISTRY.get(stage_name)
        if stage_cls is None:
            raise ValueError(f"Unknown stage: '{stage_name}'. Available: {list(STAGE_REGISTRY)}")
        stage = stage_cls(stage_config)
        logger.info("[%d/%d] Running stage: %s", idx, len(stages_order), stage_name)
        stage.run()
        prev_manifest = stage_config.get("output_manifest")

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yml"))
    parser.add_argument("--from", dest="from_stage", type=str, default=None,
                        help="Resume pipeline from this stage (skips all preceding stages)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Also write logs to this file (in addition to stdout)")
    args = parser.parse_args()
    setup_logging(log_file=args.log_file)
    run_pipeline(args.config, from_stage=args.from_stage)
