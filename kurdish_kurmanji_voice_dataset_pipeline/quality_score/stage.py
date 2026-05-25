import logging
from pathlib import Path

import torch
import torchaudio

from ..stage import BaseStage

logger = logging.getLogger(__name__)
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

def _dns_mos(waveform: torch.Tensor, sr: int, device: str) -> dict:
    scores = deep_noise_suppression_mean_opinion_score(
        preds=waveform.to(device),
        fs=sr,
        personalized=False,
        device=device,
    )
    return {
        "p808_mos": round(float(scores[0]), 2),
        "mos_sig": round(float(scores[1]), 2),
        "mos_bak": round(float(scores[2]), 2),
        "mos_ovr": round(float(scores[3]), 2),
    }


METRIC_REGISTRY = {
    "dns_mos": _dns_mos,
}


class QualityScoreStage(BaseStage):
    name = "quality_score"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_manifest = Path(config["output_manifest"])
        self.metrics = self._build_metrics(config.get("metrics", ["dns_mos"]))

    def run(self) -> None:
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)

        entries = self._load_metadata(self.input_manifest)
        logger.info("%d segments to score", len(entries))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        updated_entries = []

        for i, entry in enumerate(entries, 1):
            waveform, sr = torchaudio.load(entry["audio_file"])
            waveform = waveform.squeeze()

            scores = {}
            for name, fn in self.metrics.items():
                scores[name] = fn(waveform, sr, device)

            updated_entries.append({**entry, **scores})

            if i % 50 == 0:
                logger.info("[%d/%d] scored", i, len(entries))

        self._save_metadata(updated_entries, self.output_manifest)
        logger.info("Quality scoring complete")

    def _build_metrics(self, metric_names: list[str]) -> dict:
        metrics = {}
        for name in metric_names:
            fn = METRIC_REGISTRY.get(name)
            if fn is None:
                raise ValueError(f"Unknown metric: '{name}'. Available: {list(METRIC_REGISTRY)}")
            metrics[name] = fn
        return metrics

