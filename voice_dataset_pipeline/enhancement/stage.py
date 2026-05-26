import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

from .tools.deepfilternet import DeepFilterNetTool
from ..stage import BaseStage

logger = logging.getLogger(__name__)


TOOL_REGISTRY = {
    "deepfilternet": DeepFilterNetTool,
}


class EnhancementStage(BaseStage):
    name = "enhancement"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_manifest = Path(config["output_manifest"])
        self.audio_dir = Path(config["audio_dir"])
        self.tools = self._build_tools(config.get("tools", {}))

    def run(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)

        entries = self._load_metadata(self.input_manifest)
        manifest_dir = self.input_manifest.parent
        for entry in entries:
            for key in ("audio_file", "text_file"):
                if key in entry:
                    p = Path(entry[key])
                    if not p.is_absolute():
                        entry[key] = str(manifest_dir / p)
        updated_entries = []
        enhanced_count = 0
        skipped_count = 0

        for i, entry in enumerate(entries, 1):
            audio_path = Path(entry["audio_file"])
            out_audio_path = self.audio_dir / audio_path.name

            if not audio_path.exists():
                logger.warning("Missing audio: %s", audio_path)
                continue

            logger.info("[%d/%d] %s", i, len(entries), entry['title'][:60])

            enhanced = self._apply_tools(audio_path, out_audio_path)
            if enhanced:
                enhanced_count += 1
            else:
                self._symlink(audio_path, out_audio_path)
                skipped_count += 1

            updated_entries.append({**entry, "audio_file": str(out_audio_path), "enhanced": enhanced})

        self._save_metadata(updated_entries, self.output_manifest)
        logger.info("Enhanced: %d | Skipped (already clean): %d", enhanced_count, skipped_count)

    def _build_tools(self, tools_config: dict) -> list:
        tools = []
        for name, tool_config in tools_config.items():
            cls = TOOL_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown enhancement tool: '{name}'. Available: {list(TOOL_REGISTRY)}")
            tools.append(cls(tool_config or {}))
        return tools

    def _apply_tools(self, input_path: Path, output_path: Path) -> bool:
        original_audio, original_sr = sf.read(str(input_path))
        original_metrics = self._compute_dns_mos(original_audio, original_sr)
        applied_any = False

        for tool in self.tools:
            if not tool.should_run(original_metrics):
                logger.info("Skipping %s (run_if not met)", type(tool).__name__)
                self._log_original_metrics(input_path.name, original_metrics)
                continue

            logger.info("Applying %s...", type(tool).__name__)
            enhanced_audio, enhanced_sr = tool.enhance(input_path)
            enhanced_metrics = self._compute_dns_mos(enhanced_audio, enhanced_sr)
            self._log_metrics(input_path.name, original_metrics, enhanced_metrics)

            if tool.should_replace(original_metrics, enhanced_metrics):
                logger.info("Keeping enhanced audio for %s", input_path.name)
                sf.write(str(output_path), enhanced_audio, enhanced_sr)
                applied_any = True
            else:
                logger.warning("Reverting %s — enhancement caused damage", input_path.name)

        return applied_any

    def _log_original_metrics(self, filename: str, metrics: dict) -> None:
        m = metrics["dns_mos"]
        rows = "\n".join(f"  {key:<8} {m[key]:>8.2f}" for key in ("bak", "sig", "ovr", "p808"))
        logger.info("DNS-MOS %s\n  %s\n%s", filename, f"{'metric':<8} {'value':>8}", rows)

    def _log_metrics(self, filename: str, original: dict, enhanced: dict) -> None:
        orig = original["dns_mos"]
        enh = enhanced["dns_mos"]
        rows = []
        for key in ("bak", "sig", "ovr", "p808"):
            change = enh[key] - orig[key]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            rows.append(f"  {key:<8} {orig[key]:>8.2f} {enh[key]:>8.2f} {change:>+8.2f} {arrow}")
        header = f"  {'metric':<8} {'original':>8} {'enhanced':>8} {'change':>8}"
        logger.info("DNS-MOS %s\n%s\n%s", filename, header, "\n".join(rows))

    def _compute_dns_mos(self, audio: np.ndarray, sample_rate: int) -> dict:
        waveform = torch.from_numpy(audio).float()
        scores = deep_noise_suppression_mean_opinion_score(
            preds=waveform, fs=sample_rate, personalized=False,
        )
        return {
            "dns_mos": {
                "p808": round(float(scores[0]), 2),
                "sig": round(float(scores[1]), 2),
                "bak": round(float(scores[2]), 2),
                "ovr": round(float(scores[3]), 2),
            }
        }

