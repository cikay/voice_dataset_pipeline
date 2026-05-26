import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from ..stage import BaseStage

logger = logging.getLogger(__name__)


class StatsStage(BaseStage):
    name = "stats"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_file = Path(config["output_file"])

    def run(self) -> None:
        entries = self._load_metadata(self.input_manifest)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        stats = self._compute(entries)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("Stats saved to %s", self.output_file)
        logger.info(
            "Segments: %d | Duration: %.2fh | Speakers: %d",
            stats["total_segments"],
            stats["total_duration_hours"],
            len(stats["speakers"]),
        )

    def _compute(self, entries: list[dict]) -> dict:
        durations = [e["duration"] for e in entries]
        word_counts = [e["word_count"] for e in entries]
        total_duration = sum(durations)

        by_speaker: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_speaker[e.get("speaker_id", "unknown")].append(e)

        dns_keys = ["mos_sig", "mos_bak", "mos_ovr", "p808_mos"]
        dns_stats = {}
        for key in dns_keys:
            values = [e["dns_mos"][key] for e in entries if "dns_mos" in e]
            if values:
                dns_stats[key] = {
                    "mean": round(statistics.mean(values), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                }

        return {
            "total_segments": len(entries),
            "total_duration_hours": round(total_duration / 3600, 3),
            "total_duration_seconds": round(total_duration, 2),
            "duration": {
                "mean": round(statistics.mean(durations), 2),
                "min": round(min(durations), 2),
                "max": round(max(durations), 2),
                "stdev": (
                    round(statistics.stdev(durations), 2) if len(durations) > 1 else 0
                ),
            },
            "word_count": {
                "mean": round(statistics.mean(word_counts), 2),
                "min": min(word_counts),
                "max": max(word_counts),
            },
            "speakers": {
                speaker_id: {
                    "segments": len(segs),
                    "duration_hours": round(sum(s["duration"] for s in segs) / 3600, 3),
                    "audio_urls": sorted(
                        {s["audio_source_url"] for s in segs if "audio_source_url" in s}
                    ),
                }
                for speaker_id, segs in sorted(by_speaker.items())
            },
            "dns_mos": dns_stats,
        }
