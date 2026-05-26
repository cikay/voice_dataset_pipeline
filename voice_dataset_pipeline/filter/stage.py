from collections import Counter
import logging
import re
import sys
from pathlib import Path

from ..stage import BaseStage

logger = logging.getLogger(__name__)


MIN_DURATION = 2.0
MAX_DURATION = 15.0
MIN_WORDS = 3
MIN_SCORE = -7.0

ABBR_RE = re.compile(
    r"(?:"
    r"\b(?:(?:[A-ZÇĞÎÛŞ]\.){2,}|[A-ZÇĞÎÛŞ][a-zçğıîûş]{1,3}\.)(?=\s*[A-Za-zÇĞÎÛŞçğıîûş])"
    r"|"
    r"\b[A-ZÇĞÎÛŞ]{2,}[a-zçğêıîûş]*\b"
    r")"
)
DIGIT_RE = re.compile(r"\d")


class FilterStage(BaseStage):
    name = "filter"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_manifest = Path(config["output_manifest"])
        self.min_duration = float(config.get("min_duration", MIN_DURATION))
        self.max_duration = float(config.get("max_duration", MAX_DURATION))
        self.min_words = int(config.get("min_words", MIN_WORDS))
        self.min_align_score = float(config.get("min_align_score", MIN_SCORE))
        self.exclude_abbr = bool(config.get("exclude_abbr", True))
        self.exclude_number = bool(config.get("exclude_number", True))

    def run(self) -> None:
        if not self.input_manifest.exists():
            logger.error("Missing manifest: %s", self.input_manifest)
            sys.exit(1)

        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)

        entries = self._load_metadata(self.input_manifest)
        logger.info("%d segments to filter", len(entries))

        kept_entries, discard_counts = self._filter_entries(entries)
        self._save_metadata(kept_entries, self.output_manifest)
        self._log_run_stats(len(entries), kept_entries, discard_counts)

    def _filter_entries(self, entries: list[dict]) -> tuple[list[dict], Counter]:
        kept_entries = []
        discard_counts: Counter = Counter()

        for entry in entries:
            should_discard, reason = self._should_discard(entry["text"], entry["duration"], entry["align_score"])
            if should_discard:
                discard_counts[reason] += 1
                logger.info(
                    "Discarding segment: '%s' | reason=%s",
                    entry.get("text", ""),
                    reason,
                )
                continue
            kept_entries.append(entry)

        return kept_entries, discard_counts

    def _should_discard(
        self, sentence: str, duration: float, align_score: float
    ) -> tuple[bool, str]:
        if duration < self.min_duration or duration > self.max_duration:
            return True, "duration"
        if len(sentence.split()) < self.min_words:
            return True, "too_few_words"
        if align_score < self.min_align_score:
            return True, "low_score"
        if self.exclude_abbr and bool(ABBR_RE.search(sentence)):
            return True, "abbreviations"
        if self.exclude_number and bool(DIGIT_RE.search(sentence)):
            return True, "digits"
        return False, ""

    def _log_run_stats(
        self, total_entries: int, kept_entries: list[dict], discard_counts: Counter
    ) -> None:
        if not kept_entries:
            logger.error("No segments left after filtering. Exiting.")
            sys.exit(1)

        total_discarded = sum(discard_counts.values())
        discard_lines = "".join(
            f"\n   - {reason}: {count}" for reason, count in sorted(discard_counts.items())
        )
        total_dur = sum(s["duration"] for s in kept_entries)
        avg_dur = total_dur / len(kept_entries)
        avg_score = sum(s["align_score"] for s in kept_entries) / len(kept_entries)
        avg_words = sum(s["word_count"] for s in kept_entries) / len(kept_entries)
        logger.info(
            "Input segments: %d | Kept: %d | Discarded: %d%s\n"
            "Total duration: %.1fh | Avg duration: %.1fs | Avg align score: %.2f | Avg word count: %.1f",
            total_entries,
            len(kept_entries),
            total_discarded,
            discard_lines,
            total_dur / 3600,
            avg_dur,
            avg_score,
            avg_words,
        )
