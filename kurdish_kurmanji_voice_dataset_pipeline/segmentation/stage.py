import logging
import re
import sys
import unicodedata
from pathlib import Path

from ..stage import BaseStage

logger = logging.getLogger(__name__)

import soundfile as sf
import torch
import torchaudio
import numpy as np
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_spans,
    postprocess_results,
)
from ctc_forced_aligner.alignment_utils import (
    forced_align,
    merge_repeats,
)


MAX_DURATION = 15.0

SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+")
SENTENCE_LIKE_RE = re.compile(r"[^;:]*[;:]+|[^;:]+")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


class SegmentationStage(BaseStage):
    name = "segmentation"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_manifest = Path(config["output_manifest"])
        self.audio_dir = Path(config["audio_dir"])
        self.language = config.get("align_language", "kmr")
        self.max_duration = float(
            config.get("max_duration_to_split_sentence", MAX_DURATION)
        )
        self.end_padding = float(config.get("end_padding", 0.15))
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.batch_size = int(config.get("batch_size", 16))
        self.window_size = int(config.get("window_size", 30))
        self.context_size = int(config.get("context_size", 2))

    def run(self) -> None:
        if not self.input_manifest.exists():
            logger.error("Missing manifest: %s", self.input_manifest)
            sys.exit(1)

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)

        device, dtype = self._resolve_device()
        entries = self._load_metadata(self.input_manifest)
        logger.info("%d entries to process", len(entries))

        alignment_model, alignment_tokenizer = self._load_alignment_model(device, dtype)

        all_segments = self._segment_entries(
            entries,
            alignment_model,
            alignment_tokenizer,
            self.audio_dir,
            device,
            dtype,
        )

        self._log_run_stats(all_segments)
        self._save_metadata(all_segments, self.output_manifest)
        logger.info("Segmentation complete!")

    def _resolve_device(self) -> tuple[str, torch.dtype]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Using device: %s", device)
        return device, dtype

    def _load_alignment_model(self, device: str, dtype: torch.dtype):
        logger.info("Loading MMS forced alignment model on %s...", device)
        model, tokenizer = load_alignment_model(device, dtype=dtype)
        logger.info("Alignment model loaded")
        return model, tokenizer

    def _segment_entries(
        self,
        entries: list[dict],
        alignment_model,
        alignment_tokenizer,
        segments_dir: Path,
        device: str,
        dtype: torch.dtype,
    ) -> list[dict]:
        all_segments = []

        for idx, entry in enumerate(entries):
            audio_path = Path(entry["audio_file"])
            text_path = Path(entry["text_file"])
            video_id = entry["id"]

            if not audio_path.exists():
                logger.warning("Missing audio: %s", audio_path)
                continue
            if not text_path.exists():
                logger.warning("Missing text: %s", text_path)
                continue

            text = self._read_entry_text(text_path)
            if text is None:
                logger.warning("Text too short for %s", video_id)
                continue

            logger.info("[%d/%d] %s", idx + 1, len(entries), entry['title'][:60])

            try:
                segments = self._segment_entry(
                    alignment_model,
                    alignment_tokenizer,
                    audio_path,
                    text,
                    video_id,
                    segments_dir,
                    device,
                    dtype,
                )
                self._set_parent_fields(segments, entry)
                all_segments.extend(segments)
                logger.info("  -> %d segments", len(segments))
            except Exception as e:
                logger.error("Error processing %s: %s", video_id, e, exc_info=True)

        return all_segments

    def _set_parent_fields(self, segments: list[dict], parent: dict) -> None:
        for seg in segments:
            for field, value in parent.items():
                if field not in seg:
                    seg[field] = value

    def _read_entry_text(self, text_path: Path) -> str | None:
        with open(text_path, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())
        return text if text and len(text) >= 20 else None

    def _segment_entry(
        self,
        alignment_model,
        alignment_tokenizer,
        audio_path: Path,
        text: str,
        output_prefix: str,
        segments_dir: Path,
        device: str,
        dtype: torch.dtype,
    ) -> list[dict]:
        audio_waveform = load_audio(str(audio_path), dtype, device)
        emissions, stride = generate_emissions(
            alignment_model,
            audio_waveform,
            window_length=self.window_size,
            context_length=self.context_size,
            batch_size=self.batch_size,
        )

        sentences, tokens_starred, text_starred = self._prepare_text(text)
        if not tokens_starred:
            logger.warning("No alignable tokens for %s", output_prefix)
            return []

        word_timestamps = self._run_alignment(
            emissions, tokens_starred, alignment_tokenizer, text_starred, stride
        )
        if not word_timestamps:
            logger.warning("No word timestamps for %s", output_prefix)
            return []

        audio_data, sr = sf.read(audio_path)

        chunks = self._map_words_to_sentence_like_chunks(sentences, word_timestamps)

        segments_out = []
        seg_idx = 0
        for chunk_text, start_sec, end_sec, align_score in chunks:
            new_seg, seg_idx = self._build_segment(
                chunk_text,
                start_sec,
                end_sec,
                align_score,
                audio_data,
                sr,
                segments_dir,
                output_prefix,
                seg_idx,
            )
            if new_seg:
                segments_out.append(new_seg)

        return segments_out

    def _prepare_text(self, text: str) -> tuple[list[str], list, str]:
        sentences = split_into_sentences(text)
        if not sentences:
            sentences = [text]
        full_text = " ".join(sentences)
        logger.info("Text split into %d sentences", len(sentences))
        tokens_starred, text_starred = preprocess_text(
            full_text, romanize=True, language=self.language
        )
        return sentences, tokens_starred, text_starred

    def _run_alignment(
        self,
        emissions: torch.Tensor,
        tokens_starred: list,
        alignment_tokenizer,
        text_starred: str,
        stride,
    ) -> list:
        segments, scores, blank_token = self._get_alignments_fixed(
            emissions, tokens_starred, alignment_tokenizer
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def _get_alignments_fixed(self, emissions: torch.Tensor, tokens: list, tokenizer):
        """
        Like ctc_forced_aligner.get_alignments but fixes the <star> token index.

        The HuggingFace tokenizer has extra special tokens that inflate the vocab
        size. generate_emissions adds the star column at the last emissions index,
        so <star> must map there, not to len(vocab).
        """
        assert len(tokens) > 0, "Empty transcript"

        dictionary = tokenizer.get_vocab()
        dictionary = {k.lower(): v for k, v in dictionary.items()}
        dictionary["<star>"] = emissions.shape[-1] - 1

        token_indices = [
            dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
        ]
        blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

        if not emissions.is_cpu:
            emissions = emissions.cpu()
        targets = np.asarray([token_indices], dtype=np.int64)

        path, scores = forced_align(
            emissions.unsqueeze(0).float().numpy(),
            targets,
            blank=blank_id,
        )
        path = path.squeeze().tolist()

        idx_to_token_map = {v: k for k, v in dictionary.items()}
        segments = merge_repeats(path, idx_to_token_map)
        return segments, scores, idx_to_token_map[blank_id]

    def _sentence_time_span(self, word_ts: list) -> tuple[float, float, float]:
        starts = [wt["start"] for wt in word_ts]
        ends = [wt["end"] for wt in word_ts]
        scores = [wt.get("score", 0) for wt in word_ts]
        return min(starts), max(ends), sum(scores) / len(scores)

    def _sub_split_sentence(
        self, sentence: str, sent_word_ts: list
    ) -> list[tuple[str, float, float, float]]:
        sub_chunks = [
            c.strip() for c in SENTENCE_LIKE_RE.findall(sentence) if c.strip()
        ]
        if len(sub_chunks) <= 1:
            return []

        result = []
        sub_word_idx = 0
        for chunk in sub_chunks:
            chunk_end = min(sub_word_idx + len(chunk.split()), len(sent_word_ts))
            chunk_wts = sent_word_ts[sub_word_idx:chunk_end]
            if chunk_wts:
                start, end, score = self._sentence_time_span(chunk_wts)
                result.append((chunk, start, end, score))
            sub_word_idx = chunk_end
        return result

    def _map_words_to_sentence_like_chunks(
        self,
        sentences: list[str],
        word_timestamps: list,
    ) -> list[tuple[str, float, float, float]]:
        result = []
        total_aligned_words = len(word_timestamps)
        if total_aligned_words == 0:
            return result

        word_idx = 0
        for sentence in sentences:
            if word_idx >= total_aligned_words:
                break

            n_words = len(sentence.split())
            end_word_idx = min(word_idx + n_words, total_aligned_words)
            sent_word_ts = word_timestamps[word_idx:end_word_idx]

            if not sent_word_ts:
                word_idx = end_word_idx
                continue

            start_sec, end_sec, align_score = self._sentence_time_span(sent_word_ts)

            if end_sec - start_sec <= self.max_duration:
                result.append((sentence, start_sec, end_sec, align_score))
                word_idx = end_word_idx
                continue

            sub_chunks = self._sub_split_sentence(sentence, sent_word_ts)
            result.extend(
                sub_chunks
                if sub_chunks
                else [(sentence, start_sec, end_sec, align_score)]
            )
            word_idx = end_word_idx

        return result

    def _extract_audio_slice(
        self, audio_data, sr: int, start_sec: float, end_sec: float
    ) -> np.ndarray:
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(audio_data), int((end_sec + self.end_padding) * sr))
        return audio_data[start_sample:end_sample]

    def _build_segment(
        self,
        chunk_text: str,
        start_sec: float,
        end_sec: float,
        align_score: float,
        audio_data,
        sr: int,
        segments_dir: Path,
        output_prefix: str,
        seg_idx: int,
    ) -> tuple[dict | None, int]:
        duration = end_sec - start_sec

        logger.info("Writing segment: '%s' | duration=%.1fs | align_score=%.2f",
                     chunk_text, duration, align_score)

        segment_audio = self._extract_audio_slice(audio_data, sr, start_sec, end_sec)

        if sr != self.sample_rate:
            tensor = torch.from_numpy(segment_audio).float()
            segment_audio = torchaudio.functional.resample(tensor, sr, self.sample_rate).numpy()

        segment_id = f"{output_prefix}_{seg_idx:04d}"
        seg_path = segments_dir / f"{segment_id}.wav"
        sf.write(seg_path, segment_audio, self.sample_rate)

        segment = {
            "id": segment_id,
            "audio_file": str(seg_path),
            "text": chunk_text,
            "duration": round(duration, 2),
            "align_score": round(align_score, 2),
            "word_count": len(chunk_text.split()),
        }
        return segment, seg_idx + 1

    def _log_run_stats(self, all_segments: list[dict]) -> None:
        if not all_segments:
            logger.error("No segments produced. Exiting.")
            sys.exit(1)

        total_dur = sum(s["duration"] for s in all_segments)
        avg_dur = total_dur / len(all_segments)
        avg_score = sum(s["align_score"] for s in all_segments) / len(all_segments)
        avg_words = sum(s["word_count"] for s in all_segments) / len(all_segments)
        logger.info(
            "Total segments: %d\n"
            "Total duration: %.1fh | Avg duration: %.1fs | Avg align score: %.2f | Avg word count: %.1f",
            len(all_segments),
            total_dur / 3600, avg_dur, avg_score, avg_words,
        )
