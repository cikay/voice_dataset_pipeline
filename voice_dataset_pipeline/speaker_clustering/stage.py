from collections import Counter, defaultdict
from functools import lru_cache
import logging
from pathlib import Path

from ..stage import BaseStage

logger = logging.getLogger(__name__)

import numpy as np
from sklearn.cluster import AgglomerativeClustering


@lru_cache(maxsize=1)
def _load_model():
    import torch
    import wespeaker

    speaker = wespeaker.load_model("english")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker.set_device(device)
    speaker.model.eval()
    speaker.model.register_forward_pre_hook(
        lambda _, inputs: tuple(
            x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
        )
    )
    return speaker, device


class SpeakerClusteringStage(BaseStage):
    name = "speaker_clustering"

    def __init__(self, config: dict) -> None:
        self.input_manifest = Path(config["input_manifest"])
        self.output_manifest = Path(config["output_manifest"])
        self.threshold = float(config.get("threshold", 0.75))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("speaker_clustering.threshold must be between 0.0 and 1.0")
        self.id_format = config.get("id_format", "speaker_{n:03d}")

    def run(self) -> None:
        out_meta_file = self.output_manifest
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)

        entries = self._load_metadata(self.input_manifest)
        logger.info("%d segments to process", len(entries))

        groups = self._group_by_audio_id(entries)
        logger.info("%d audio files to cluster", len(groups))

        speaker, device = _load_model()
        logger.info("Embedding model loaded on %s", device)

        groups_list = list(groups.items())
        groups_list_len = len(groups_list)
        audio_ids = []
        audio_centroids = []

        for i, (audio_id, seg_list) in enumerate(groups_list, 1):
            logger.info("[%d/%d] Extracting centroid for %s (%d segments)", i, groups_list_len, audio_id, len(seg_list))
            embeddings = self._extract_embeddings(seg_list, speaker)
            audio_ids.append(audio_id)
            audio_centroids.append(self._normalize(embeddings.mean(axis=0)))

        audio_to_speaker = self._cluster_audio_centroids(audio_ids, np.array(audio_centroids))

        speaker_audio_map: dict[str, list[str]] = defaultdict(list)
        updated_segments = []
        for audio_id, seg_list in groups_list:
            speaker_id = audio_to_speaker[audio_id]
            speaker_audio_map[speaker_id].append(audio_id)
            logger.info("  %s -> %s", audio_id, speaker_id)

            for seg in seg_list:
                updated_segments.append({**seg, "speaker_id": speaker_id})

        mapping_lines = []
        for speaker_id, speaker_audio_ids in sorted(speaker_audio_map.items()):
            mapping_lines.append(f"  {speaker_id}: " + ", ".join(speaker_audio_ids))
        logger.info("Speaker -> audio mapping:\n%s", "\n".join(mapping_lines))

        self._save_metadata(updated_segments, out_meta_file)
        self._log_stats(updated_segments)

    def _group_by_audio_id(self, entries: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            groups[entry["audio_id"]].append(entry)
        return groups

    def _extract_embeddings(self, entries: list[dict], model) -> np.ndarray:
        embeddings = [model.extract_embedding(entry["audio_file"]) for entry in entries]
        return np.array(embeddings)

    def _cluster_audio_centroids(self, audio_ids: list[str], centroids: np.ndarray) -> dict[str, str]:
        if not audio_ids:
            return {}
        if len(audio_ids) == 1:
            return {audio_ids[0]: self.id_format.format(n=1)}

        distance_threshold = 1.0 - self.threshold
        logger.info(
            "Clustering %d audio centroids with agglomerative clustering "
            "(cosine similarity >= %.3f, distance <= %.3f)",
            len(audio_ids),
            self.threshold,
            distance_threshold,
        )
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(centroids)
        label_to_speaker = {
            label: self.id_format.format(n=i + 1)
            for i, label in enumerate(dict.fromkeys(labels))
        }
        return {
            audio_id: label_to_speaker[label]
            for audio_id, label in zip(audio_ids, labels)
        }

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _log_stats(self, entries: list[dict]) -> None:
        counts = Counter(e.get("speaker_id") for e in entries)
        lines = []
        for speaker_id, count in sorted(counts.items()):
            total_dur = sum(e["duration"] for e in entries if e.get("speaker_id") == speaker_id)
            lines.append(f"  {speaker_id}: {count} segments, {total_dur / 60:.1f} min")
        logger.info("Speaker distribution:\n%s", "\n".join(lines))
