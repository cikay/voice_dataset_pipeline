import logging
import operator as op
from functools import lru_cache
from pathlib import Path

import numpy as np
from df.enhance import enhance, load_audio

logger = logging.getLogger(__name__)

OPS = {
    "lt": op.lt,
    "lte": op.le,
    "gt": op.gt,
    "gte": op.ge,
    "eq": op.eq,
}


@lru_cache(maxsize=1)
def _load_model():
    from df.enhance import init_df
    return init_df()


class DeepFilterNetTool:
    CHUNK_SECONDS = 600  # 10 minutes

    def __init__(self, config: dict) -> None:
        self.run_if = config.get("run_if", {})
        self.replace_if = config.get("replace_if", {})

    def should_run(self, metrics: dict) -> bool:
        return all(
            self._check_condition(key, threshold, metrics)
            for key, threshold in self.run_if.items()
        )

    def should_replace(self, original_metrics: dict, enhanced_metrics: dict) -> bool:
        if not self.replace_if:
            return True
        return all(
            self._check_damage_condition(key, threshold, original_metrics, enhanced_metrics)
            for key, threshold in self.replace_if.items()
        )

    def enhance(self, input_path: Path) -> tuple[np.ndarray, int]:
        model, df_state, _ = _load_model()
        audio, _ = load_audio(str(input_path), sr=df_state.sr())

        chunk_samples = self.CHUNK_SECONDS * df_state.sr()
        total_samples = audio.shape[-1]

        if total_samples <= chunk_samples:
            enhanced = enhance(model, df_state, audio.contiguous())
            return np.array(enhanced).squeeze(), df_state.sr()

        n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info(
            "  🔪 Audio too long (%.1f min) — processing in %d chunks",
            total_samples / df_state.sr() / 60,
            n_chunks,
        )
        chunks = []
        for i, start in enumerate(range(0, total_samples, chunk_samples), 1):
            logger.info("     chunk [%d/%d]...", i, n_chunks)
            chunk = audio[..., start:start + chunk_samples].contiguous()
            enhanced_chunk = enhance(model, df_state, chunk)
            chunks.append(np.array(enhanced_chunk).squeeze())

        return np.concatenate(chunks), df_state.sr()

    def _check_condition(self, key: str, threshold, metrics: dict) -> bool:
        *path_parts, operator_name = key.split("__")
        value = metrics
        for part in path_parts:
            value = value[part]
        return OPS[operator_name](value, threshold)

    def _check_damage_condition(
        self, key: str, threshold, original: dict, enhanced: dict
    ) -> bool:
        # key format: (enhance|damage)__<metric_path...>__<operator>
        parts = key.split("__")
        prefix = parts[0]
        metric_path = parts[1:-1]
        operator_name = parts[-1]

        orig_value = original
        enh_value = enhanced
        for part in metric_path:
            orig_value = orig_value[part]
            enh_value = enh_value[part]

        change = (enh_value - orig_value) if prefix == "enhance" else (orig_value - enh_value)
        return OPS[operator_name](change, threshold)
