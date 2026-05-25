import json
from pathlib import Path
from typing import Protocol


class PipelineStage(Protocol):
    name: str

    def run(self) -> None:
        ...


class BaseStage:
    def _load_metadata(self, meta_file: Path) -> list[dict]:
        content = meta_file.read_text(encoding="utf-8").strip()
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    def _save_metadata(self, entries: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _symlink(self, src: Path, dst: Path) -> None:
        if not dst.exists():
            dst.symlink_to(src.resolve())
