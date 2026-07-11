from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .jsonutil import load_json, read_label_map


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    prompt: str
    ideal: str
    keywords: tuple[str, ...]
    source: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] | None = None

    def to_manifest_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "ideal": self.ideal,
            "keywords": list(self.keywords),
        }
        if self.source is not None:
            record["source"] = self.source
        if self.created_at is not None:
            record["created_at"] = self.created_at
        return record


def load_benchmark(
    path: str | Path,
    *,
    label_map_path: str | Path | None = None,
) -> list[BenchmarkItem]:
    label_map = read_label_map(label_map_path)
    data = load_json(path)
    rows = _extract_rows(data)
    items: list[BenchmarkItem] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark row {index} must be an object")
        item_id = str(row.get("id", row.get("question_id", index))).strip()
        prompt = _first_text(row, ["prompt", "question", "input", "text"])
        ideal = _first_text(row, ["ideal", "ideal_response", "reference", "answer", "gold"])
        if not prompt:
            raise ValueError(f"Benchmark row {item_id} is missing a prompt/question field")
        if not ideal:
            raise ValueError(f"Benchmark row {item_id} is missing an ideal/reference field")
        keywords = tuple(_normalize_keywords(row.get("keywords", row.get("metadata", [])), label_map))
        items.append(
            BenchmarkItem(
                id=item_id,
                prompt=prompt,
                ideal=ideal,
                keywords=keywords,
                source=_optional_str(row.get("source")),
                created_at=_optional_str(row.get("created_at")),
                raw=row,
            )
        )
    return items


def _extract_rows(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("Benchmark JSON must be an object or list")
    for key in ("submissions", "items", "examples", "questions", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(
        "Benchmark JSON object must contain one of: submissions, items, examples, questions, data"
    )


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_keywords(raw: Any, label_map: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if isinstance(raw, dict):
        def sort_key(key: Any) -> tuple[int, str]:
            text = str(key)
            return (0, f"{int(text):08d}") if text.isdigit() else (1, text)

        values = [raw[key] for key in sorted(raw.keys(), key=sort_key)]
    elif isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = raw.split(",")
    elif raw is None:
        values = []
    else:
        values = [raw]

    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if not label:
            continue
        label = label_map.get(label, label)
        if not label:
            continue
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels

