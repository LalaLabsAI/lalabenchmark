from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from reliable_eval.jsonutil import load_json


SEPARATOR = "=" * 80


def main() -> None:
    parser = argparse.ArgumentParser(description="Print prompt rewrites grouped by benchmark item.")
    parser.add_argument("path", help="Path to variants.json or combined outputs.json")
    parser.add_argument("--show-warnings", action="store_true", help="Print rewrite warnings under each variant")
    parser.add_argument("--show-ideal", action="store_true", help="Print each item's ideal text for context")
    parser.add_argument("--no-original", action="store_true", help="Do not print the original prompt above the rewrites")
    args = parser.parse_args()

    data = load_json(Path(args.path))
    manifest = extract_variants_manifest(data)
    print(
        format_rewrites(
            manifest,
            show_original=not args.no_original,
            show_warnings=args.show_warnings,
            show_ideal=args.show_ideal,
        )
    )


def extract_variants_manifest(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data
    if isinstance(data, dict) and isinstance(data.get("variants"), dict):
        variants = data["variants"]
        if isinstance(variants.get("items"), list):
            return variants
    raise ValueError("Expected a variants.json manifest or outputs.json containing a variants object")


def format_rewrites(
    manifest: dict[str, Any],
    *,
    show_original: bool = True,
    show_warnings: bool = False,
    show_ideal: bool = False,
) -> str:
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("variants manifest must contain an items list")

    lines: list[str] = []
    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {item_index} must be an object")
        item_id = str(item.get("id", item_index))
        prompt = str(item.get("prompt", ""))
        variants = item.get("variants", [])
        if not isinstance(variants, list):
            raise ValueError(f"manifest item {item_id} variants must be a list")

        if lines:
            lines.append("")
        lines.append(SEPARATOR)
        lines.append(f"Item {item_id}")
        keywords = item.get("keywords") or []
        if keywords:
            lines.append("Keywords: " + ", ".join(str(keyword) for keyword in keywords))
        if show_original:
            lines.append("")
            lines.append("Original prompt:")
            lines.extend(_indent_block(prompt))
        if show_ideal and item.get("ideal") is not None:
            lines.append("")
            lines.append("Ideal text:")
            lines.extend(_indent_block(str(item.get("ideal", ""))))

        rewrite_variants = [variant for variant in variants if not _is_original_variant(variant)]
        lines.append("")
        lines.append(f"Prompt rewrites ({len(rewrite_variants)}):")
        if not rewrite_variants:
            lines.append("  (no generated rewrites found)")
            continue

        for rewrite_index, variant in enumerate(rewrite_variants, start=1):
            if not isinstance(variant, dict):
                raise ValueError(f"variant {rewrite_index} for item {item_id} must be an object")
            variant_id = str(variant.get("variant_id", f"{item_id}::rewrite{rewrite_index:03d}"))
            method = str(variant.get("method", "unknown"))
            review_status = str(variant.get("review_status", "unknown"))
            text = str(variant.get("text", ""))
            lines.append("")
            lines.append(f"{rewrite_index}. {variant_id} [{method}; {review_status}]")
            lines.extend(_indent_block(text))
            warnings = variant.get("warnings") or []
            if show_warnings and warnings:
                lines.append("  Warnings: " + ", ".join(str(warning) for warning in warnings))
            rejected = variant.get("rejected_candidate")
            if show_warnings and rejected:
                lines.append("  Rejected candidate:")
                lines.extend(_indent_block(str(rejected), prefix="    "))

    return "\n".join(lines)


def _is_original_variant(variant: Any) -> bool:
    if not isinstance(variant, dict):
        return False
    return str(variant.get("method", "")).strip().lower() == "original"


def _indent_block(text: str, *, prefix: str = "  ") -> list[str]:
    if not text:
        return [prefix.rstrip()]
    return [prefix + line for line in text.splitlines()]


if __name__ == "__main__":
    main()
