from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TEXT_EXTENSIONS = {".md", ".txt"}
WORD_CHARS = "0-9A-Za-zА-Яа-яЁё_"


def load_terms(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        terms = json.load(file)

    if not isinstance(terms, dict):
        raise SystemExit(f"Terms map must be a JSON object: {path}")

    invalid = [key for key, value in terms.items() if not isinstance(key, str) or not isinstance(value, str)]
    if invalid:
        raise SystemExit(f"Terms map contains non-string keys or values: {invalid[:5]}")

    return {key: value for key, value in terms.items() if key}


def compile_patterns(terms: dict[str, str]) -> list[tuple[str, str, re.Pattern[str]]]:
    items = sorted(terms.items(), key=lambda item: len(item[0]), reverse=True)
    patterns: list[tuple[str, str, re.Pattern[str]]] = []
    for source, target in items:
        pattern = re.compile(
            rf"(?<![{WORD_CHARS}]){re.escape(source)}(?![{WORD_CHARS}])",
            flags=re.IGNORECASE,
        )
        patterns.append((source, target, pattern))
    return patterns


def replace_text(
    text: str,
    patterns: list[tuple[str, str, re.Pattern[str]]],
) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    result = text

    for source, target, pattern in patterns:
        result, count = pattern.subn(target, result)
        if count:
            counts[source] += count

    return result, counts


def iter_pages(raw_pages_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_pages_dir.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
    )


def process_file(
    path: Path,
    output_path: Path,
    patterns: list[tuple[str, str, re.Pattern[str]]],
    dry_run: bool,
) -> Counter[str]:
    original = path.read_text(encoding="utf-8-sig")
    replaced, counts = replace_text(original, patterns)

    if counts and not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(replaced, encoding="utf-8", newline="\n")
    elif not dry_run and output_path != path and not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(original, encoding="utf-8", newline="\n")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace recognizable source-universe terms in raw pages."
    )
    parser.add_argument(
        "raw_pages_dir",
        nargs="?",
        default="raw_pages",
        type=Path,
        help="Directory with cleaned raw pages. Defaults to raw_pages.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("knowledge_base"),
        type=Path,
        help="Directory for replaced pages. Defaults to knowledge_base.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite files in raw_pages_dir instead of writing to output-dir.",
    )
    parser.add_argument(
        "--terms-map",
        default=Path("terms_map.json"),
        type=Path,
        help="Path to JSON object with source term -> replacement term.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show replacement counts without writing files.",
    )
    args = parser.parse_args()

    if not args.raw_pages_dir.exists() or not args.raw_pages_dir.is_dir():
        raise SystemExit(f"Directory not found: {args.raw_pages_dir}")
    if not args.terms_map.exists():
        raise SystemExit(f"Terms map not found: {args.terms_map}")

    terms = load_terms(args.terms_map)
    patterns = compile_patterns(terms)
    pages = iter_pages(args.raw_pages_dir)

    if not pages:
        raise SystemExit(f"No text pages found in {args.raw_pages_dir}")

    total_counts: Counter[str] = Counter()
    changed_files = 0

    for path in pages:
        output_path = path if args.in_place else args.output_dir / path.name
        counts = process_file(path, output_path, patterns, dry_run=args.dry_run)
        replacements = sum(counts.values())
        if replacements:
            changed_files += 1
            total_counts.update(counts)
        action = "would replace" if args.dry_run else "replaced"
        destination = path if args.in_place else output_path
        print(f"{action}: {path} -> {destination} ({replacements} replacements)")

    print(
        f"Processed {len(pages)} files, changed {changed_files}, "
        f"total replacements {sum(total_counts.values())}."
    )

    if total_counts:
        print("Top replacements:")
        for source, count in total_counts.most_common(20):
            print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
