from __future__ import annotations

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path


TEXT_EXTENSIONS = {".html", ".htm", ".md", ".txt"}
SKIP_LINES = {
    "Войдите, чтобы сохранить",
    "Править",
    "Содержание",
}


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }

    IGNORED_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"</?(?:html|body|div|p|span|table|script|style|h[1-6])\b", text, re.I))


def cyrillic_score(text: str) -> int:
    cyrillic = sum("А" <= char <= "я" or char == "ё" or char == "Ё" for char in text)
    mojibake_markers = text.count("Р") + text.count("СЃ") * 3 + text.count("вЂ") * 3
    return cyrillic - mojibake_markers


def repair_mojibake(text: str) -> str:
    try:
        repaired = text.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return text

    return repaired if cyrillic_score(repaired) > cyrillic_score(text) else text


def strip_html(text: str) -> str:
    if not looks_like_html(text):
        return html.unescape(text)

    parser = TextExtractor()
    parser.feed(text)
    parser.close()
    return html.unescape(parser.text())


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\((?:[^()]|\([^)]*\))*\)", r"\1", text)
    text = re.sub(r"\[\d+]", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~]{1,3}", "", text)
    text = re.sub(r"^\s*>+\s?", "", text, flags=re.MULTILINE)
    return text


def is_toc_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)*\s+.+", line))


def is_noise_line(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    if normalized in SKIP_LINES:
        return True
    if is_toc_line(normalized):
        return True
    if re.fullmatch(r"\d+px[-_ ].+", normalized, re.I):
        return True
    if re.fullmatch(r"(?:Left|Right) pointing .+", normalized, re.I):
        return True
    return False


def normalize_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines: list[str] = []
    previous_non_empty = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if is_noise_line(line):
            continue
        if line and line == previous_non_empty:
            continue
        cleaned_lines.append(line)
        if line:
            previous_non_empty = line

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"


def clean_text(text: str) -> str:
    text = repair_mojibake(text)
    text = strip_html(text)
    text = strip_markdown(text)
    return normalize_lines(text)


def clean_file(path: Path, dry_run: bool) -> tuple[int, int]:
    original = path.read_text(encoding="utf-8-sig")
    cleaned = clean_text(original)

    if not dry_run and cleaned != original:
        path.write_text(cleaned, encoding="utf-8", newline="\n")

    return len(original), len(cleaned)


def iter_pages(raw_pages_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_pages_dir.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw wiki pages and overwrite them with plain UTF-8 text."
    )
    parser.add_argument(
        "raw_pages_dir",
        nargs="?",
        default="raw_pages",
        type=Path,
        help="Directory with raw pages. Defaults to raw_pages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing files.",
    )
    args = parser.parse_args()

    raw_pages_dir = args.raw_pages_dir
    if not raw_pages_dir.exists() or not raw_pages_dir.is_dir():
        raise SystemExit(f"Directory not found: {raw_pages_dir}")

    pages = iter_pages(raw_pages_dir)
    if not pages:
        raise SystemExit(f"No text pages found in {raw_pages_dir}")

    changed = 0
    for path in pages:
        before, after = clean_file(path, dry_run=args.dry_run)
        file_changed = before != after
        if file_changed:
            changed += 1
        if args.dry_run:
            action = "would clean" if file_changed else "unchanged"
        else:
            action = "cleaned" if file_changed else "unchanged"
        print(f"{action}: {path} ({before} -> {after} chars)")

    print(f"Processed {len(pages)} files, changed {changed}.")


if __name__ == "__main__":
    main()
