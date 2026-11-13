from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp"}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "")).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def node_text(node: ET.Element | None) -> str:
    return normalize_space(" ".join(node.itertext())) if node is not None else ""


def first_descendant(node: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in node.iter() if local_name(item.tag) == name), None)


def find_nxmls(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for case_dir in input_dir.iterdir()
        if case_dir.is_dir()
        for path in case_dir.iterdir()
        if path.suffix.lower() in {".nxml", ".xml"}
    )


def match_image_paths(case_dir: Path, graphic_href: str) -> list[str]:
    href_name = Path(graphic_href).name
    href_stem = Path(href_name).stem.lower()
    candidates = []
    for path in sorted(case_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem.lower()
        if stem == href_stem or href_stem in stem or stem in href_stem:
            candidates.append(str(path.resolve()))
    # PMC folders often contain both GIF and JPG copies. Prefer one JPG/PNG copy.
    preferred = [p for p in candidates if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return (preferred or candidates)[:1]


def parse_nxml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    article_title = node_text(first_descendant(root, "article-title"))
    pmcid = ""
    for node in root.iter():
        if local_name(node.tag) == "article-id" and node.attrib.get("pub-id-type") == "pmc":
            pmcid = node_text(node)
            break
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    figures: list[dict[str, Any]] = []
    for figure in (node for node in root.iter() if local_name(node.tag) == "fig"):
        graphic = first_descendant(figure, "graphic")
        href = ""
        if graphic is not None:
            href = next((v for k, v in graphic.attrib.items() if local_name(k) == "href"), "")
        figures.append(
            {
                "figure_id": figure.attrib.get("id", ""),
                "label": node_text(first_descendant(figure, "label")),
                "caption": node_text(first_descendant(figure, "caption")),
                "graphic_href": href,
                "image_paths": match_image_paths(path.parent, href) if href else [],
            }
        )

    sections: list[dict[str, Any]] = []
    body = first_descendant(root, "body")
    if body is not None:
        for section in (node for node in body.iter() if local_name(node.tag) == "sec"):
            title = node_text(first_descendant(section, "title"))
            paragraphs = [node_text(p) for p in section if local_name(p.tag) == "p"]
            paragraphs = [p for p in paragraphs if p]
            if paragraphs:
                sections.append({"title": title, "paragraphs": paragraphs})

    return {
        "pmcid": pmcid or path.parent.name,
        "article_title": article_title,
        "nxml_path": str(path.resolve()),
        "image_folder": str(path.parent.resolve()),
        "figures": figures,
        "body_sections": sections,
    }

