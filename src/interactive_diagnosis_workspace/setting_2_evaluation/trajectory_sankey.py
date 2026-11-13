from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


CATEGORY_ORDER = [
    "initial_presentation",
    "clinical_history",
    "physical_examination",
    "lab_tests",
    "microbiology",
    "imaging",
    "pathology",
    "procedure_or_intervention",
    "medication_or_treatment",
    "admission_or_clinical_course",
    "other_or_off_protocol",
    "end",
]

CATEGORY_COLORS = {
    "initial_presentation": "#4C78A8",
    "clinical_history": "#72B7B2",
    "physical_examination": "#54A24B",
    "lab_tests": "#E45756",
    "microbiology": "#59A14F",
    "imaging": "#4992C2",
    "pathology": "#B279A2",
    "procedure_or_intervention": "#F58518",
    "medication_or_treatment": "#FF9DA6",
    "admission_or_clinical_course": "#9D755D",
    "other_or_off_protocol": "#BAB0AC",
    "end": "#D9D9D9",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_label(category: str) -> str:
    labels = {
        "initial_presentation": "Initial",
        "clinical_history": "History",
        "physical_examination": "Exam",
        "lab_tests": "Labs",
        "microbiology": "Microbiology",
        "imaging": "Imaging",
        "pathology": "Pathology",
        "procedure_or_intervention": "Procedure",
        "medication_or_treatment": "Treatment",
        "admission_or_clinical_course": "Course",
        "other_or_off_protocol": "Other",
        "end": "End",
    }
    return labels.get(category, category.replace("_", " ").title())


def collect_detail_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "case_details.json":
            out.append(path)
        elif path.is_dir():
            if (path / "case_details.json").is_file():
                out.append(path / "case_details.json")
            else:
                out.extend(sorted(path.glob("*/case_details.json")))
    return sorted(dict.fromkeys(out))


def load_sequences(detail_paths: list[Path], sequence_key: str) -> dict[str, list[str]]:
    sequences: dict[str, list[str]] = {}
    for path in detail_paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        specialty = path.parent.name
        for case_id, detail in payload.items():
            if not isinstance(detail, dict) or detail.get("status") != "completed":
                continue
            sequence = detail.get(sequence_key)
            if not isinstance(sequence, list) or not sequence:
                continue
            namespaced_id = f"{specialty}/{case_id}"
            sequences[namespaced_id] = [str(item) for item in sequence]
    return sequences


def truncate_sequence(sequence: list[str], max_steps: int, include_end: bool) -> list[str]:
    truncated = sequence[:max_steps]
    if include_end and len(truncated) < max_steps + 1:
        truncated = truncated + ["end"]
    return truncated


def build_counts(
    sequences: dict[str, list[str]],
    max_steps: int,
    include_end: bool,
) -> tuple[Counter[tuple[int, str]], Counter[tuple[int, str, int, str]], dict[tuple[int, str, int, str], list[str]]]:
    node_counts: Counter[tuple[int, str]] = Counter()
    link_counts: Counter[tuple[int, str, int, str]] = Counter()
    link_cases: dict[tuple[int, str, int, str], list[str]] = defaultdict(list)
    for case_id, sequence in sequences.items():
        steps = truncate_sequence(sequence, max_steps=max_steps, include_end=include_end)
        for index, category in enumerate(steps):
            node_counts[(index, category)] += 1
        for index, (source, target) in enumerate(zip(steps, steps[1:])):
            key = (index, source, index + 1, target)
            link_counts[key] += 1
            link_cases[key].append(case_id)
    return node_counts, link_counts, link_cases


def draw_ribbon(
    ax: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    color: str,
    alpha: float,
) -> None:
    curve = 0.38 * (x1 - x0)
    verts = [
        (x0, y0 + width / 2),
        (x0 + curve, y0 + width / 2),
        (x1 - curve, y1 + width / 2),
        (x1, y1 + width / 2),
        (x1, y1 - width / 2),
        (x1 - curve, y1 - width / 2),
        (x0 + curve, y0 - width / 2),
        (x0, y0 - width / 2),
        (x0, y0 + width / 2),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=alpha,
            zorder=1,
        )
    )


def stack_layout(
    node_counts: Counter[tuple[int, str]],
    step_index: int,
    scale: float,
    gap: float,
    top: float,
) -> dict[str, dict[str, float]]:
    categories = [cat for cat in CATEGORY_ORDER if node_counts[(step_index, cat)] > 0]
    categories.extend(
        sorted(
            {
                cat
                for (idx, cat), count in node_counts.items()
                if idx == step_index and count > 0 and cat not in CATEGORY_ORDER
            }
        )
    )
    cursor = top
    layout: dict[str, dict[str, float]] = {}
    for category in categories:
        count = node_counts[(step_index, category)]
        height = count * scale
        bottom = cursor - height
        layout[category] = {"top": cursor, "bottom": bottom, "center": (cursor + bottom) / 2, "height": height}
        cursor = bottom - gap
    return layout


def write_links_csv(
    path: Path,
    link_counts: Counter[tuple[int, str, int, str]],
    link_cases: dict[tuple[int, str, int, str], list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_step", "source_category", "target_step", "target_category", "count", "case_ids"],
        )
        writer.writeheader()
        for key, count in sorted(link_counts.items()):
            writer.writerow(
                {
                    "source_step": key[0],
                    "source_category": key[1],
                    "target_step": key[2],
                    "target_category": key[3],
                    "count": count,
                    "case_ids": " ".join(link_cases[key]),
                }
            )


def draw_sankey(
    *,
    sequences: dict[str, list[str]],
    mode: str,
    max_steps: int,
    include_end: bool,
    min_link_count: int,
    output_png: Path,
    output_pdf: Path | None,
    output_links: Path | None,
) -> None:
    node_counts, link_counts, link_cases = build_counts(sequences, max_steps=max_steps, include_end=include_end)
    if min_link_count > 1:
        link_counts = Counter({key: value for key, value in link_counts.items() if value >= min_link_count})
        link_cases = {key: link_cases[key] for key in link_counts}

    if output_links:
        write_links_csv(output_links, link_counts, link_cases)

    step_count = max((idx for idx, _ in node_counts), default=0) + 1
    scale = 0.14
    gap = 0.035
    top = 2.7
    rect_width = 0.68
    layouts = {idx: stack_layout(node_counts, idx, scale=scale, gap=gap, top=top) for idx in range(step_count)}
    bottoms = [node["bottom"] for layout in layouts.values() for node in layout.values()]

    fig_width = max(11, 1.7 * step_count)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=300)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.88, bottom=0.08)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    title_mode = "Gold source-derived" if mode == "gold" else "Predicted Setting 2"
    fig.suptitle(f"{title_mode} diagnostic trajectory category flow (N = {len(sequences)})", fontsize=14, weight="bold")

    source_offsets: Counter[tuple[int, str]] = Counter()
    target_offsets: Counter[tuple[int, str]] = Counter()
    ordered_links = sorted(
        link_counts.items(),
        key=lambda item: (
            item[0][0],
            CATEGORY_ORDER.index(item[0][1]) if item[0][1] in CATEGORY_ORDER else 999,
            CATEGORY_ORDER.index(item[0][3]) if item[0][3] in CATEGORY_ORDER else 999,
        ),
    )
    for (source_step, source, target_step, target), count in ordered_links:
        if source not in layouts[source_step] or target not in layouts[target_step]:
            continue
        width = count * scale
        source_node = layouts[source_step][source]
        target_node = layouts[target_step][target]
        source_y = source_node["top"] - source_offsets[(source_step, source)] * scale - width / 2
        target_y = target_node["top"] - target_offsets[(target_step, target)] * scale - width / 2
        draw_ribbon(
            ax,
            source_step + rect_width / 2,
            source_y,
            target_step - rect_width / 2,
            target_y,
            width,
            CATEGORY_COLORS.get(source, "#9E9E9E"),
            alpha=0.48,
        )
        source_offsets[(source_step, source)] += count
        target_offsets[(target_step, target)] += count

    for step_index in range(step_count):
        ax.text(step_index, top + 0.15, f"Step {step_index}", ha="center", va="bottom", fontsize=11, weight="bold")
        for category, node in layouts[step_index].items():
            height = node["height"]
            y = node["center"]
            count = node_counts[(step_index, category)]
            color = CATEGORY_COLORS.get(category, "#9E9E9E")
            ax.add_patch(
                Rectangle(
                    (step_index - rect_width / 2, y - height / 2),
                    rect_width,
                    height,
                    facecolor=color,
                    edgecolor="#3F3F3F",
                    linewidth=0.65,
                    alpha=0.96,
                    zorder=3,
                )
            )
            if height >= 0.24:
                ax.text(
                    step_index,
                    y,
                    f"{display_label(category)}\n{count}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if category in {"initial_presentation", "imaging", "lab_tests"} else "#202020",
                    zorder=4,
                )

    ax.set_xlim(-0.65, step_count - 0.35)
    ax.set_ylim(min(bottoms, default=0) - 0.18, top + 0.42)
    ax.axis("off")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Setting 2 trajectory category Sankey diagrams.")
    parser.add_argument("--detail-path", action="append", type=Path, default=[], help="case_details.json, specialty output dir, or evaluation root.")
    parser.add_argument("--mode", choices=["gold", "pred"], default="pred")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--no-end", action="store_true", help="Do not append an End node after shorter trajectories.")
    parser.add_argument("--min-link-count", type=int, default=1)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--output-links", type=Path, default=None)
    args = parser.parse_args()

    detail_paths = collect_detail_paths(args.detail_path)
    if not detail_paths:
        raise SystemExit("No case_details.json files found.")
    sequence_key = "gold_sequence" if args.mode == "gold" else "prediction_sequence"
    sequences = load_sequences(detail_paths, sequence_key)
    if not sequences:
        raise SystemExit(f"No completed {sequence_key} sequences found.")
    draw_sankey(
        sequences=sequences,
        mode=args.mode,
        max_steps=args.max_steps,
        include_end=not args.no_end,
        min_link_count=args.min_link_count,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        output_links=args.output_links,
    )
    print(f"Wrote {args.output_png} from {len(sequences)} cases.")


if __name__ == "__main__":
    main()
