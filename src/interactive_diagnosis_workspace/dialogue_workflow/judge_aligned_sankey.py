from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


STATE_ORDER = ["Rank 1", "Rank 2", "Rank 3", "Rank 4+", "Absent", "No turn"]
STATE_COLORS = {
    "Rank 1": "#2ca02c",
    "Rank 2": "#8cc84b",
    "Rank 3": "#ffc107",
    "Rank 4+": "#ff7f0e",
    "Absent": "#a6a6a6",
    "No turn": "#d8d8d8",
}


@dataclass(frozen=True)
class JudgeTerm:
    text: str
    level: str
    reference_id: str
    explanation: str


@dataclass(frozen=True)
class JudgeAlignment:
    case_id: str
    model_text: str
    terms: tuple[JudgeTerm, ...]
    outcome: str
    judge_path: str


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def rank_state(rank: int | None) -> str:
    if rank is None:
        return "Absent"
    if rank <= 3:
        return f"Rank {rank}"
    return "Rank 4+"


def non_final_ddx_turns(episode: dict[str, Any]) -> list[tuple[int, list[Any]]]:
    out: list[tuple[int, list[Any]]] = []
    for turn in episode.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        clinician = turn.get("clinician") if isinstance(turn.get("clinician"), dict) else {}
        if str(clinician.get("action") or "").upper() == "FINAL":
            continue
        ddx = clinician.get("current_ddx")
        if isinstance(ddx, list) and ddx:
            out.append((int(turn.get("turn_number") or len(out) + 1), ddx))
    return out


def load_judge_alignments(judge_dirs: list[Path], include_partial: bool) -> dict[str, JudgeAlignment]:
    alignments: dict[str, JudgeAlignment] = {}
    for judge_dir in judge_dirs:
        cases_dir = judge_dir / "cases" if judge_dir.name != "cases" else judge_dir
        for path in sorted(cases_dir.glob("*.final_diagnosis_judge.json")):
            data = read_json(path)
            evaluation = data.get("final_diagnosis_evaluation") or {}
            case_id = str(data.get("case_name") or evaluation.get("case_name") or path.name.split(".")[0])
            model_items = {
                str(item.get("id")): str(item.get("text") or "")
                for item in (evaluation.get("model_extraction") or {}).get("items", [])
                if isinstance(item, dict) and item.get("id") and item.get("text")
            }
            matching = evaluation.get("matching_result") or {}
            terms: list[JudgeTerm] = []
            seen: set[tuple[str, str]] = set()
            for level, key in [("full", "full_matches"), ("partial", "partial_matches")]:
                if level == "partial" and not include_partial:
                    continue
                for match in matching.get(key, []) or []:
                    if not isinstance(match, dict):
                        continue
                    model_id = str(match.get("model_id") or "")
                    text = model_items.get(model_id, "")
                    if not text:
                        continue
                    dedupe_key = (normalize_text(text), level)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    terms.append(
                        JudgeTerm(
                            text=text,
                            level=level,
                            reference_id=str(match.get("reference_id") or ""),
                            explanation=str(match.get("explanation") or ""),
                        )
                    )
            alignments[case_id] = JudgeAlignment(
                case_id=case_id,
                model_text=str(evaluation.get("model_text") or ""),
                terms=tuple(terms),
                outcome="accepted" if terms else "no_match",
                judge_path=str(path),
            )
    return alignments


def load_ddx_alignment_cases(alignment_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for alignment_dir in alignment_dirs:
        cases_dir = alignment_dir / "cases" if alignment_dir.name != "cases" else alignment_dir
        for path in sorted(cases_dir.glob("*.ddx_alignment_judge.json")):
            data = read_json(path)
            case_id = str(data.get("case_id") or path.name.split(".")[0])
            data["_alignment_path"] = str(path)
            cases[case_id] = data
    return cases


def best_ddx_alignment_from_cache(
    ddx: list[Any],
    ddx_case: dict[str, Any],
    include_partial: bool,
) -> tuple[int | None, str, str, str]:
    allowed = {"full", "partial"} if include_partial else {"full"}
    item_by_text = {
        normalize_text(str(item.get("text") or "")): str(item.get("id") or "")
        for item in ddx_case.get("ddx_items", []) or []
        if isinstance(item, dict)
    }
    alignments = ddx_case.get("item_alignments") if isinstance(ddx_case.get("item_alignments"), dict) else {}
    for rank, diagnosis in enumerate(ddx, start=1):
        model_id = item_by_text.get(normalize_text(str(diagnosis)))
        alignment = alignments.get(model_id, {}) if isinstance(alignments.get(model_id), dict) else {}
        level = str(alignment.get("alignment") or "none")
        if level in allowed:
            return rank, str(diagnosis), level, str(alignment.get("explanation") or "")
    return None, "", "none", ""


def collect_rows(
    run_dirs: list[Path],
    alignments: dict[str, JudgeAlignment],
    window: int,
    drop_short: bool,
    ddx_alignment_cases: dict[str, dict[str, Any]],
    include_partial: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    checkpoints = [f"T-{index}" for index in range(window, 0, -1)]
    for run_dir in run_dirs:
        for episode_path in sorted((run_dir / "episodes").glob("*.json")):
            case_id = episode_path.stem
            if case_id in seen_cases or case_id not in alignments or case_id not in ddx_alignment_cases:
                continue
            episode = read_json(episode_path)
            turns = non_final_ddx_turns(episode)
            if drop_short and len(turns) < window:
                continue
            selected = turns[-window:]
            if len(selected) < window:
                selected = [(-1, []) for _ in range(window - len(selected))] + selected
            alignment = alignments[case_id]
            ddx_case = ddx_alignment_cases[case_id]
            for checkpoint, (turn_number, ddx) in zip(checkpoints, selected):
                if turn_number < 0:
                    rank, matched_ddx, level, explanation = None, "", "none", ""
                else:
                    rank, matched_ddx, level, explanation = best_ddx_alignment_from_cache(ddx, ddx_case, include_partial)
                rows.append(
                    {
                        "case_id": case_id,
                        "checkpoint": checkpoint,
                        "turn_number": "" if turn_number < 0 else turn_number,
                        "state": "No turn" if turn_number < 0 else rank_state(rank),
                        "rank": "" if rank is None else rank,
                        "alignment_level": level,
                        "judge_outcome": alignment.outcome,
                        "matched_ddx": matched_ddx,
                        "judge_model_text": alignment.model_text,
                        "judge_terms": " | ".join(term.text for term in alignment.terms),
                        "match_explanation": explanation,
                        "episode_path": str(episode_path),
                        "judge_path": alignment.judge_path,
                        "ddx_alignment_path": str(ddx_case.get("_alignment_path", "")),
                    }
                )
            seen_cases.add(case_id)
    return rows


def build_links(rows: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    checkpoint_order = {f"T-{index}": offset for offset, index in enumerate(range(window, 0, -1))}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    cases_by_link: dict[tuple[str, str, str, str], list[str]] = {}
    for case_id, case_rows in by_case.items():
        ordered = sorted(case_rows, key=lambda row: checkpoint_order[str(row["checkpoint"])])
        for left, right in zip(ordered, ordered[1:]):
            key = (
                str(left["checkpoint"]),
                str(left["state"]),
                str(right["checkpoint"]),
                str(right["state"]),
            )
            counts[key] += 1
            cases_by_link.setdefault(key, []).append(case_id)
    return [
        {
            "source_checkpoint": key[0],
            "source_state": key[1],
            "target_checkpoint": key[2],
            "target_state": key[3],
            "count": value,
            "case_ids": cases_by_link[key],
        }
        for key, value in sorted(counts.items())
    ]


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "checkpoint",
        "turn_number",
        "state",
        "rank",
        "alignment_level",
        "judge_outcome",
        "matched_ddx",
        "judge_model_text",
        "judge_terms",
        "match_explanation",
        "episode_path",
        "judge_path",
        "ddx_alignment_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_state_label(state: str) -> str:
    if state == "Absent":
        return "Not listed"
    if state == "Rank 4+":
        return "Rank ≥ 4"
    return state


def case_label(count: int) -> str:
    return "1 case" if count == 1 else f"{count} cases"


def draw_ribbon(
    ax: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    color: str,
    alpha: float = 0.30,
) -> None:
    curve = 0.35 * (x1 - x0)
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
            edgecolor="#ffffff",
            linewidth=0.25,
            alpha=alpha,
            zorder=1,
        )
    )


def checkpoint_label(checkpoint: str) -> str:
    if checkpoint == "T-1":
        return "Last DDx\nbefore final\n(T-1)"
    index = checkpoint.removeprefix("T-")
    suffix = {"2": "2nd", "3": "3rd", "4": "4th"}.get(index, f"{index}th")
    return f"{suffix} DDx\nbefore final\n({checkpoint})"


def sankey_png(
    rows: list[dict[str, Any]],
    links: list[dict[str, Any]],
    window: int,
    output_png: Path,
    output_pdf: Path | None = None,
) -> None:
    checkpoints = [f"T-{index}" for index in range(window, 0, -1)]
    node_counts: Counter[tuple[str, str]] = Counter((str(row["checkpoint"]), str(row["state"])) for row in rows)
    case_outcomes = {
        str(row["case_id"]): str(row["judge_outcome"])
        for row in rows
        if row["checkpoint"] == "T-1"
    }
    case_count = len(case_outcomes)
    final_counts = Counter(case_outcomes.values())
    scale = 0.16
    gap = 0.06
    stack_top = 2.35
    stack_order = ["Rank 1", "Rank 2", "Rank 3", "Rank 4+", "Absent", "No turn"]
    final_order = ["accepted", "no_match"]
    rect_width = 0.68
    final_width = 1.08
    x_step = 1.32
    x_positions = {checkpoint: idx * x_step for idx, checkpoint in enumerate(checkpoints)}
    final_x = len(checkpoints) * x_step + 0.28
    final_colors = {"accepted": "#1f77b4", "no_match": "#6f6f6f"}
    final_labels = {"accepted": "Accepted final diagnosis", "no_match": "No-match final answer"}

    def build_stack(items: list[tuple[str, int]]) -> dict[str, dict[str, float]]:
        cursor = stack_top
        layout: dict[str, dict[str, float]] = {}
        for key, count in items:
            if count <= 0:
                continue
            height = count * scale
            top = cursor
            bottom = top - height
            layout[key] = {
                "top": top,
                "bottom": bottom,
                "center": (top + bottom) / 2,
                "height": height,
            }
            cursor = bottom - gap
        return layout

    node_layouts = {
        checkpoint: build_stack([(state, node_counts[(checkpoint, state)]) for state in stack_order])
        for checkpoint in checkpoints
    }
    final_layout = build_stack([(outcome, final_counts[outcome]) for outcome in final_order])
    stack_bottom = min(
        [node["bottom"] for layout in node_layouts.values() for node in layout.values()]
        + [node["bottom"] for node in final_layout.values()],
        default=0.0,
    )

    fig, ax = plt.subplots(figsize=(19, 6.9), dpi=300)
    fig.subplots_adjust(top=0.82, bottom=0.075, left=0.025, right=0.985)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.suptitle(
        f"Final-Diagnosis Rank Trajectory Across the Clinician’s Last Five DDx Updates (N = {case_count})",
        fontsize=12.5,
        weight="bold",
        y=0.96,
    )
    legend_items = [("Yellow = Rank 3", STATE_COLORS["Rank 3"]), ("Orange = Rank >= 4", STATE_COLORS["Rank 4+"])]
    legend_start_x = 0.415
    legend_y = 0.905
    for index, (label, color) in enumerate(legend_items):
        x = legend_start_x + index * 0.095
        fig.add_artist(
            Rectangle(
                (x, legend_y - 0.008),
                0.012,
                0.014,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="#555555",
                linewidth=0.4,
            )
        )
        fig.text(x + 0.016, legend_y, label, ha="left", va="center", fontsize=8.5, color="#333333")

    source_offsets: Counter[tuple[str, str]] = Counter()
    target_offsets: Counter[tuple[str, str]] = Counter()
    ordered_links = sorted(
        links,
        key=lambda item: (
            checkpoints.index(str(item["source_checkpoint"])),
            STATE_ORDER.index(str(item["source_state"])),
            STATE_ORDER.index(str(item["target_state"])),
        ),
    )
    for link in ordered_links:
        source_key = (str(link["source_checkpoint"]), str(link["source_state"]))
        target_key = (str(link["target_checkpoint"]), str(link["target_state"]))
        count = int(link["count"])
        width = count * scale
        source_node = node_layouts[source_key[0]][source_key[1]]
        target_node = node_layouts[target_key[0]][target_key[1]]
        source_y = (
            source_node["top"]
            - source_offsets[source_key] * scale
            - width / 2
        )
        target_y = (
            target_node["top"]
            - target_offsets[target_key] * scale
            - width / 2
        )
        draw_ribbon(
            ax,
            x_positions[source_key[0]] + rect_width / 2,
            source_y,
            x_positions[target_key[0]] - rect_width / 2,
            target_y,
            width,
            STATE_COLORS[source_key[1]],
            alpha=0.50,
        )
        source_offsets[source_key] += count
        target_offsets[target_key] += count

    final_target_offsets: Counter[str] = Counter()
    t_minus_1_counts = Counter(
        (str(row["state"]), str(row["judge_outcome"])) for row in rows if row["checkpoint"] == "T-1"
    )
    for outcome in ["accepted", "no_match"]:
        for state in [state for state in STATE_ORDER if t_minus_1_counts[(state, outcome)]]:
            count = t_minus_1_counts[(state, outcome)]
            width = count * scale
            source_key = ("T-1", state)
            source_node = node_layouts["T-1"][state]
            final_node = final_layout[outcome]
            source_y = (
                source_node["top"]
                - source_offsets[source_key] * scale
                - width / 2
            )
            target_y = (
                final_node["top"]
                - final_target_offsets[outcome] * scale
                - width / 2
            )
            draw_ribbon(
                ax,
                x_positions["T-1"] + rect_width / 2,
                source_y,
                final_x - final_width / 2,
                target_y,
                width,
                STATE_COLORS[state],
                alpha=0.60,
            )
            source_offsets[source_key] += count
            final_target_offsets[outcome] += count

    for checkpoint in checkpoints:
        x = x_positions[checkpoint]
        ax.text(
            x,
            stack_top + 0.22,
            checkpoint_label(checkpoint),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            linespacing=1.15,
        )
        for state in stack_order:
            count = node_counts[(checkpoint, state)]
            if count <= 0:
                continue
            node = node_layouts[checkpoint][state]
            height = node["height"]
            y = node["center"]
            ax.add_patch(
                Rectangle(
                    (x - rect_width / 2, y - height / 2),
                    rect_width,
                    height,
                    facecolor=STATE_COLORS[state],
                    edgecolor="#3f3f3f",
                    linewidth=0.8,
                    alpha=0.95,
                    zorder=3,
                )
            )
            if height < 1.05:
                continue
            else:
                ax.text(
                    x,
                    y,
                    f"{display_state_label(state)}\n{case_label(count)}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if state in {"Rank 1", "Absent"} else "#222222",
                    zorder=4,
            )

    ax.text(
        final_x,
        stack_top + 0.22,
        "Final-answer\njudge outcome",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
        linespacing=1.15,
    )
    for outcome in ["accepted", "no_match"]:
        count = final_counts[outcome]
        if count <= 0:
            continue
        node = final_layout[outcome]
        final_total = node["height"]
        y = node["center"]
        ax.add_patch(
            Rectangle(
                (final_x - final_width / 2, y - final_total / 2),
                final_width,
                final_total,
                facecolor=final_colors[outcome],
                edgecolor="#3f3f3f",
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )
        )
        ax.text(
            final_x,
            y,
            f"{final_labels[outcome]}\n{case_label(count)}",
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
            zorder=4,
        )

    ax.set_xlim(-0.55, final_x + final_width / 2 + 0.25)
    ax.set_ylim(stack_bottom - 0.18, stack_top + 0.56)
    ax.axis("off")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a static Sankey diagram for final diagnoses accepted by LLM-as-judge."
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="Setting 1 run directory.")
    parser.add_argument(
        "--judge-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing final_diagnosis/cases or cases/*.final_diagnosis_judge.json.",
    )
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--window", type=int, default=4, help="Number of non-final DDX turns before FINAL to include.")
    parser.add_argument("--strict", action="store_true", help="Use only full judge matches; default includes partial matches.")
    parser.add_argument(
        "--ddx-alignment-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing cases/*.ddx_alignment_judge.json for per-DDX rank matching.",
    )
    parser.add_argument("--drop-short", action="store_true", help="Drop cases with fewer than --window non-final DDX turns.")
    args = parser.parse_args()

    if args.window < 2:
        raise ValueError("--window must be at least 2")

    alignments = load_judge_alignments(args.judge_dir, include_partial=not args.strict)
    ddx_alignment_cases = load_ddx_alignment_cases(args.ddx_alignment_dir)
    rows = collect_rows(
        args.run_dir,
        alignments,
        args.window,
        args.drop_short,
        ddx_alignment_cases=ddx_alignment_cases,
        include_partial=not args.strict,
    )
    links = build_links(rows, args.window)
    stem = args.output_png.with_suffix("")
    write_rows_csv(stem.with_name(stem.name + "_rank_states.csv"), rows)
    stem.with_name(stem.name + "_links.json").write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    output_pdf = args.output_pdf or args.output_png.with_suffix(".pdf")
    sankey_png(rows, links, args.window, args.output_png, output_pdf)
    print(args.output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
