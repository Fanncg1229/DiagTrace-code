from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normalize_diagnosis(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace("‐", "-").replace("–", "-").replace("—", "-")
    value = value.replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"\s+", " ", value)
    return value


def canonical_diagnosis(value: str) -> str:
    value = normalize_diagnosis(value)
    aliases = {
        "acute febrile neutrophilic dermatosis (Sweet's syndrome)": "Sweet's syndrome",
        "acute febrile neutrophilic dermatosis (Sweet’s syndrome)": "Sweet's syndrome",
    }
    return aliases.get(value, value)


def display_label(value: str, max_chars: int = 48) -> str:
    label = canonical_diagnosis(value)
    replacements = {
        "acute febrile neutrophilic dermatosis (Sweet's syndrome)": "Sweet's syndrome",
        "acute febrile neutrophilic dermatosis (Sweet’s syndrome)": "Sweet's syndrome",
        "acute myelogenous leukemia (AML)": "AML",
        "myelodysplastic syndrome": "MDS",
        "leukocytoclastic vasculitis": "LCV",
        "drug-induced skin reaction": "Drug-induced reaction",
        "infective endocarditis": "Infective endocarditis",
        "VEXAS syndrome": "VEXAS syndrome",
    }
    label = replacements.get(label, label)
    if len(label) > max_chars:
        return label[: max_chars - 1] + "…"
    return label


def collect_turn_ddx(episode: dict[str, Any]) -> list[tuple[str, list[str]]]:
    turns: list[tuple[str, list[str]]] = []
    for turn in episode.get("turns", []) or []:
        clinician = turn.get("clinician") if isinstance(turn, dict) else {}
        if not isinstance(clinician, dict):
            continue
        ddx = [canonical_diagnosis(item) for item in clinician.get("current_ddx", []) if canonical_diagnosis(item)]
        if ddx:
            turns.append((str(turn.get("turn_number") or len(turns) + 1), ddx[:5]))

    final = episode.get("final") if isinstance(episode.get("final"), dict) else {}
    final_ddx = [canonical_diagnosis(item) for item in final.get("current_ddx", []) if canonical_diagnosis(item)]
    if final_ddx:
        turns.append(("Final", final_ddx[:5]))
    return turns


def build_rank_table(turns: list[tuple[str, list[str]]]) -> dict[str, list[int | None]]:
    ranks: dict[str, list[int | None]] = defaultdict(lambda: [None] * len(turns))
    for x_index, (_, ddx) in enumerate(turns):
        for rank, diagnosis in enumerate(ddx, start=1):
            ranks[diagnosis][x_index] = rank
    return dict(ranks)


def rank_sort_key(item: tuple[str, list[int | None]]) -> tuple[int, int, str]:
    diagnosis, ranks = item
    first_seen = next((index for index, rank in enumerate(ranks) if rank is not None), 999)
    best_rank = min((rank for rank in ranks if rank is not None), default=999)
    return first_seen, best_rank, diagnosis.lower()


def plot_episode(episode_path: Path, output_path: Path) -> None:
    episode = _read_json(episode_path)
    turns = collect_turn_ddx(episode)
    if not turns:
        raise ValueError(f"No per-turn DDx found in {episode_path}")

    ranks = dict(sorted(build_rank_table(turns).items(), key=rank_sort_key))
    x = list(range(1, len(turns) + 1))
    x_labels = [label for label, _ in turns]
    case_id = str(episode.get("case_id") or episode_path.stem)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=220)

    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)
    for idx, (diagnosis, series) in enumerate(ranks.items()):
        color = palette[idx % len(palette)]
        visible_points = sum(rank is not None for rank in series)
        linewidth = 2.8 if diagnosis == "VEXAS syndrome" else 2.0
        alpha = 0.95 if visible_points >= 2 or diagnosis == "VEXAS syndrome" else 0.55
        marker_size = 5.5 if diagnosis == "VEXAS syndrome" else 4.5
        ax.plot(
            x,
            series,
            marker="o",
            markersize=marker_size,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            solid_capstyle="round",
            label=display_label(diagnosis),
        )

    ax.set_title(f"Differential Diagnosis Rank Trajectory ({case_id})", pad=14)
    ax.set_xlabel("Dialogue turn")
    ax.set_ylabel("Rank in clinician DDx")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(5.35, 0.65)
    ax.set_xlim(0.7, len(x) + 0.3)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.18)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    legend = ax.legend(
        title="Diagnosis",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0,
        handlelength=2.4,
    )
    legend._legend_box.align = "left"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-turn DDx rank trajectory from a dialogue episode.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_episode(args.episode, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
