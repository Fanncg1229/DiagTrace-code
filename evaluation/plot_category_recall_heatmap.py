from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "diagtrace-matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["GPT-4.1 mini", "Claude Haiku 4.5", "Gemini 2.5 Flash", "Kimi K2.5"]
MODEL_LABELS = {
    "GPT-4.1 mini": "GPT-4.1\nmini",
    "Claude Haiku 4.5": "Claude\nHaiku 4.5",
    "Gemini 2.5 Flash": "Gemini\n2.5 Flash",
    "Kimi K2.5": "Kimi\nK2.5",
}
CATEGORY_ORDER = [
    "physical_examination",
    "lab_tests",
    "microbiology",
    "imaging",
    "pathology",
    "procedure_or_intervention",
    "medication_or_treatment",
    "admission_or_clinical_course",
]
CATEGORY_LABELS = {
    "physical_examination": "Physical exam",
    "lab_tests": "Lab tests",
    "microbiology": "Microbiology",
    "imaging": "Imaging",
    "pathology": "Pathology",
    "procedure_or_intervention": "Procedure/intervention",
    "medication_or_treatment": "Treatment/medication",
    "admission_or_clinical_course": "Admission/course",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = df[df["category"].isin(CATEGORY_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)
    mat = df.pivot(index="category", columns="model", values="category_recall")
    mat = mat.loc[CATEGORY_ORDER, MODEL_ORDER]
    mat.index = [CATEGORY_LABELS[x] for x in mat.index]
    mat.columns = [MODEL_LABELS[x] for x in mat.columns]

    sns.set_theme(style="white", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(6.54, 3.41), dpi=300)
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.0},
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "Recall",
            "fraction": 0.025,
            "pad": 0.012,
            "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelrotation=0, labelsize=9.0, pad=2)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9.0, pad=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8.5, length=2)
    cbar.ax.yaxis.label.set_size(9.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.03)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"category_recall_heatmap_acl.{suffix}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()
