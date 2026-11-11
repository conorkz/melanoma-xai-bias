#!/usr/bin/env python
"""Generate the LaTeX tables in paper/tables from the result CSVs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.utils import ensure_dir

BACKBONES = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_s": "ViT-S/16",
}
CONDITIONS = {
    "ce_random": "CE",
    "ce_random_strong": "CE+strong",
    "focal2_cb": "Focal+CB",
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--stage1", default="results/rq1_stage1.csv")
    parser.add_argument("--stage2", default="results/rq1_stage2.csv")
    parser.add_argument("--fairness", default="results/fairness/lightness/fairness_summary.csv")
    parser.add_argument("--xai", default="results/xai/efficientnet_b0/xai_summary.csv")
    parser.add_argument("--out-dir", default="paper/tables")
    return parser.parse_args(argv)


def escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def plus_minus(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def confidence_interval(mean, low, high):
    return f"{mean:.3f} ({low:.3f} to {high:.3f})"


def stage_table(frame, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & Val sens & Test sens (95\% CI) & Specificity & Balanced acc & AUC \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{escape(row['config_id'])} & "
            f"{plus_minus(row['val_sensitivity_mean'], row['val_sensitivity_std'])} & "
            f"{confidence_interval(row['sensitivity_mean'], row['sensitivity_ci_low'], row['sensitivity_ci_high'])} & "
            f"{row['specificity_mean']:.3f} & "
            f"{row['balanced_accuracy_mean']:.3f} & "
            f"{row['auc_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def short_name(config_id):
    for backbone_key, backbone_name in BACKBONES.items():
        prefix = f"stage2_{backbone_key}_"
        if config_id.startswith(prefix):
            condition = config_id[len(prefix):]
            return f"{backbone_name} / {CONDITIONS.get(condition, condition)}"
    return config_id


def fairness_table(summary, caption, label):
    means = summary.pivot(index="config_id", columns="tone_group", values="sensitivity_mean")
    stds = summary.pivot(index="config_id", columns="tone_group", values="sensitivity_std")
    dark = summary[summary["tone_group"] == "dark"].set_index("config_id")
    dark_n = int(dark["n"].iloc[0]) if not dark.empty else 0
    dark_positive = int(dark["n_positive"].iloc[0]) if not dark.empty else 0

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Configuration & Light sens & Medium sens & Dark sens & Dark AUC & Dark FPR & Dark $n$ / pos. \\",
        r"\midrule",
    ]
    for config_id in means.index:
        light = plus_minus(means.loc[config_id, "light"], stds.loc[config_id, "light"])
        medium = plus_minus(means.loc[config_id, "medium"], stds.loc[config_id, "medium"])
        dark_sens = plus_minus(means.loc[config_id, "dark"], stds.loc[config_id, "dark"])
        dark_row = dark.loc[config_id]
        lines.append(
            f"{escape(short_name(config_id))} & {light} & {medium} & {dark_sens} & "
            f"{dark_row['auc_mean']:.3f} & {dark_row['fpr_mean']:.3f} & "
            f"{dark_n} / {dark_positive} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def xai_table(summary, caption, label):
    order = {"light": 0, "medium": 1, "dark": 2}
    frame = summary.copy()
    frame["_order"] = frame["tone_group"].map(order).fillna(3)
    frame = frame.sort_values("_order")
    columns = [
        "tone_group",
        "n",
        "n_positive",
        "attribution_mass_mean",
        "pointing_game_mean",
        "attribution_iou_mean",
        "error_rate",
    ]
    headers = [
        "Tone group",
        "$n$",
        "$n$ positive",
        "Attribution mass",
        "Pointing game",
        "Attribution IoU",
        "Error rate",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        values = [
            str(row["tone_group"]),
            str(int(row["n"])),
            str(int(row["n_positive"])),
            f"{row['attribution_mass_mean']:.3f}",
            f"{row['pointing_game_mean']:.3f}",
            f"{row['attribution_iou_mean']:.3f}",
            f"{row['error_rate']:.3f}",
        ]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    out_dir = ensure_dir(resolve(args.out_dir))

    stage1 = pd.read_csv(resolve(args.stage1)).sort_values(
        "val_sensitivity_mean", ascending=False
    )
    stage2 = pd.read_csv(resolve(args.stage2)).sort_values(
        ["backbone", "sensitivity_mean"], ascending=[True, False]
    )
    fairness = pd.read_csv(resolve(args.fairness))

    stage1_path = out_dir / "stage1.tex"
    stage2_path = out_dir / "stage2.tex"
    fairness_path = out_dir / "fairness.tex"

    stage1_path.write_text(
        stage_table(
            stage1,
            "Stage 1 screen on EfficientNet-B0 at 160 pixels. Validation sensitivity is used for model selection. Test sensitivity is reported at the validation-selected 95\\% specificity threshold. Values are mean and standard deviation across three seeds.",
            "tab:stage1",
        ),
        encoding="utf-8",
    )
    stage2_path.write_text(
        stage_table(
            stage2,
            "Stage 2 comparison at 224 pixels. Test sensitivity is reported at the validation-selected 95\\% specificity threshold. Values are mean and standard deviation across three seeds.",
            "tab:stage2",
        ),
        encoding="utf-8",
    )
    fairness_path.write_text(
        fairness_table(
            fairness,
            "Lightness-stratified test sensitivity at 95\\% specificity. Values are mean and standard deviation across three seeds. The dark stratum contains 43 test images and six melanoma cases. These are image-based lightness groups, not clinical Fitzpatrick labels.",
            "tab:fairness",
        ),
        encoding="utf-8",
    )

    print(stage1_path)
    print(stage2_path)
    print(fairness_path)

    xai_path = resolve(args.xai)
    if xai_path.exists():
        xai_summary = pd.read_csv(xai_path)
        xai_output = out_dir / "xai.tex"
        xai_output.write_text(
            xai_table(
                xai_summary,
                "EfficientNet-B0 XAI pilot on the HAM10000 test split. Values are means across three seeds. The dark stratum contains 129 images and 18 melanoma cases in this sample.",
                "tab:xai",
            ),
            encoding="utf-8",
        )
        print(xai_output)
    else:
        print(f"Skipping XAI table: {xai_path} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
