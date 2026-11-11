#!/usr/bin/env python
"""Generate the LaTeX tables in paper/tables from the result CSVs."""

import argparse
import math
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
    "vit_small_patch16_224": "ViT-S/16",
}
FITZPATRICK_ORDER = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5}
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
    parser.add_argument("--xai", default="results/xai/xai_all_summary.csv")
    parser.add_argument(
        "--xai-correlations", default="results/xai/xai_all_correlations.csv"
    )
    parser.add_argument(
        "--external", default="results/external/hiba/external_summary.csv"
    )
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
    tone_order = {"light": 0, "medium": 1, "dark": 2}
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame = summary.copy()
    frame["_tone"] = frame["tone_group"].map(tone_order).fillna(3)
    frame["_backbone"] = frame["backbone"].map(backbone_order).fillna(99)
    frame = frame.sort_values(["_backbone", "_tone"])

    headers = [
        "Architecture",
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
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for block, (backbone, group) in enumerate(frame.groupby("backbone", sort=False)):
        if block:
            lines.append(r"\midrule")
        rows = list(group.iterrows())
        architecture = BACKBONES.get(backbone, backbone)
        for index, (_, row) in enumerate(rows):
            name = (
                f"\\multirow{{{len(rows)}}}{{*}}{{{architecture}}}"
                if index == 0
                else ""
            )
            values = [
                name,
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


def format_p(value):
    if value is None or not math.isfinite(value):
        return "nan"
    if value < 1e-4:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / 10**exponent
        return f"${mantissa:.1f}\\times10^{{{exponent}}}$"
    return f"{value:.3f}"


def xai_correlation_table(frame, caption, label):
    headers = [
        "Architecture",
        "Images",
        "$\\rho$ (mass, tone)",
        "$p$",
        "$\\rho$ (mass, error)",
        "$p$",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        architecture = BACKBONES.get(row["backbone"], row["backbone"])
        lines.append(
            f"{architecture} & {int(row['n'])} & "
            f"{row['attribution_vs_tone_rho']:.3f} & "
            f"{format_p(row['attribution_vs_tone_p'])} & "
            f"{row['attribution_vs_error_rho']:.3f} & "
            f"{format_p(row['attribution_vs_error_p'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def format_metric(value):
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.3f}"


def external_table(summary, caption, label):
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    group_order = dict(FITZPATRICK_ORDER)
    group_order["all"] = 6
    frame = summary[summary["tone_group"] != "unknown"].copy()
    frame["_backbone"] = frame["backbone"].map(backbone_order).fillna(99)
    frame["_group"] = frame["tone_group"].map(group_order).fillna(99)
    frame = frame.sort_values(["_backbone", "_group"])

    headers = [
        "Architecture",
        "Fitzpatrick",
        "$n$",
        "Melanomas",
        "Sensitivity",
        "Specificity",
        "AUC",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for block, (backbone, group) in enumerate(frame.groupby("backbone", sort=False)):
        if block:
            lines.append(r"\midrule")
        rows = list(group.iterrows())
        architecture = BACKBONES.get(backbone, backbone)
        for index, (_, row) in enumerate(rows):
            name = (
                f"\\multirow{{{len(rows)}}}{{*}}{{{architecture}}}"
                if index == 0
                else ""
            )
            group_name = "All" if row["tone_group"] == "all" else row["tone_group"]
            values = [
                name,
                group_name,
                str(int(row["n_mean"])),
                str(int(row["n_positive_mean"])),
                format_metric(row["sensitivity_mean"]),
                format_metric(row["specificity_mean"]),
                format_metric(row["auc_mean"]),
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
                "Quantitative XAI at the CE baseline on the HAM10000 test split. Each architecture uses three seeds and a tone-stratified sample of 343 test images per seed. Attribution mass is the fraction of the Grad-CAM heatmap inside the lesion mask. Values are means across seeds.",
                "tab:xai",
            ),
            encoding="utf-8",
        )
        print(xai_output)

        correlations_path = resolve(args.xai_correlations)
        if correlations_path.exists():
            correlations = pd.read_csv(correlations_path)
            correlation_output = out_dir / "xai_correlations.tex"
            correlation_output.write_text(
                xai_correlation_table(
                    correlations,
                    "Spearman correlations between attribution mass inside the lesion, lightness group, and error status at the CE baseline. Tone is coded light $=0$, medium $=1$, dark $=2$. A negative mass-tone correlation means less attribution inside the lesion on darker images.",
                    "tab:xaicorr",
                ),
                encoding="utf-8",
            )
            print(correlation_output)
        else:
            print(f"Skipping XAI correlation table: {correlations_path} not found")
    else:
        print(f"Skipping XAI table: {xai_path} not found")

    external_path = resolve(args.external)
    if external_path.exists():
        external_summary = pd.read_csv(external_path)
        external_output = out_dir / "external.tex"
        external_output.write_text(
            external_table(
                external_summary,
                "External validation on the HIBA dermoscopic subset (1,280 images, 196 melanomas). The threshold is the HAM10000 validation threshold at 95\\% specificity, frozen before the external evaluation. Values are means across three seeds. Fitzpatrick IV contains 25 images and no melanomas, so sensitivity is undefined. The 108 images without a Fitzpatrick label are included in the All row only.",
                "tab:external",
            ),
            encoding="utf-8",
        )
        print(external_output)
    else:
        print(f"Skipping external table: {external_path} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
