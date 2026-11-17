#!/usr/bin/env python
"""Generate the LaTeX tables in paper/tables from the result CSVs."""

import argparse
import re
import math
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

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
    "focal2_random": "Focal",
    "ce_cb": "CE+CB",
    "wce_random": "Weighted CE",
    "focal2_cb_strong": "Focal+CB+strong",
    "focal3_cb": "Focal ($\\gamma=3$)+CB",
}
CONDITION_ORDER = {
    "ce_random": 0,
    "focal2_cb": 1,
    "focal2_random": 2,
    "ce_cb": 3,
    "ce_random_strong": 4,
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--stage1", default="results/rq1_stage1.csv")
    parser.add_argument("--stage2", default="results/rq1_stage2.csv")
    parser.add_argument("--stage3", default="results/rq1_stage3.csv")
    parser.add_argument(
        "--fairness", default="results/stage3/fairness/lightness/fairness_summary.csv"
    )
    parser.add_argument("--xai", default="results/stage3/xai_true/xai_summary.csv")
    parser.add_argument(
        "--xai-correlations", default="results/stage3/xai_true/xai_correlations.csv"
    )
    parser.add_argument(
        "--external", default="results/stage3/external/hiba/external_summary.csv"
    )
    parser.add_argument(
        "--dark-skin", default="results/stage3/dark_skin/threshold_sensitivity.csv"
    )
    parser.add_argument(
        "--adjusted-model", default="results/stage3/dark_skin/continuous_model.csv"
    )
    parser.add_argument(
        "--calibration", default="results/stage3/calibration/calibration_overall.csv"
    )
    parser.add_argument(
        "--seed-intervals", default="results/stage3/seed_intervals"
    )
    parser.add_argument(
        "--seed-intervals-stage2", default="results/seed_intervals"
    )
    parser.add_argument(
        "--lr-runs", default="outputs/kaggle/rq1-stage-3-lr/results/runs"
    )
    parser.add_argument("--out-dir", default="paper/tables")
    return parser.parse_args(argv)


def escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def plus_minus(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def t_interval(mean, std, n=3):
    """95% t interval for a mean across n seeds."""
    if n < 2:
        raise ValueError(f"No t critical value for n={n}")
    t_critical = float(stats.t.ppf(0.975, n - 1))
    half = t_critical * std / math.sqrt(n)
    return f"{mean:.3f} ({mean - half:.3f} to {mean + half:.3f})"


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
        r"Configuration & Val sens & Test sens (95\% t CI) & Specificity & Balanced acc & AUC \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{short_name(row['config_id'])} & "
            f"{plus_minus(row['val_sensitivity_mean'], row['val_sensitivity_std'])} & "
            f"{t_interval(row['sensitivity_mean'], row['sensitivity_std'], int(row.get('n_seeds', 3)))} & "
            f"{row['specificity_mean']:.3f} & "
            f"{row['balanced_accuracy_mean']:.3f} & "
            f"{row['auc_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def short_name(config_id):
    """Readable label such as 'ResNet-50 / Focal+CB' for a config id."""
    stripped = re.sub(r"^stage\d+[a-z]*_", "", str(config_id))
    for backbone_key, backbone_name in BACKBONES.items():
        prefix = f"{backbone_key}_"
        if stripped.startswith(prefix):
            condition = stripped[len(prefix):]
            return f"{backbone_name} / {CONDITIONS.get(condition, escape(condition))}"
    return CONDITIONS.get(stripped, escape(stripped))


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


def format_mean_sd(mean, std):
    if mean is None or not math.isfinite(mean):
        return "--"
    if std is None or not math.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def external_table(summary, caption, label):
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    group_order = dict(FITZPATRICK_ORDER)
    group_order["all"] = 6
    frame = summary[summary["tone_group"] != "unknown"].copy()
    if "condition" in frame.columns:
        frame = frame[frame["condition"] == "ce_random"]
    frame = frame
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
                format_mean_sd(row["sensitivity_mean"], row["sensitivity_std"]),
                format_mean_sd(row["specificity_mean"], row["specificity_std"]),
                format_mean_sd(row["auc_mean"], row["auc_std"]),
            ]
            lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def dark_skin_table(summary, caption, label):
    group_order = {"light": 0, "medium": 1, "dark": 2}
    frame = summary.copy()
    frame["_group"] = frame["tone_group"].map(group_order).fillna(3)
    frame = frame.sort_values(["threshold", "_group"])
    aggregated = (
        frame.groupby(["threshold", "tone_group"])
        .agg(
            n=("n", "first"),
            n_positive=("n_positive", "first"),
            sensitivity=("sensitivity", "mean"),
            auc=("auc", "mean"),
            fpr=("fpr", "mean"),
        )
        .reset_index()
    )
    aggregated["_group"] = aggregated["tone_group"].map(group_order).fillna(3)
    aggregated = aggregated.sort_values(["threshold", "_group"])

    headers = [
        "$L^*$ cutoff",
        "Group",
        "$n$",
        "Melanomas",
        "Sensitivity",
        "AUC",
        "FPR",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for threshold, block in aggregated.groupby("threshold", sort=True):
        for index, (_, row) in enumerate(block.iterrows()):
            cutoff = str(int(threshold)) if index == 0 else ""
            lines.append(
                f"{cutoff} & {row['tone_group']} & {int(row['n'])} & "
                f"{int(row['n_positive'])} & {format_metric(row['sensitivity'])} & "
                f"{format_metric(row['auc'])} & {format_metric(row['fpr'])} \\\\"
            )
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _format_or(frame, term):
    subset = frame[frame["term"] == term]
    if subset.empty:
        return "--"
    row = subset.iloc[0]
    if not math.isfinite(row["odds_ratio"]):
        return "--"
    return (
        f"{row['odds_ratio']:.2f} "
        f"({row['ci_low']:.2f} to {row['ci_high']:.2f})"
    )


def adjusted_model_table(summary, caption, label):
    order = {"resnet50": 0, "efficientnet_b0": 1, "vit_s": 2}
    outcomes = [
        ("false_positive", "False positive"),
        ("false_negative", "False negative"),
    ]
    headers = ["Error type", "Architecture", "$L^*$ OR (95\\% CI)", "Contrast OR (95\\% CI)"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for outcome, outcome_label in outcomes:
        subset = summary[summary["outcome"] == outcome]
        rows = [backbone for backbone in BACKBONES if backbone in set(subset["backbone"])]
        rows = sorted(rows, key=lambda value: order.get(value, 99))
        for index, backbone in enumerate(rows):
            block = subset[subset["backbone"] == backbone]
            first = outcome_label if index == 0 else ""
            lines.append(
                f"{first} & {BACKBONES.get(backbone, backbone)} & "
                f"{_format_or(block, 'L')} & {_format_or(block, 'delta_L')} \\\\"
            )
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def calibration_table(summary, caption, label, corrected=None):
    frame = summary.copy()
    if corrected is not None:
        frame = frame.merge(
            corrected[["config_id", "ece_prior_ts_mean", "ece_prior_ts_std", "temperature_mean"]],
            on="config_id",
            how="left",
        )
    frame["_backbone"] = frame["config_id"].str.extract(
        r"^stage\d+_(resnet50|efficientnet_b0|vit_s)_"
    )[0]
    frame["_condition"] = frame["config_id"].str.replace(
        r"^stage\d+_(resnet50|efficientnet_b0|vit_s)_", "", regex=True
    )
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    condition_order = CONDITION_ORDER
    frame["_b"] = frame["_backbone"].map(backbone_order).fillna(99)
    frame["_c"] = frame["_condition"].map(condition_order).fillna(99)
    frame = frame.sort_values(["_b", "_c"])

    has_corrected = "ece_prior_ts_mean" in frame.columns
    headers = ["Configuration", "ECE", "Brier"]
    if has_corrected:
        headers += ["ECE, corrected", "$T$"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{l" + "c" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        configuration = (
            f"{BACKBONES.get(row['_backbone'], row['_backbone'])} / "
            f"{CONDITIONS.get(row['_condition'], row['_condition'])}"
        )
        cells = [
            configuration,
            f"{row['ece_mean']:.3f} $\\pm$ {row['ece_std']:.3f}",
            f"{row['brier_mean']:.3f} $\\pm$ {row['brier_std']:.3f}",
        ]
        if has_corrected:
            cells += [
                f"{row['ece_prior_ts_mean']:.3f} $\\pm$ {row['ece_prior_ts_std']:.3f}",
                f"{row['temperature_mean']:.2f}",
            ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def external_condition_table(summary, caption, label):
    condition_order = CONDITION_ORDER
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame = summary[summary["tone_group"] == "all"].copy()
    frame["_b"] = frame["backbone"].map(backbone_order).fillna(99)
    frame["_c"] = frame["condition"].map(condition_order).fillna(99)
    frame = frame.sort_values(["_b", "_c"])

    headers = [
        "Architecture",
        "Condition",
        "Sensitivity",
        "Specificity",
        "AUC",
        "Sens. at 95\\% spec.",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{BACKBONES.get(row['backbone'], row['backbone'])} & "
            f"{CONDITIONS.get(row['condition'], row['condition'])} & "
            f"{format_mean_sd(row['sensitivity_mean'], row['sensitivity_std'])} & "
            f"{format_mean_sd(row['specificity_mean'], row['specificity_std'])} & "
            f"{format_mean_sd(row['auc_mean'], row['auc_std'])} & "
            f"{format_mean_sd(row['sensitivity_at_95_spec_mean'], row['sensitivity_at_95_spec_std'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def _direction(values):
    values = pd.Series(values)
    negative = int((values < 0).sum())
    positive = int((values > 0).sum())
    n = len(values)
    if negative >= positive:
        return f"{negative}/{n} negative"
    return f"{positive}/{n} positive"


def seed_ranking_table(
    paired_summary, did_summary, paired_per_seed, did_per_seed, caption, label,
    did_conditions=("focal2_cb",),
):
    headers = [
        "Test",
        "Architecture",
        "$\\Delta$ sensitivity",
        "95\\% t CI",
        "Direction",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    paired = paired_summary[paired_summary["metric"] == "delta_sensitivity"]
    conditions = sorted(
        paired["condition"].unique(), key=lambda value: CONDITION_ORDER.get(value, 99)
    )
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    for condition in conditions:
        subset = paired[paired["condition"] == condition]
        subset = subset.assign(_b=subset["backbone"].map(backbone_order)).sort_values("_b")
        for _, row in subset.iterrows():
            seeds = paired_per_seed[
                (paired_per_seed["backbone"] == row["backbone"])
                & (paired_per_seed["condition"] == condition)
            ]["delta_sensitivity"]
            lines.append(
                f"{CONDITIONS.get(condition, condition)} vs CE & "
                f"{BACKBONES.get(row['backbone'], row['backbone'])} & "
                f"{row['mean']:+.3f} & {row['t_low']:+.3f} to {row['t_high']:+.3f} & "
                f"{_direction(seeds)} \\\\"
            )
    did = did_summary[did_summary["metric"] == "delta_sensitivity"]
    if "condition" not in did.columns:
        did = did.assign(condition="focal2_cb")
        did_per_seed = did_per_seed.assign(condition="focal2_cb")
    for condition in did_conditions:
        block = did[did["condition"] == condition]
        if block.empty:
            continue
        lines.append(r"\midrule")
        for _, row in block.iterrows():
            seeds = did_per_seed[
                (did_per_seed["condition"] == condition)
                & (did_per_seed["architecture_a"] == row["architecture_a"])
                & (did_per_seed["architecture_b"] == row["architecture_b"])
            ]["delta_sensitivity"]
            lines.append(
                f"Interaction, {CONDITIONS.get(condition, condition)} & "
                f"{BACKBONES.get(row['architecture_a'], row['architecture_a'])} minus "
                f"{BACKBONES.get(row['architecture_b'], row['architecture_b'])} & "
                f"{row['mean']:+.3f} & {row['t_low']:+.3f} to {row['t_high']:+.3f} & "
                f"{_direction(seeds)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def supp_interaction_table(did_summary, caption, label):
    did = did_summary[did_summary["metric"] == "delta_sensitivity"].copy()
    if "condition" not in did.columns:
        did = did.assign(condition="focal2_cb")
    headers = ["Correction", "Comparison", "$\\Delta$ sensitivity", "95\\% t CI"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    pairs = (
        ("efficientnet_b0", "vit_s"),
        ("resnet50", "vit_s"),
        ("resnet50", "efficientnet_b0"),
    )
    conditions = sorted(
        did["condition"].unique(), key=lambda value: CONDITION_ORDER.get(value, 99)
    )
    for condition in conditions:
        block = did[did["condition"] == condition]
        for first, second in pairs:
            row = block[
                (block["architecture_a"] == first)
                & (block["architecture_b"] == second)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                f"{CONDITIONS.get(condition, condition)} & "
                f"{BACKBONES[first]} minus {BACKBONES[second]} & "
                f"{row['mean']:+.3f} & {row['t_low']:+.3f} to {row['t_high']:+.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def protocol_table(stage2_dir, stage3_dir, caption, label):
    """Focal+CB effect and CE AUC under the original and revised protocols."""
    rows = []
    for name, directory in (("Original", stage2_dir), ("Revised", stage3_dir)):
        paired = pd.read_csv(directory / "paired_summary.csv")
        paired = paired[
            (paired["metric"] == "delta_sensitivity") & (paired["condition"] == "focal2_cb")
        ]
        per_seed = next(directory.glob("stage*_per_seed.csv"))
        per_seed = pd.read_csv(per_seed)
        base = per_seed[per_seed["condition"] == "ce_random"]
        for _, row in paired.iterrows():
            ce = base[base["backbone"] == row["backbone"]]
            rows.append(
                {
                    "protocol": name,
                    "backbone": row["backbone"],
                    "n": int(row["n_seeds"]),
                    "ce_sens": ce["sensitivity"].mean(),
                    "ce_auc": ce["auc"].mean(),
                    "delta": row["mean"],
                    "low": row["t_low"],
                    "high": row["t_high"],
                }
            )
    frame = pd.DataFrame(rows)
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame["_b"] = frame["backbone"].map(backbone_order)
    frame = frame.sort_values(["_b", "protocol"])
    headers = [
        "Architecture",
        "Protocol",
        "Seeds",
        "CE sensitivity",
        "CE AUC",
        "$\\Delta$ sensitivity, Focal+CB vs CE (95\\% t CI)",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for block, (backbone, group) in enumerate(frame.groupby("_b", sort=True)):
        if block:
            lines.append(r"\midrule")
        for index, (_, row) in enumerate(group.iterrows()):
            name = BACKBONES.get(row["backbone"], row["backbone"]) if index == 0 else ""
            lines.append(
                f"{name} & {row['protocol']} & {row['n']} & {row['ce_sens']:.3f} & "
                f"{row['ce_auc']:.3f} & {row['delta']:+.3f} "
                f"({row['low']:+.3f} to {row['high']:+.3f}) \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def auc_tests_table(summary, caption, label):
    labels = {"ce_random": "CE", "focal2_cb": "Focal+CB"}
    headers = ["Condition", "Comparison", "$\\Delta$ AUC", "95\\% t CI"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for condition in ("ce_random", "focal2_cb"):
        subset = summary[summary["condition"] == condition]
        for _, row in subset.iterrows():
            lines.append(
                f"{labels[condition]} & "
                f"{BACKBONES.get(row['architecture_a'], row['architecture_a'])} minus "
                f"{BACKBONES.get(row['architecture_b'], row['architecture_b'])} & "
                f"{row['mean']:+.3f} & {row['t_low']:+.3f} to {row['t_high']:+.3f} \\\\"
            )
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def per_seed_table(per_seed, caption, label):
    """Compact per-seed sensitivity and AUC, one row per configuration."""
    frame = per_seed.copy()
    seeds = sorted(frame["seed"].unique())
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame["_b"] = frame["backbone"].map(backbone_order)
    frame["_c"] = frame["condition"].map(CONDITION_ORDER).fillna(99)
    columns = "ll" + "c" * (2 * len(seeds))
    seed_header = " & ".join(str(seed) for seed in seeds)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        r" & & \multicolumn{" + str(len(seeds)) + r"}{c}{Sensitivity by seed} & \multicolumn{"
        + str(len(seeds)) + r"}{c}{AUC by seed} \\",
        r"Architecture & Condition & " + seed_header + " & " + seed_header + r" \\",
        r"\midrule",
    ]
    for (_, _, backbone, condition), group in frame.groupby(
        ["_b", "_c", "backbone", "condition"], sort=True
    ):
        group = group.set_index("seed")
        cells = [BACKBONES.get(backbone, backbone), CONDITIONS.get(condition, condition)]
        cells += [f"{group.loc[seed, 'sensitivity']:.3f}" for seed in seeds]
        cells += [f"{group.loc[seed, 'auc']:.3f}" for seed in seeds]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def lr_table(runs, caption, label):
    """Learning-rate check: validation and test AUC per architecture and rate."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Architecture & Learning rate & Val AUC & Best epoch & Test AUC & Test sensitivity \\",
        r"\midrule",
    ]
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame = runs.assign(_b=runs["backbone"].map(backbone_order)).sort_values(["_b", "lr"])
    for block, (_, group) in enumerate(frame.groupby("_b", sort=True)):
        if block:
            lines.append(r"\midrule")
        best = group["val_auc"].idxmax()
        for index, (row_index, row) in enumerate(group.iterrows()):
            name = BACKBONES.get(row["backbone"], row["backbone"]) if index == 0 else ""
            mantissa, exponent = f"{row['lr']:.0e}".split("e")
            rate = f"${int(mantissa)}\\times10^{{{int(exponent)}}}$"
            val = f"{row['val_auc']:.3f}"
            if row_index == best:
                val = r"\textbf{" + val + "}"
            lines.append(
                f"{name} & {rate} & {val} & {int(row['best_epoch'])} & "
                f"{row['test_auc']:.3f} & {row['test_sensitivity']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def error_type_table(frame, caption, label):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Architecture & Error type & Image evaluations & Attribution mass \\",
        r"\midrule",
    ]
    backbone_order = {key: index for index, key in enumerate(BACKBONES)}
    frame = frame.assign(_b=frame["backbone"].map(backbone_order)).sort_values(
        ["_b", "error_type"]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"{BACKBONES.get(row['backbone'], row['backbone'])} & {row['error_type']} & "
            f"{int(row['n_images'])} & {row['attribution_mass_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def load_lr_runs(runs_dir):
    import json

    rows = []
    for path in sorted(Path(runs_dir).glob("stage3lr_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        match = re.match(
            r"^stage3lr_(resnet50|efficientnet_b0|vit_s)_.*_lr([0-9.e-]+)_seed\d+$",
            data.get("run_id", path.parent.name),
        )
        if match is None:
            continue
        rows.append(
            {
                "backbone": match.group(1),
                "lr": float(match.group(2)),
                "val_auc": data["val"]["at_0.5"]["auc"],
                "best_epoch": data.get("best_epoch"),
                "test_auc": data["test"]["at_target"]["auc"],
                "test_sensitivity": data["test"]["at_target"]["sensitivity"],
            }
        )
    return pd.DataFrame(rows)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    print(path)


def main(argv=None):
    args = parse_args(argv)
    out_dir = ensure_dir(resolve(args.out_dir))

    stage1 = pd.read_csv(resolve(args.stage1)).sort_values(
        "val_sensitivity_mean", ascending=False
    )
    _write(
        out_dir / "stage1.tex",
        stage_table(
            stage1,
            "Stage 1 screen on EfficientNet-B0 at 160 pixels. Validation sensitivity is used for model selection. Validation values are mean and standard deviation across three seeds. Test sensitivity is the mean with a 95\\% t interval across three seeds at the validation-selected 95\\% specificity threshold.",
            "tab:stage1",
        ),
    )

    backbone_order = {key: index for index, key in enumerate(BACKBONES)}

    def ordered(frame):
        frame = frame.copy()
        frame["_b"] = frame["backbone"].map(backbone_order).fillna(99)
        frame["_c"] = (
            frame["config_id"]
            .str.replace(r"^stage\d+_(resnet50|efficientnet_b0|vit_s)_", "", regex=True)
            .map(CONDITION_ORDER)
            .fillna(99)
        )
        return frame.sort_values(["_b", "_c"])

    stage2 = ordered(pd.read_csv(resolve(args.stage2)))
    _write(
        out_dir / "stage2.tex",
        stage_table(
            stage2,
            "Stage 2 under the original protocol (shared learning rate $3\\times10^{-4}$, checkpoint selected by validation balanced accuracy at the 0.5 threshold) at 224 pixels. Validation values are mean and standard deviation across three seeds. Test sensitivity is the mean with a 95\\% t interval across three seeds at the validation-selected 95\\% specificity threshold.",
            "tab:stage2",
        ),
    )

    stage3_path = resolve(args.stage3)
    if stage3_path.exists():
        stage3 = ordered(pd.read_csv(stage3_path))
        _write(
            out_dir / "stage3.tex",
            stage_table(
                stage3,
                "Stage 3 under the revised protocol (learning rate selected per architecture, checkpoint selected by validation AUC) at 224 pixels. Validation values are mean and standard deviation across five seeds. Test sensitivity is the mean with a 95\\% t interval across five seeds at the validation-selected 95\\% specificity threshold. CB is class-balanced sampling; Focal uses $\\gamma=2$ and $\\alpha=0.75$.",
                "tab:stage3",
            ),
        )

    fairness_path = resolve(args.fairness)
    if fairness_path.exists():
        fairness = pd.read_csv(fairness_path)
        fairness = fairness.assign(
            backbone=fairness["config_id"].str.extract(
                r"^stage\d+_(resnet50|efficientnet_b0|vit_s)_"
            )[0]
        )
        order = ordered(
            fairness.drop_duplicates("config_id")[["config_id", "backbone"]]
        )["config_id"].tolist()
        fairness["config_id"] = pd.Categorical(
            fairness["config_id"], categories=order, ordered=True
        )
        _write(
            out_dir / "fairness.tex",
            fairness_table(
                fairness.drop(columns="backbone"),
                "Lightness-stratified test sensitivity at the validation-selected 95\\% specificity threshold, Stage 3. Values are mean and standard deviation across five seeds. The dark stratum ($L^*\\leq55$) contains 43 test images and six melanoma cases. These are image-based lightness groups, not clinical Fitzpatrick labels.",
                "tab:fairness",
            ),
        )

    xai_path = resolve(args.xai)
    if xai_path.exists():
        _write(
            out_dir / "xai.tex",
            xai_table(
                pd.read_csv(xai_path),
                "Quantitative XAI at the CE baseline on the HAM10000 test split, Stage 3. Grad-CAM targets the true class. Each architecture uses five seeds and all 1,527 test images per seed; $n$ counts image evaluations summed over seeds. Attribution mass is the fraction of the heatmap inside the lesion mask. Values are means across seeds.",
                "tab:xai",
            ),
        )
        correlations_path = resolve(args.xai_correlations)
        if correlations_path.exists():
            _write(
                out_dir / "xai_correlations.tex",
                xai_correlation_table(
                    pd.read_csv(correlations_path),
                    "Spearman correlations between attribution mass inside the lesion, lightness group, and error status at the CE baseline, Stage 3. Tone is coded light $=0$, medium $=1$, dark $=2$. A negative mass-tone correlation means less attribution inside the lesion on darker images. Images pool five seeds, so the $p$-values are descriptive.",
                    "tab:xaicorr",
                ),
            )
    else:
        print(f"Skipping XAI table: {xai_path} not found")

    dark_path = resolve(args.dark_skin)
    if dark_path.exists():
        _write(
            out_dir / "dark_skin.tex",
            dark_skin_table(
                pd.read_csv(dark_path),
                "Dark-stratum cutoff sensitivity at the validation-selected 95\\% specificity threshold, Stage 3. Values are means across the 15 CE-baseline runs (three architectures, five seeds). These are image-based lightness groups, not clinical Fitzpatrick labels.",
                "tab:darkskin",
            ),
        )

    adjusted_path = resolve(args.adjusted_model)
    if adjusted_path.exists():
        _write(
            out_dir / "adjusted_model.tex",
            adjusted_model_table(
                pd.read_csv(adjusted_path),
                "Adjusted odds ratios for false positives and false negatives at the CE baseline, Stage 3. Each model includes perilesional $L^*$, lesion contrast, log lesion area, age, sex, anatomical site, and acquisition source. Odds ratios are per one standard deviation; intervals are lesion-cluster bootstrap 95\\% CIs.",
                "tab:adjusted",
            ),
        )

    calibration_path = resolve(args.calibration)
    if calibration_path.exists():
        corrected_path = calibration_path.parent / "calibration_corrected.csv"
        _write(
            out_dir / "calibration.tex",
            calibration_table(
                pd.read_csv(calibration_path),
                "Test-set calibration, Stage 3. ECE uses 10 equal-width bins. The corrected column first moves the scores from the effective training prior (set by the sampler and the focal $\\alpha$) to the training prevalence, then applies a temperature $T$ fitted on the validation split. Both corrections are monotone, so AUC and the validation-selected operating point are unchanged. Values are mean and standard deviation across five seeds.",
                "tab:calibration",
                corrected=pd.read_csv(corrected_path) if corrected_path.exists() else None,
            ),
        )

    seed_dir = resolve(args.seed_intervals)
    stage2_seed_dir = resolve(args.seed_intervals_stage2)
    needed = ("paired_summary.csv", "did_summary.csv", "paired_per_seed.csv", "did_per_seed.csv")
    if all((seed_dir / name).exists() for name in needed):
        _write(
            out_dir / "ranking.tex",
            seed_ranking_table(
                pd.read_csv(seed_dir / "paired_summary.csv"),
                pd.read_csv(seed_dir / "did_summary.csv"),
                pd.read_csv(seed_dir / "paired_per_seed.csv"),
                pd.read_csv(seed_dir / "did_per_seed.csv"),
                "Seed-level paired sensitivity changes relative to the CE baseline, Stage 3. Intervals are 95\\% t intervals across five seeds. The direction column counts the seeds with the stated sign. The interaction rows compare the Focal+CB change of one architecture with that of another.",
                "tab:ranking",
            ),
        )
        _write(
            out_dir / "supp_interactions.tex",
            supp_interaction_table(
                pd.read_csv(seed_dir / "did_summary.csv"),
                "Stage 3 interaction intervals for all four corrections and all three architecture pairs. Each value is the difference between two architectures in their change relative to the CE baseline (difference-in-differences in sensitivity at the validation-selected 95\\% specificity threshold). Intervals are 95\\% t intervals across five seeds. Only the focal-only ResNet-50 minus ViT-S/16 interval excludes zero.",
                "tab:suppinteractions",
            ),
        )
    if all((stage2_seed_dir / name).exists() for name in needed) and (
        seed_dir / "paired_summary.csv"
    ).exists():
        _write(
            out_dir / "protocol.tex",
            protocol_table(
                stage2_seed_dir,
                seed_dir,
                "Effect of focal loss with class-balanced sampling under the original protocol (Stage 2: shared learning rate, checkpoint by validation balanced accuracy at 0.5, three seeds) and the revised protocol (Stage 3: per-architecture learning rate, checkpoint by validation AUC, five seeds). Sensitivity is at the validation-selected 95\\% specificity threshold. Intervals are 95\\% t intervals across seeds.",
                "tab:protocol",
            ),
        )
        did2 = pd.read_csv(stage2_seed_dir / "did_summary.csv")
        did3 = pd.read_csv(seed_dir / "did_summary.csv")
        if "condition" in did3.columns:
            did3 = did3[did3["condition"] == "focal2_cb"]
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Architecture interaction for focal loss with class-balanced sampling: the difference between two architectures in their Focal+CB minus CE sensitivity change, under the original (Stage 2, three seeds) and revised (Stage 3, five seeds) protocols. Intervals are 95\% t intervals across seeds.}",
            r"\label{tab:interaction}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Comparison & Original protocol & Revised protocol \\",
            r"\midrule",
        ]
        for first, second in (
            ("efficientnet_b0", "vit_s"),
            ("resnet50", "vit_s"),
            ("resnet50", "efficientnet_b0"),
        ):
            cells = []
            for did in (did2, did3):
                row = did[
                    (did["metric"] == "delta_sensitivity")
                    & (did["architecture_a"] == first)
                    & (did["architecture_b"] == second)
                ]
                if row.empty:
                    cells.append("--")
                    continue
                row = row.iloc[0]
                cells.append(
                    f"{row['mean']:+.3f} ({row['t_low']:+.3f} to {row['t_high']:+.3f})"
                )
            lines.append(
                f"{BACKBONES[first]} minus {BACKBONES[second]} & {cells[0]} & {cells[1]} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        _write(out_dir / "interaction.tex", "\n".join(lines))

    auc_summary_path = seed_dir / "auc_summary.csv"
    if auc_summary_path.exists():
        _write(
            out_dir / "auc_tests.tex",
            auc_tests_table(
                pd.read_csv(auc_summary_path),
                "AUC differences between architectures at the CE baseline and under focal loss with class-balanced sampling, Stage 3. Intervals are 95\\% t intervals across five seeds. Positive values mean the first architecture has the higher AUC.",
                "tab:auctests",
            ),
        )

    per_seed_path = seed_dir / "stage3_per_seed.csv"
    if per_seed_path.exists():
        _write(
            out_dir / "supp_stage3_per_seed.tex",
            per_seed_table(
                pd.read_csv(per_seed_path),
                "Per-seed Stage 3 test sensitivity at the validation-selected 95\\% specificity threshold and test AUC.",
                "tab:suppstage3",
            ),
        )
    lr_runs_dir = resolve(args.lr_runs)
    if lr_runs_dir.exists():
        lr_runs = load_lr_runs(lr_runs_dir)
        if not lr_runs.empty:
            _write(
                out_dir / "supp_lr.tex",
                lr_table(
                    lr_runs,
                    "Learning rate check: CE baseline, seed 1, checkpoint selected by validation AUC. The learning rate with the highest validation AUC (bold) was used in Stage 3. Test values are shown for completeness and were not used for the choice.",
                    "tab:supplr",
                ),
            )
    error_type_path = xai_path.parent / "error_type_mass.csv"
    if error_type_path.exists():
        _write(
            out_dir / "supp_xai_error_type.tex",
            error_type_table(
                pd.read_csv(error_type_path),
                "Attribution mass inside the lesion by error type for the Stage 3 CE baselines, with Grad-CAM targeting the true class. Values are means across image evaluations pooled over five seeds.",
                "tab:suppxai",
            ),
        )

    external_path = resolve(args.external)
    if external_path.exists():
        external_summary = pd.read_csv(external_path)
        n_images = int(
            external_summary[external_summary["tone_group"] == "all"]["n_mean"].iloc[0]
        )
        n_melanoma = int(
            external_summary[external_summary["tone_group"] == "all"]["n_positive_mean"].iloc[0]
        )
        _write(
            out_dir / "external.tex",
            external_table(
                external_summary,
                f"External validation on the HIBA dermoscopic subset for the CE-baseline models, Stage 3 ({n_images:,} images, {n_melanoma} melanomas). The threshold is the HAM10000 validation threshold at 95\\% specificity, frozen before the external evaluation. Values are mean and standard deviation across five seeds. Fitzpatrick IV contains no melanomas, so sensitivity is undefined. Images without a Fitzpatrick label are included in the All row only.",
                "tab:external",
            ),
        )
        if "condition" in external_summary.columns:
            _write(
                out_dir / "external_conditions.tex",
                external_condition_table(
                    external_summary,
                    "External validation on HIBA for three training conditions, Stage 3. Thresholds are the frozen HAM10000 validation thresholds. Sensitivity at 95\\% specificity uses an oracle threshold fitted on HIBA and is an upper bound, not a deployable result. Values are mean and standard deviation across five seeds.",
                    "tab:externalcond",
                ),
            )
    else:
        print(f"Skipping external table: {external_path} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
