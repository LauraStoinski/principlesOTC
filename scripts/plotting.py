from os.path import join as pjoin
from pathlib import Path

import cortex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import t

okabe_ito = ["#E69F00", "#56B4E9"]


def plot_flatmap(sub, img, cm, vlims, outname, legend = False):
    if type(vlims) == list:
        lower = vlims[0]
        upper = vlims[1]
    else:
        lower = -vlims
        upper = vlims
    roi_select = ['V1', 'V2', 'V3', 'FFA','OFA', 'PPA', 'EBA', 'pSTS', 'MPA', 'OPA']
    arr = np.array(img.dataobj)
    input = np.swapaxes(arr,0,2) 
   
    
    vol_data = cortex.Volume(input, f'S{sub}', xfmname = 'align_auto', vmin=lower, vmax=upper, cmap = cm)

    fig = plt.figure(figsize=(20,10))


    cortex.quickflat.make_figure(
            vol_data,pixelwise=True, with_rois=True, roi_list=roi_select,with_sulci=True,
            with_colorbar = legend , fig=fig, dpi=600, labelsize=30, with_curvature=True)

    fig.savefig(outname, dpi=600)
    
    
def plot_tripartite(resdir, subdf, outname, voxMem=True, plot_original=False, plot_confi = True, conds = ["large animals", "small animals", "large objects", "small objects"], legend = False):
    """
    plot_original: plot non interpoated dots on top of line
    plot_confi: if True, SEM* 1.96 is plotted, otherwise only the SEM is plotted. Elde
    """
    anchors = [7, 5, 7, 5, 7]
    others = [' ']
    sphere_labels = ['MPA'] + others * (anchors[0] - 1) + ['PPA'] + others * (anchors[1] - 1) + ['FFA'] + others * (anchors[2] - 1) + ['EBA'] + others * (anchors[3] - 1) + ['OPA'] + others * (anchors[4] - 2) + ['MPA']

    matplotlib.rcParams['font.sans-serif'] = ['Arial']

    
    subdf["average"] = subdf[conds].mean(axis=1)  # compute mean across groups for normalization
    subdf[conds[0]] = subdf[conds[0]] - subdf["average"]
    subdf[conds[1]] = subdf[conds[1]] - subdf["average"]
    subdf[conds[2]] = subdf[conds[2]] - subdf["average"]
    subdf[conds[3]] = subdf[conds[3]] - subdf["average"]

    if voxMem: # plot SEM of members of a group
       conds2 = ["sem " + i for i in conds]
    else: # plot SEM of subjects
        conds2 = ["sem_subj_" + i for i in conds]
    
    # as I scale the response profiles, also normalize the sems
    subdf["sem average"] = subdf[conds2].mean(axis=1)
    subdf[conds2[0]] = subdf[conds2[0]] - subdf["sem average"]
    subdf[conds2[1]] = subdf[conds2[1]] - subdf["sem average"]
    subdf[conds2[2]] = subdf[conds2[2]] - subdf["sem average"]
    subdf[conds2[3]] = subdf[conds2[3]] - subdf["sem average"]
 
    # subtract
    # the grand mean from the values
    fig, ax = plt.subplots(figsize=(18, 6))

    x = list(range(len(subdf)))
    x_interp = np.linspace(0, len(x) - 1, num=100)

    colors = ["#4B0082", "#ff57f1", "#00BFFF", "#ffc400"]
    # plot each line with error bars
    for i, col in enumerate(conds):
        y = subdf[col].values[::-1]  # reverse the original data
        y_err = subdf["sem " + col].values[::-1]  # reverse the original error data

        f = interp1d(x, y, kind='cubic')
        y_interp = f(x_interp)

        f_err = interp1d(x, y_err, kind='cubic')
        y_err_interp = f_err(x_interp) 
        
        if plot_confi:
            y_err_interp =  y_err_interp * 1.96 
            
        plt.errorbar(x_interp, y_interp, label=col, color=colors[i], linewidth=3)
        plt.fill_between(x_interp, y_interp - y_err_interp, y_interp + y_err_interp, alpha=0.2, color=colors[i])

        # optional: plot uninterploated values onto the line
        if plot_original:
            plt.scatter(x, y, color=colors[i], s=50, zorder=3)

    if legend:
        ax.legend(fontsize=13)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(0, len(x), 1))
    ax.tick_params(axis='y', labelsize=20)

    # Customize y-axis tick labels
    ax.yaxis.set_tick_params(labelsize=20)

    ax.set_xticklabels(sphere_labels, fontsize=30)  # Set the tick labels at desired locations

    plt.savefig(pjoin(resdir, outname), dpi=600)

    # display the plot
    plt.show()


############################################################################################################################

# Fig6: smoothness

############################################################################################################################

def avg_bootstrap(df, sub, metrics, groups=None, alpha=0.05):
    if groups is None:
        groups = ["regressor"]

    def q_low(x):
        return x.quantile(alpha / 2.0)

    def q_high(x):
        return x.quantile(1.0 - alpha / 2.0)

    q_low.__name__ = "ci_low"
    q_high.__name__ = "ci_high"

    agg = (
        df.groupby(groups)[metrics]
        .agg(["mean", q_low, q_high])
        .reset_index()
    )

    # flatten multiindex: metric_stat
    agg.columns = [
        "_".join([str(part) for part in col if part != ""])
        if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    agg["subject"] = sub
    return agg


def compute_mean_and_ci(series, ci_level=0.95):
    """
    Approximate across-subject CI from subject means (normal approx).
    If you want proper bootstrap across subjects, you need the resamples,
    not just these summary stats.
    """
    series = series.dropna()
    n = series.size
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        val = series.iloc[0]
        return val, val, val

    mean_val = series.mean()
    std_val = series.std(ddof=1)
    sem_val = std_val / np.sqrt(n)

    z = 1.96 if np.isclose(ci_level, 0.95) else 1.96
    half_width = z * sem_val
    return mean_val, mean_val - half_width, mean_val + half_width



def save_metric_summaries(subject_to_df, metrics, out_dir, ci_level=0.95):
    """Save per-subject bootstrap summaries and across-subject summaries.

    Per-subject error bars are the asymmetric distances from the mean to the
    bootstrap percentile CI bounds already computed by avg_bootstrap().
    Across-subject intervals match plot_metric_all_subjects(): a normal-
    approximation CI computed from the subject-level means.
    """
    per_subject_rows = []

    for sub, df_sub in subject_to_df.items():
        for _, row in df_sub.iterrows():
            regressor = row["regressor"]

            for metric in metrics:
                mean_val = row[f"{metric}_mean"]
                ci_low = row[f"{metric}_ci_low"]
                ci_high = row[f"{metric}_ci_high"]

                per_subject_rows.append(
                    {
                        "subject": sub,
                        "regressor": regressor,
                        "metric": metric,
                        "mean": mean_val,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "error_low": mean_val - ci_low,
                        "error_high": ci_high - mean_val,
                    }
                )

    df_per_subject = pd.DataFrame(per_subject_rows)
    per_subject_path = Path(out_dir) / "smoothness_per_subject_regressor.csv"
    df_per_subject.to_csv(per_subject_path, index=False)

    across_subject_rows = []
    for metric in metrics:
        metric_df = df_per_subject[df_per_subject["metric"] == metric]

        for regressor, df_reg in metric_df.groupby("regressor"):
            mean_val, ci_low, ci_high = compute_mean_and_ci(
                df_reg["mean"], ci_level=ci_level
            )

            across_subject_rows.append(
                {
                    "regressor": regressor,
                    "metric": metric,
                    "n_subjects": df_reg["mean"].notna().sum(),
                    "mean_across_subjects": mean_val,
                    "ci_low_across_subjects": ci_low,
                    "ci_high_across_subjects": ci_high,
                    "error_low_across_subjects": mean_val - ci_low,
                    "error_high_across_subjects": ci_high - mean_val,
                }
            )

    df_across_subjects = pd.DataFrame(across_subject_rows)
    across_subjects_path = Path(out_dir) / "smoothness_across_subjects_regressor.csv"
    df_across_subjects.to_csv(across_subjects_path, index=False)

    return per_subject_path, across_subjects_path

def get_regressor_order(subject_to_df, sort_metric, include_shuffled=False):
    """
    Build a single global regressor ordering to be reused by all plots.

    Ordering rule:
    - compute across-subject mean per regressor for sort_metric
    - sort regressors by that mean (ascending)
    """
    sort_col = f"{sort_metric}_mean"

    df_all = pd.concat(subject_to_df.values(), ignore_index=True)

    if not include_shuffled:
        df_all = df_all[df_all["regressor"] != "shuffled"].copy()

    if sort_col not in df_all.columns:
        raise KeyError(f"sort metric column not found: {sort_col}")

    summary_rows = []
    for regressor, df_reg in df_all.groupby("regressor"):
        mean_val, _, _ = compute_mean_and_ci(df_reg[sort_col], ci_level=0.95)
        summary_rows.append({"regressor": regressor, "mean": mean_val})

    df_summary = pd.DataFrame(summary_rows).sort_values(by="mean", ascending=True)
    regressor_order = df_summary["regressor"].tolist()
    return regressor_order


def order_dataframe_by_regressors(df, regressor_order):
    """
    Return a copy of df ordered by regressor_order, keeping any extra regressors at the end.
    """
    df_ordered = df.copy()
    df_ordered["regressor"] = df_ordered["regressor"].astype(str)

    order_map = {name: idx for idx, name in enumerate(regressor_order)}
    missing_rank = len(regressor_order)

    df_ordered["_rank"] = df_ordered["regressor"].map(order_map).fillna(missing_rank).astype(int)
    df_ordered = df_ordered.sort_values(by=["_rank", "regressor"]).drop(columns=["_rank"])
    return df_ordered


def plot_metrics(sub, df, metrics, regressor_order=None, *, out_dir):
    """Per-subject: one figure per metric with bootstrap CIs. Uses a shared regressor ordering if provided."""
    df_shuffled = df[df["regressor"] == "shuffled"].copy()
    df_main = df[df["regressor"] != "shuffled"].copy()

    if regressor_order is not None:
        df_main = order_dataframe_by_regressors(df_main, regressor_order)

    regs = df_main["regressor"].tolist()
    x_idx = np.arange(len(regs))

    for met in metrics:
        y_vals = df_main[f"{met}_mean"].values
        ci_low = df_main[f"{met}_ci_low"].values
        ci_high = df_main[f"{met}_ci_high"].values

        yerr = np.vstack([y_vals - ci_low, ci_high - y_vals])

        fig_width = max(8.0, 0.5 * len(regs))
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))

        ax.errorbar(
            x_idx,
            y_vals,
            yerr=yerr,
            fmt="o",
            linestyle="none",
            color="black",
            ecolor="black",
            capsize=3,
        )

        # shuffled baseline (per metric)
        if len(df_shuffled) > 0 and f"{met}_mean" in df_shuffled.columns:
            shuffled_mean = df_shuffled[f"{met}_mean"].values[0]
            shuffled_low = df_shuffled[f"{met}_ci_low"].values[0]
            shuffled_high = df_shuffled[f"{met}_ci_high"].values[0]

            ax.axhspan(
                shuffled_low,
                shuffled_high,
                color="red",
                alpha=0.15,
                label="shuffled ci",
            )
            ax.axhline(shuffled_low, color="red", linestyle="--", linewidth=1.0)
            ax.axhline(shuffled_high, color="red", linestyle="--", linewidth=1.0)
            ax.axhline(
                shuffled_mean,
                color="red",
                linestyle="-",
                linewidth=1.2,
                label="shuffled mean",
            )
            ax.legend(frameon=False)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(regs, rotation=90)
        ax.set_ylabel(met)
        ax.set_title(f"{sub}: {met}")

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out_path = Path(out_dir) / f"{sub}_{met}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_two_metrics_subjectwise(sub, df, reltype, regressor_order=None, *, out_dir):
    """
    For each subject: plot wtv_sigma_3mm_mean and wtv_rel_*_sigma_3mm_rel_mean
    as points on the same axes (no CIs), using shared regressor ordering if provided.
    """
    met_a = "wtv_sigma_3mm"
    met_b = f"wtv_rel_{reltype}_g1.0_sigma_3mm_rel"

    col_a = f"{met_a}_mean"
    col_b = f"{met_b}_mean"

    df_main = df[df["regressor"] != "shuffled"].copy()

    if regressor_order is not None:
        df_main = order_dataframe_by_regressors(df_main, regressor_order)

    regs = df_main["regressor"].tolist()
    x_idx = np.arange(len(regs))

    y_a = df_main[col_a].values
    y_b = df_main[col_b].values

    fig_width = max(10.0, 0.6 * len(regs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    ax.plot(
        x_idx,
        y_a,
        marker="o",
        linewidth=0,
        linestyle="none",
        label=met_a,
        color=okabe_ito[0],
    )
    ax.plot(
        x_idx,
        y_b,
        marker="s",
        linewidth=0,
        linestyle="none",
        label=met_b,
        color=okabe_ito[1],
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(regs, rotation=90, fontsize = 15)
    ax.set_ylabel("metric value")
    ax.set_title(f"{sub}: {met_a} vs {met_b}")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_dir) / f"{sub}_wtv_sigma3mm_and_rel.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_metric_all_subjects(
    subject_to_df,
    shuf_to_df,
    metric,
    regressor_order,
    ci_level=0.95,
    *,
    out_dir,
):
    """
    All subjects: one figure per metric.
    Uses the shared regressor_order (computed from the chosen sort metric).
    Shows across-subject mean + CI per regressor,
    plus shuffled baseline CI band (no mean line).
    """
    col = f"{metric}_mean"

    df_all = pd.concat(subject_to_df.values(), ignore_index=True)
    df_main = df_all[df_all["regressor"] != "shuffled"].copy()

    summary_rows = []
    for regressor, df_reg in df_main.groupby("regressor"):
        mean_val, low_val, high_val = compute_mean_and_ci(df_reg[col], ci_level)
        summary_rows.append(
            {
                "regressor": regressor,
                "n_subjects": df_reg[col].notna().sum(),
                "mean": mean_val,
                "low": low_val,
                "high": high_val,
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # apply shared ordering (not sorting by the plotted metric anymore)
    df_summary = order_dataframe_by_regressors(df_summary, regressor_order)

    regs = df_summary["regressor"].tolist()
    x_idx = np.arange(len(regs))

    y = df_summary["mean"].values
    yerr = np.vstack(
        [
            y - df_summary["low"].values,
            df_summary["high"].values - y,
        ]
    )

    fig_width = max(10.0, 0.6 * len(regs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    ax.plot(
        x_idx,
        y,
        marker="o",
        linestyle="none",
        color="black",
    )

    ax.errorbar(
        x_idx,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        alpha=0.6,
    )

    # shuffled CI band
    shuffled_vals = []
    key = f"{metric}_mean"
    for sub, shuf_series in shuf_to_df.items():
        if key in shuf_series:
            shuffled_vals.append(shuf_series[key])

    shuffled_summary = None
    if len(shuffled_vals) > 0:
        shuffled_vals = pd.Series(shuffled_vals)
        shuffled_mean, shuffled_low, shuffled_high = compute_mean_and_ci(
            shuffled_vals, ci_level
        )
        shuffled_summary = {
            "regressor": "shuffled",
            "n_subjects": shuffled_vals.notna().sum(),
            "mean": shuffled_mean,
            "low": shuffled_low,
            "high": shuffled_high,
        }

        if len(shuffled_vals) > 1:
            ax.axhspan(
                shuffled_low,
                shuffled_high,
                color="red",
                alpha=0.15,
                label="shuffled ci",
            )
            ax.axhline(shuffled_low, color="red", linestyle="--", linewidth=1.0)
            ax.axhline(shuffled_high, color="red", linestyle="--", linewidth=1.0)

    # Save the exact across-subject values used in this plot to a separate CSV.
    df_export = df_summary.rename(
        columns={
            "mean": "mean_across_subjects",
            "low": "ci_low_across_subjects",
            "high": "ci_high_across_subjects",
        }
    ).copy()

    df_export["error_low_across_subjects"] = (
        df_export["mean_across_subjects"] - df_export["ci_low_across_subjects"]
    )
    df_export["error_high_across_subjects"] = (
        df_export["ci_high_across_subjects"] - df_export["mean_across_subjects"]
    )

    if shuffled_summary is not None:
        shuffled_export = pd.DataFrame(
            [
                {
                    "regressor": shuffled_summary["regressor"],
                    "n_subjects": shuffled_summary["n_subjects"],
                    "mean_across_subjects": shuffled_summary["mean"],
                    "ci_low_across_subjects": shuffled_summary["low"],
                    "ci_high_across_subjects": shuffled_summary["high"],
                    "error_low_across_subjects": (
                        shuffled_summary["mean"] - shuffled_summary["low"]
                    ),
                    "error_high_across_subjects": (
                        shuffled_summary["high"] - shuffled_summary["mean"]
                    ),
                }
            ]
        )
        df_export = pd.concat([df_export, shuffled_export], ignore_index=True)

    csv_path = Path(out_dir) / f"All_subjects_{metric}.csv"
    df_export.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(regs, rotation=90, fontsize=15)
    ax.set_ylabel(metric)
    ax.set_title(f"all subjects: {metric}")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_dir) / f"All_subjects_{metric}.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    return out_path



def create_rainbow_colormap(num_colors):
    # Generate a list of evenly spaced values from 0 to 1 (excluding 0)
    values = np.linspace(0, 1, num_colors - 1)

    # Create a list of rainbow colors based on the hue values
    rainbow_colors = [(value, plt.cm.rainbow(value)) for value in values]

    # Add black as the first color
    rainbow_colors.insert(0, (0, 'black'))

    # Create the colormap using LinearSegmentedColormap
    cmap_name = f'rainbow_{num_colors}'
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, rainbow_colors, N=num_colors)

    return colormap
