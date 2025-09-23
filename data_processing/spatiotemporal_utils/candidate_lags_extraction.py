from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline


def get_auto_mutual_info_df(lag_mi_df_map: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    records = []

    for lag, corr_df in lag_mi_df_map.items():
        for col in corr_df.columns:
            # Extract the feature name by removing '_lag0'
            feature = col.replace(f"_lag{lag}", "")
            row = f"{feature}_lag0"

            if row in corr_df.index:
                autocorr = corr_df.loc[row, col]

                # Detect group
                if feature.startswith("ASKs"):
                    group = "ASK"
                    level = feature.replace("ASKs", "Level ")
                elif feature.startswith("BIDs"):
                    group = "BID"
                    level = feature.replace("BIDs", "Level ")
                else:
                    group = "OTHER"
                    level = feature

                records.append(
                    {
                        "Lag": lag,
                        "Feature": feature,
                        "Correlation": autocorr,
                        "Group": group,
                        "Level": level,
                    }
                )

    plot_df = pd.DataFrame(records)

    return plot_df


def save_auto_mutual_info_plot(plot_df: pd.DataFrame, path: Path) -> None:

    max_lag = plot_df["Lag"].max()

    # Get unique levels and map to Viridis palette
    unique_levels = sorted(plot_df["Level"].unique(), key=lambda x: int(x.split()[-1]))
    palette = sns.color_palette("viridis", n_colors=len(unique_levels))
    palette_dict = dict(zip(unique_levels, palette))

    # Plot using seaborn.relplot with 'Level' as hue and Viridis palette
    g = sns.relplot(
        data=plot_df,
        x="Lag",
        y="Correlation",
        hue="Level",
        kind="line",
        marker="o",
        row="Group",
        facet_kws={"sharey": True},
        palette=palette_dict,
        height=4,
        aspect=2,  # Makes the plot wider
    )

    # Enable grid on each subplot
    for ax in g.axes.flat:
        ax.grid(True)
        # ax.set_ylim(0, 7)
        ax.set_xlim(0, max_lag)

    g.fig.suptitle(
        "Auto Mutual Information of ASK and BID Features at Different Lags", y=1.02
    )
    g.set_titles(row_template="{row_name} Features")
    g.set_axis_labels("Lag [# Updates]", "Mutual Information [-]")
    g.tight_layout()

    plt.savefig(path / "auto_mi_vs_lag.png", format="png", dpi=300)
    plt.savefig(path / "auto_mi_vs_lag.pdf", format="pdf")
    plt.close()


def get_prefix(name: str) -> str:
    if name.startswith("ASKs"):
        return "ASK"
    elif name.startswith("BIDs"):
        return "BID"
    else:
        return "OTHER"


def get_aggregated_mutual_info_df_stats(
    lag_mi_df_map: Dict[int, pd.DataFrame], category_fn: callable
) -> pd.DataFrame:
    stats = []

    for lag, df in lag_mi_df_map.items():
        selected_vals = []

        for i in df.index:
            base_feat = i.replace("_lag0", "")
            for j in df.columns:
                lagged_feat = j.replace(f"_lag{lag}", "")
                val = df.loc[i, j]

                if category_fn(base_feat, lagged_feat):
                    selected_vals.append(val)

        stats.append(
            {
                "Lag": lag,
                "Mean": pd.Series(selected_vals).mean(),
                "Std": pd.Series(selected_vals).std(),
            }
        )

    return pd.DataFrame(stats)


def save_mutual_info_stats_plot(
    path: Path,
    filename_no_extension: str,
    data_series: List[Dict[str, object]],
    title: str,
    xlabel: str = "Lag",
    ylabel: str = "Correlation",
    figsize: tuple = (10, 5),
) -> None:
    plt.figure(figsize=figsize)

    for series in data_series:
        df: pd.DataFrame = series["df"]
        label: str = series["label"]
        color: str = series["color"]

        plt.plot(df["Lag"], df["Mean"], label=label, color=color)
        plt.fill_between(
            df["Lag"],
            df["Mean"] - df["Std"],
            df["Mean"] + df["Std"],
            alpha=0.3,
            color=color,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path / f"{filename_no_extension}.pdf", format="pdf")
    plt.close()


def get_fitted_spline(
    x_values: np.ndarray, y_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    spline = UnivariateSpline(x_values, y_values, s=0)
    x_new = np.arange(min(x_values), max(x_values), 1)
    y_new = spline(x_new)

    return x_new, y_new


def get_normalized_finite_difference_derivative(
    x_values: np.ndarray, y_values: np.ndarray
) -> np.ndarray:
    dx = np.diff(x_values)
    dy = np.diff(y_values)
    slope = np.abs(dy / dx)

    slope = slope / slope.max()

    # insert a zero at the end of the slope array to match the length of x_new
    slope = np.append(slope, slope[-1])

    return slope


def get_cumulative_distribution(
    x_values: np.ndarray, y_values: np.ndarray
) -> np.ndarray:
    area = np.trapz(y_values, x_values)
    y_norm = y_values / area

    cdf = cumulative_trapezoid(y_norm, x_values, initial=0)
    return cdf


def get_x_values_for_equally_y_spaced_division(
    x_values: np.ndarray, y_values: np.ndarray, n_intervals: int
) -> np.ndarray:
    target_cdf = np.linspace(0, 1, n_intervals + 1)
    x_interp = np.interp(target_cdf, y_values, x_values)

    x_rounded = np.round(x_interp).astype(int)

    x_unique = np.unique(x_rounded)
    print("Unique X values after rounding:", x_unique)

    if len(x_unique) < n_intervals + 1:
        x_unique = pad_increasing_array(x_unique, n_intervals + 1)

    return x_unique


def pad_increasing_array(arr, L):
    arr = np.array(arr)
    if len(arr) >= L:
        return arr[:L]  # Truncate if longer than L

    # Start from arr[0] and incrementally fill in numbers
    result = list(arr)
    i = 0
    while len(result) < L:
        if i + 1 < len(result):
            expected_next = result[i] + 1
            if result[i + 1] != expected_next:
                result.insert(i + 1, expected_next)
            else:
                i += 1
        else:
            result.append(result[-1] + 1)
    return np.array(result)


def get_intervals_from_sequence(
    sequence: np.ndarray,
) -> List[Tuple[int, int]]:
    if sequence[0] != 0:
        sequence = np.insert(sequence, 0, 0)

    intervals = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
    return intervals


def get_x_values_for_mean_y_spaced_intervals(
    x_values: np.ndarray,
    y_values: np.ndarray,
    end_excluded_intervals: List[Tuple[int, int]],
) -> np.ndarray:
    candidates = []

    for start, end in end_excluded_intervals:
        mask = (x_values >= start) & (x_values < end)
        x_in_bin = x_values[mask]
        y_in_bin = y_values[mask]

        if len(x_in_bin) == 0:
            candidates.append(None)
            continue

        y_min, y_max = y_in_bin[0], y_in_bin[-1]
        target_y = (y_min + y_max) / 2

        closest_idx = np.argmin(np.abs(y_in_bin - target_y))
        candidate = x_in_bin[closest_idx]

        candidates.append(candidate)

    sorted([int(c) for c in candidates], reverse=True)
    return candidates


def get_candidates_lags(
    lag_mi_df_map: dict[int, pd.DataFrame],
    num_lag_used: int,
    initial_lags: list[int] | None = None,
    images_folder_path: Path | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    if initial_lags is not None:
        lag_mi_df_map = {
            lag: df for lag, df in lag_mi_df_map.items() if lag in initial_lags
        }

    all_vals = lambda i, j: True
    auto_vals = lambda i, j: i == j
    cross_vals = lambda i, j: i != j

    overall_df = get_aggregated_mutual_info_df_stats(lag_mi_df_map, all_vals)
    overall_df = overall_df.sort_values(by="Lag")

    auto_df = get_aggregated_mutual_info_df_stats(lag_mi_df_map, auto_vals)
    cross_df = get_aggregated_mutual_info_df_stats(lag_mi_df_map, cross_vals)

    mean_df = auto_df.merge(cross_df, on="Lag", suffixes=("_auto", "_cross"))
    mean_df["Mean"] = (mean_df["Mean_auto"] + mean_df["Mean_cross"]) / 2
    mean_df = mean_df[["Lag", "Mean"]].sort_values(by="Lag")

    if images_folder_path is not None:
        auto_mutual_info_df = get_auto_mutual_info_df(lag_mi_df_map)
        images_folder_path.mkdir(parents=True, exist_ok=True)

        save_auto_mutual_info_plot(
            auto_mutual_info_df,
            images_folder_path,
        )

        same_prefix_vals = lambda i, j: i != j and get_prefix(i) == get_prefix(j)
        diff_prefix_vals = lambda i, j: i != j and get_prefix(i) != get_prefix(j)

        auto_df = get_aggregated_mutual_info_df_stats(lag_mi_df_map, auto_vals)
        cross_df = get_aggregated_mutual_info_df_stats(lag_mi_df_map, cross_vals)
        same_prefix_df = get_aggregated_mutual_info_df_stats(
            lag_mi_df_map, same_prefix_vals
        )
        diff_prefix_df = get_aggregated_mutual_info_df_stats(
            lag_mi_df_map, diff_prefix_vals
        )

        save_mutual_info_stats_plot(
            images_folder_path,
            "overall_mi_vs_lag",
            data_series=[
                {"df": overall_df, "label": "Mutual Information", "color": "blue"},
            ],
            title="Average Mutual Information at each lag",
        )
        save_mutual_info_stats_plot(
            images_folder_path,
            "auto_cross_mi_vs_lag",
            data_series=[
                {"df": auto_df, "label": "Auto MI", "color": "green"},
                {"df": cross_df, "label": "Cross MI", "color": "orange"},
            ],
            title="Auto vs Cross Mutual Information at each lag",
        )
        save_mutual_info_stats_plot(
            images_folder_path,
            "same_diff_prefix_mi_vs_lag",
            data_series=[
                {
                    "df": same_prefix_df,
                    "label": "Same Prefix (ASK↔ASK, BID↔BID)",
                    "color": "purple",
                },
                {
                    "df": diff_prefix_df,
                    "label": "Different Prefix (ASK↔BID)",
                    "color": "red",
                },
            ],
            title="Cross Mutual Information by prefix grouping at each lag",
        )

    spline_x, spline_y = get_fitted_spline(
        mean_df["Lag"].values,
        mean_df["Mean"].values,
    )

    slope_values = get_normalized_finite_difference_derivative(spline_x, spline_y)
    cdf = get_cumulative_distribution(spline_x, slope_values)
    print("CDF values:", cdf)

    x_values = get_x_values_for_equally_y_spaced_division(
        spline_x, cdf, n_intervals=num_lag_used
    )
    print("X values for equally spaced intervals:", x_values)

    end_excluded_intervals = get_intervals_from_sequence(x_values)

    candidates_x_values = get_x_values_for_mean_y_spaced_intervals(
        spline_x, spline_y, end_excluded_intervals
    )

    return end_excluded_intervals, candidates_x_values
