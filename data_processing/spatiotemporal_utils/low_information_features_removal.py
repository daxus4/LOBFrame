import pandas as pd


def get_lagged_features_means(mi_df: pd.DataFrame) -> pd.Series:
    return mi_df.mean(axis=0)


def get_mi_df_without_pruned_features(
    corr_df: pd.DataFrame, prev_avg_corr: float
) -> tuple[pd.DataFrame, float]:
    """
    Iteratively remove the feature with the lowest mean correlation until
    removing another feature would make the average correlation > prev_avg_corr.

    Returns the pruned correlation dataframe and its average correlation.
    """
    working_df = corr_df.copy()

    while working_df.shape[1] > 1:  # need at least 2 features
        # compute candidate feature to remove
        feature_means = get_lagged_features_means(working_df)
        feature_to_remove = feature_means.idxmin()

        # simulate removing it
        candidate_df = working_df.drop(columns=feature_to_remove)
        candidate_avg = candidate_df.to_numpy().mean()

        # stop if removing it would push us over the threshold
        if candidate_avg > prev_avg_corr:
            break

        # otherwise, actually remove it
        working_df = candidate_df

    return working_df, working_df.to_numpy().mean()


def get_lag_mi_df_map_without_pruned_features(
    lag_mi_df_map: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """
    Perform iterative pruning of features across lagged correlation matrices.

    Parameters
    ----------
    corr_matrices : dict[int, pd.DataFrame]
        Dictionary where keys are lags and values are correlation matrices.

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary with the same keys but pruned correlation matrices.
    """
    # Sort by lag
    sorted_lags = sorted(lag_mi_df_map.keys())
    result: dict[int, pd.DataFrame] = {}

    # Start with lag=0
    lag0 = sorted_lags[0]
    result[lag0] = lag_mi_df_map[lag0].copy()
    prev_avg_mi = result[lag0].to_numpy().mean()

    # Process subsequent lags
    for lag in sorted_lags[1:]:
        pruned_df, avg_mi = get_mi_df_without_pruned_features(
            lag_mi_df_map[lag], prev_avg_mi
        )
        result[lag] = pruned_df
        prev_avg_mi = avg_mi

    return result


def get_lag_mi_df_map_without_low_correlated_features(
    lag_mi_df_map: dict[int, pd.DataFrame], selected_lags: list[int] | None
) -> dict[int, pd.DataFrame]:
    if selected_lags is not None:
        lag_mi_df_map = {
            lag: df for lag, df in lag_mi_df_map.items() if lag in selected_lags
        }

    pruned_lag_mi_df_map = get_lag_mi_df_map_without_pruned_features(lag_mi_df_map)

    return pruned_lag_mi_df_map
