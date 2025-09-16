import numpy as np
import pandas as pd


def get_spatiotemporal_index(lag_mi_df_map: dict[int, pd.DataFrame]) -> list[str]:
    """
    Build the global index of the big correlation matrix
    using the existing labels in corr_matrices.
    Lags are ordered decreasingly, with lag=0 last.
    """
    labels: list[str] = []

    # sort lag keys: descending, but put 0 at the end
    lags = sorted([lag for lag in lag_mi_df_map if lag != 0], reverse=True)
    lags.append(0)

    for lag in lags:
        labels.extend(lag_mi_df_map[lag].columns.tolist())
    return labels


def get_initialized_spatiotemporal_df(labels: list[str]) -> pd.DataFrame:
    size = len(labels)
    return pd.DataFrame(np.zeros((size, size)), index=labels, columns=labels)


def get_filled_spatiotemporal_df(
    spatiotemporal_df: pd.DataFrame, lag_mi_df_map: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Fill the big correlation matrix with lagged correlation blocks.
    Assumes lag_mi_df_map[lag] has proper row/col labels like 'feat_lag{lag}'.
    """
    lag0_labels = lag_mi_df_map[0].columns.tolist()

    spatiotemporal_df.loc[lag0_labels, lag0_labels] = lag_mi_df_map[0].to_numpy()

    for lag in sorted([l for l in lag_mi_df_map if l != 0], reverse=True):
        df = lag_mi_df_map[lag]
        lag_labels = df.columns.tolist()

        spatiotemporal_df.loc[lag0_labels, lag_labels] = df.to_numpy()
        spatiotemporal_df.loc[lag_labels, lag0_labels] = df.T.to_numpy()

    return spatiotemporal_df


def get_spatiotemporal_df(
    corr_matrices: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a big symmetric correlation matrix from pruned lagged correlation matrices.
    Row/column labels are taken directly from corr_matrices.
    """
    labels = get_spatiotemporal_index(corr_matrices)
    big_df = get_initialized_spatiotemporal_df(labels)
    return get_filled_spatiotemporal_df(big_df, corr_matrices)
