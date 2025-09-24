import concurrent.futures
import logging
import os
import pickle
from itertools import repeat
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def get_logger(logging_file_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"compute_lag_mi_df_map")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(logging_file_path)
    file_handler.setLevel(logging.INFO)  # Save INFO and above in file

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Show DEBUG and above in console

        # Create formatter and add it to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def get_logged_lags(log_path: str) -> List[int]:
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Used lags:" in line:
                try:
                    return sorted(
                        map(
                            int,
                            line.split("- INFO -")[1]
                            .strip()
                            .split(":")[1]
                            .strip()
                            .strip("[]")
                            .split(","),
                        )
                    )
                except Exception:
                    continue
        return []


def log_current_lags(lags: List[int], logger: logging.Logger) -> None:
    logger.info(f"Used lags: {lags}")


def get_mi_df(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    cols = df.columns
    cols_num = len(cols)
    mi_matrix = np.zeros((cols_num, cols_num))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if lag == 0 and j > i:
                continue

            x = df[col_i].values
            y = np.roll(df[col_j].values, -lag)

            valid_length = len(x) - lag
            x_valid = x[:valid_length]
            y_valid = y[:valid_length]

            mi = mutual_info_score(x_valid, y_valid)
            mi_matrix[i, j] = mi

            if lag == 0:
                mi_matrix[j, i] = mi

    not_lagged_features = [f"{c}_lag0" for c in cols]
    lagged_features = [f"{c}_lag{lag}" for c in cols]

    mi_df = pd.DataFrame(mi_matrix, index=not_lagged_features, columns=lagged_features)

    return mi_df


def get_lob_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if (c.startswith("ASKs") or c.startswith("BIDs"))]]

    return df


def get_binned_df(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    binned_df = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        binned_df[col] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")

    return binned_df


def get_mi_dfs_from_path(path: Path, lag: int, n_bins: int) -> pd.DataFrame:
    lob_df = get_lob_df(path)
    lob_df = get_binned_df(lob_df, n_bins)

    return get_mi_df(lob_df, lag)


def get_weighted_lag_mi_df_map(
    results: List[pd.DataFrame], lags: List[int], total_samples: int
) -> Dict[int, pd.DataFrame]:
    return {lag: mi_df / total_samples for lag, mi_df in zip(lags, results)}


def load_checkpoint_if_exists(filepath: Path) -> Dict[str, Dict[int, pd.DataFrame]]:
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)["per_file"]

    return {}


def is_already_finished(log_path: str, current_lags: List[int]) -> bool:
    if not os.path.exists(log_path):
        return False

    with open(log_path, "r") as f:
        lines = f.readlines()

    if not any("Final save complete. Processing finished." in line for line in lines):
        return False

    index_line_last_complete = max(
        (
            i
            for i, line in enumerate(lines)
            if "Final save complete. Processing finished." in line
        ),
        default=-1,
    )
    index_line_last_new_lags = max(
        (
            i
            for i, line in enumerate(lines)
            if "New lags detected:" in line and i > index_line_last_complete
        ),
        default=-1,
    )
    if index_line_last_complete < index_line_last_new_lags:
        return False

    previous_lags = get_logged_lags(log_path)

    return set(current_lags).issubset(previous_lags)


def get_last_saved_index_from_logs(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 0

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        if "Checkpoint saved after processing file_" in line:
            try:
                index = int(line.split("file_")[-1].split(".")[0])
                return index + 1

            except Exception:
                continue
    return 0


def get_file_lag_mi_map(
    lob_files: list[Path],
    checkpoint_path: Path,
    lags: list[int],
    num_bins: int,
    logger: logging.Logger,
    num_files_for_checkpoint: int = 5,
) -> dict[str, dict[int, pd.DataFrame]]:
    file_results = load_checkpoint_if_exists(checkpoint_path)
    if file_results:
        logger.info(f"Loading checkpoint from {checkpoint_path}")

    for i, file in enumerate(lob_files):
        file_key = str(file)
        existing_lags = file_results.get(file_key, {}).keys()
        needed_lags = list(set(lags) - set(existing_lags))

        if needed_lags:
            logger.info(f"Processing {file.name} (index {i}), new lags: {needed_lags}")
            with concurrent.futures.ProcessPoolExecutor(6) as executor:
                results = list(
                    executor.map(
                        get_mi_dfs_from_path,
                        repeat(file),
                        needed_lags,
                        repeat(num_bins),
                    )
                )
            lag_map = dict(zip(needed_lags, results))
            if file_key not in file_results:
                file_results[file_key] = {}
            file_results[file_key].update(lag_map)

        if i % num_files_for_checkpoint == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"per_file": file_results}, f)
            logger.info(f"Checkpoint saved after processing file_{i}.")

    return file_results


def get_averaged_file_lag_mi_map(
    file_results: dict[str, dict[int, pd.DataFrame]], lags: list[int]
) -> dict[int, pd.DataFrame]:
    final_result = {}
    for lag in lags:
        sum_df, count = None, 0
        for lag_map in file_results.values():
            if lag in lag_map:
                df = lag_map[lag]
                sum_df = df if sum_df is None else sum_df + df
                count += 1
        if sum_df is not None:
            final_result[lag] = sum_df / count
    return final_result


def compute_lag_mi_df_map(
    lob_files_paths: list[Path],
    lags: list[int],
    logging_file_path: Path,
    saving_path: Path,
    num_bins: int,
    num_files_for_checkpoint: int,
) -> None:
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    logging_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger(logging_file_path)

    current_lags = sorted(lags)
    if is_already_finished(logging_file_path, current_lags):
        logger.info(
            "Processing already finished for all required lags. Skipping matrix computation."
        )
        return

    logged_lags = get_logged_lags(logging_file_path)
    if not logged_lags:
        logger.info(f"No previous lag info found. Logging initial lags: {current_lags}")
        log_current_lags(current_lags, logger)

    else:
        new_lags = list(set(current_lags) - set(logged_lags))

        if new_lags:
            logger.info(f"New lags detected: {new_lags}")
            combined_lags = sorted(set(current_lags).union(set(logged_lags)))
            log_current_lags(combined_lags, logger)
            current_lags = combined_lags

    file_lag_mi_map = get_file_lag_mi_map(
        lob_files_paths,
        saving_path,
        current_lags,
        num_bins,
        logger,
        num_files_for_checkpoint=num_files_for_checkpoint,
    )
    avg_file_lag_mi_map = get_averaged_file_lag_mi_map(file_lag_mi_map, current_lags)

    with open(saving_path, "wb") as f:
        pickle.dump({"per_file": file_lag_mi_map, "final": avg_file_lag_mi_map}, f)

    log_current_lags(current_lags, logger)
    logger.info("Final save complete. Processing finished.")
