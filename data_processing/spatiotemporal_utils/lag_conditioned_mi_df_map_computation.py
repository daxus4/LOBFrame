"""
mi_lagged_matrix.py — Revised per user clarification

Key corrections:
  • Lagging is applied BEFORE class filtering (Interpretation A):
      - For each file, we create (X_t, X_{t+lag}) pairs based on original row order.
      - We assign each pair to class = class(t).
  • For each class and lag we compute ONLY the N×N cross-lag MI matrix:
        M[i,j] = MI( feature_i(t), feature_j(t+lag) )
  • No 2N×2N matrix, no masking.
  • Persistence, resumability, parallelization, and logging retained.

Usage example:
    python mi_lagged_matrix.py \
      --input-folder ./data_csvs \
      --output-folder ./mi_results \
      --class-column label \
      --lags 0 1 2 3 \
      --n-jobs 8
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import glob
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def setup_logger(log_path: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mi_lagged")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        fh.setLevel(level)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------
# IO Handler
# ---------------------------------------------------------------------


@dataclass
class IOHandler:
    input_folder: str
    output_folder: str
    class_column: str
    logger: logging.Logger

    def discover_files(self) -> List[str]:
        patterns = [
            os.path.join(self.input_folder, "*.csv"),
            os.path.join(self.input_folder, "*.parquet"),
        ]
        files = []
        for p in patterns:
            files.extend(sorted(glob.glob(p)))
        self.logger.info(f"Discovered {len(files)} files")
        return files

    def load_dataframe(self, path: str) -> pd.DataFrame:
        _, ext = os.path.splitext(path)
        if ext.lower() == ".csv":
            df = pd.read_csv(path)
        elif ext.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        if self.class_column not in df.columns:
            raise KeyError(f"Class column '{self.class_column}' not found in {path}")
        return df

    def ensure_output(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def result_path(self, class_value: str, lag: int) -> str:
        safe = str(class_value).replace("/", "_")
        return os.path.join(self.output_folder, f"mi_{safe}_lag_{lag}.npz")


# ---------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------
class DataProcessor:
    def __init__(
        self, df: pd.DataFrame, return_col: str, threshold: float, bins_number: int
    ):
        self.df = df
        self.return_col = return_col
        self.threshold = threshold
        self.bins_number = bins_number
        self.size_columns = self.get_size_columns()

    def filter_useless_columns(self):
        self.df = self.df[self.size_columns + [self.return_col]]
        return self

    def bin_size_columns(self):
        for col in self.size_columns:
            self.df[col] = pd.qcut(
                self.df[col], q=self.bins_number, labels=False, duplicates="drop"
            )
        return self

    def classify_return_column(self):
        self.df[self.return_col] = np.where(
            self.df[self.return_col] > self.threshold,
            2,
            np.where(self.df[self.return_col] < -self.threshold, 0, 1),
        )
        return self

    def get_size_columns(
        self, size_col_prefixes: List[str] = ["BIDs", "ASKs"]
    ) -> List[str]:
        return [
            col
            for col in self.df.columns
            if any(col.startswith(prefix) for prefix in size_col_prefixes)
        ]

    def get(self):
        return self.df


# ---------------------------------------------------------------------
# Mutual Information computation
# ---------------------------------------------------------------------


@dataclass
class MIComputer:
    logger: logging.Logger = field(default=logging.getLogger("mi_lagged"))

    def _mi_single(self, x: np.ndarray, y: np.ndarray) -> float:
        X = x.reshape(-1, 1)
        return mutual_info_score(X, y)

    def pairwise_mi(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        N = X.shape[1]
        M = Y.shape[1]

        def compute_j(j):
            y = Y[:, j]
            row = np.zeros(N)
            for i in range(N):
                try:
                    row[i] = self._mi_single(X[:, i], y)
                except Exception as e:
                    self.logger.exception("MI error (%d,%d): %s", i, j, e)
                    row[i] = np.nan
            return row

        cols = Parallel(n_jobs=-1, prefer="threads")(
            delayed(compute_j)(j) for j in range(M)
        )
        return np.column_stack(cols)


# ---------------------------------------------------------------------
# Job Manager
# ---------------------------------------------------------------------


class JobManager:
    def __init__(
        self,
        io: IOHandler,
        mi: MIComputer,
        lags: List[int],
        n_jobs: int,
        logger: logging.Logger,
        threshold: float,
        bins_number: int,
    ):
        self.io = io
        self.mi = mi
        self.lags = sorted(lags)
        self.n_jobs = n_jobs
        self.logger = logger
        self.threshold = threshold
        self.bins_number = bins_number

    # -----------------------------------------------------------
    # Create (X_t, X_{t+lag}, class(t)) pairs BEFORE class filtering
    # -----------------------------------------------------------
    def build_pairs_for_file(self, df: pd.DataFrame, lag: int) -> pd.DataFrame:
        if lag == 0:
            X0 = df
            Xlag = df
            cls = df[self.io.class_column].values
        else:
            X0 = df.iloc[:-lag]
            Xlag = df.iloc[lag:]
            cls = df[self.io.class_column].iloc[:-lag].values

        # restrict to numeric columns except class
        feature_cols = [c for c in df.columns if c != self.io.class_column]
        X0_vals = X0[feature_cols].values
        Xlag_vals = Xlag[feature_cols].values

        valid = ~(np.isnan(X0_vals).any(axis=1) | np.isnan(Xlag_vals).any(axis=1))

        out = pd.DataFrame({"class": cls[valid]})
        for i, c in enumerate(feature_cols):
            out[f"{c}_t"] = X0_vals[valid, i]
            out[f"{c}_lag"] = Xlag_vals[valid, i]
        return out, feature_cols

    # -----------------------------------------------------------
    # Aggregate all (X_t,X_{t+lag}) pairs per class across files
    # -----------------------------------------------------------
    def aggregate_pairs(
        self, files: List[str], lag: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        class_groups: Dict[str, Dict[str, list]] = {}
        feature_cols: Optional[List[str]] = None

        for path in files:
            try:
                df = self.io.load_dataframe(path)

            except Exception as e:
                self.logger.exception("Skipping %s: load error %s", path, e)
                continue

            df = (
                DataProcessor(
                    df, self.io.class_column, self.threshold, self.bins_number
                )
                .filter_useless_columns()
                .bin_size_columns()
                .classify_return_column()
                .get()
            )

            try:
                pairs, cols = self.build_pairs_for_file(df, lag)
            except Exception as e:
                self.logger.exception("Skipping %s: pair-build error %s", path, e)
                continue

            if feature_cols is None:
                feature_cols = cols

            for cls_val, grp in pairs.groupby("class"):
                entry = class_groups.setdefault(str(cls_val), {"X": [], "Y": []})
                # X_t columns
                X = grp[[f"{c}_t" for c in feature_cols]].values
                Y = grp[[f"{c}_lag" for c in feature_cols]].values
                entry["X"].append(X)
                entry["Y"].append(Y)

        out = {}
        for cls, data in class_groups.items():
            X = np.vstack(data["X"]) if data["X"] else np.zeros((0, len(feature_cols)))
            Y = np.vstack(data["Y"]) if data["Y"] else np.zeros((0, len(feature_cols)))
            out[cls] = {"X": X, "Y": Y, "cols": feature_cols}
        return out

    # -----------------------------------------------------------
    # Run tasks
    # -----------------------------------------------------------
    def run_all(self):
        files = self.io.discover_files()
        self.io.ensure_output()

        tasks = []
        for lag in self.lags:
            tasks.append((lag, files))

        Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._process_lag)(lag, files) for (lag, files) in tasks
        )

    def _process_lag(self, lag: int, files: List[str]):
        self.logger.info(f"Processing lag = {lag}")
        groups = self.aggregate_pairs(files, lag)

        for cls, data in groups.items():
            out_path = self.io.result_path(cls, lag)
            if os.path.exists(out_path):
                self.logger.info(f"Skipping class={cls} lag={lag}, already exists.")
                continue

            X = data["X"]
            Y = data["Y"]
            cols = data["cols"]

            self.logger.info(f"Class={cls}, lag={lag}, samples={X.shape[0]}")
            if X.shape[0] < 5:
                self.logger.warning(f"Too few samples for class={cls} lag={lag}")

            try:
                mi_matrix = self.mi.pairwise_mi(X, Y)
                meta = {"columns": cols, "lag": lag, "num_samples": X.shape[0]}
                np.savez_compressed(out_path, mi_matrix=mi_matrix, meta=meta)
                self.logger.info(f"Saved: {out_path}")
            except Exception as e:
                self.logger.exception(
                    f"Failed computing MI for class={cls} lag={lag}: {e}"
                )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-folder", required=True)
    p.add_argument("--output-folder", required=True)
    p.add_argument("--class-column", required=True)
    p.add_argument("--lags", nargs="+", type=int, required=True)
    p.add_argument(
        "--estimator", choices=["regression", "classification"], default="regression"
    )
    p.add_argument("--n-neighbors", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--log-file", default="mi_lagged.log")
    p.add_argument("--threshold", type=float, default=99)
    p.add_argument("--bins-number", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)

    io = IOHandler(args.input_folder, args.output_folder, args.class_column, logger)
    mi = MIComputer(
        logger=logger,
    )
    mgr = JobManager(
        io, mi, args.lags, args.n_jobs, logger, args.threshold, args.bins_number
    )

    try:
        mgr.run_all()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
