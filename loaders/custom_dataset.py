import glob
import random
import re
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from utils import detect_changing_points


class CustomDataset(Dataset, ABC):
    ORDER = ["ASKs", "ASKp", "BIDs", "BIDp"]

    def __init__(
        self,
        dataset,
        learning_stage,
        shuffling_seed,
        cache_size,
        lighten,
        threshold,
        all_horizons,
        prediction_horizon,
        targets_type,
        balanced_dataloader=False,
        backtest=False,
        training_stocks=None,
        validation_stocks=None,
        target_stocks=None,
    ):
        self.learning_stage = learning_stage
        self.shuffling_seed = shuffling_seed
        self.balanced_dataloader = (
            balanced_dataloader  # This option is available only for training.
        )
        self.backtest = backtest
        self.targets_type = targets_type
        self.lighten = lighten
        self.threshold = threshold
        self.prediction_horizon = prediction_horizon
        self.all_horizons = all_horizons
        self.cumulative_lengths = [
            0
        ]  # Stores the accumulated length of all processed datasets

        # Initialize file paths
        if self.learning_stage == "training":
            file_patterns = [
                f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv"
                for element in training_stocks
            ]
            self.csv_files = []
            for pattern in file_patterns:
                self.csv_files.extend(
                    glob.glob(pattern.format(dataset=dataset, self=self))
                )
            random.seed(self.shuffling_seed)
            random.shuffle(self.csv_files)
        else:
            # During the validation and testing stages it is fundamental to read the datasets in chronological order.
            if self.learning_stage == "validation":
                file_patterns = [
                    f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv"
                    for element in validation_stocks
                ]
            else:
                file_patterns = [
                    f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv"
                    for element in target_stocks
                ]
            self.csv_files = []
            for pattern in file_patterns:
                self.csv_files.extend(
                    glob.glob(pattern.format(dataset=dataset, self=self))
                )
            self.csv_files = sorted(self.csv_files)

        # Initialize cache
        self.cache_size = cache_size
        self.cache_data = [None] * self.cache_size
        self.cache_indices = [None] * self.cache_size
        self.current_cache_index = -1
        self.glob_indices = []

        # Build dataset
        self.build_dataset()

    def build_dataset(self):
        if self.balanced_dataloader:
            print(f"BALANCED dataset construction...")
        else:
            print(f"UNBALANCED dataset construction...")

        for csv_file in tqdm.tqdm(self.csv_files):
            df = pd.read_csv(csv_file)
            self.process_file(df)

    def process_file(self, df):
        max_offset = self.get_max_offset()
        self.cumulative_lengths.append(
            self.cumulative_lengths[-1] + len(df) - max_offset
        )

        if self.learning_stage == "training":
            temp_labels = []
            target_col = (
                f"Raw_Target_{self.prediction_horizon}"
                if self.targets_type == "raw"
                else f"Smooth_Target_{self.prediction_horizon}"
            )
            labels = df.iloc[:-max_offset, :][target_col]

            for label, index in zip(
                labels, range(self.cumulative_lengths[-2], self.cumulative_lengths[-1])
            ):
                if label > self.threshold:
                    temp_labels.append((2, index))
                elif label < -self.threshold:
                    temp_labels.append((0, index))
                else:
                    temp_labels.append((1, index))

            class_groups = {}
            for class_rep, index in temp_labels:
                corresponding_cumulative_length = detect_changing_points(
                    index, self.cumulative_lengths
                )
                temp_index = (
                    index - corresponding_cumulative_length
                    if corresponding_cumulative_length is not None
                    else index
                )

                # Even having a balanced dataloader, labels would be messed up once computing models' inputs.
                # Indeed, given an index 'i', the input rows go from 'i' to 'i + max_offset' and the label to be used is the one at 'i + max_offset'.
                # Therefore, we must subtract the max_offset from the index of each sample.
                if temp_index >= max_offset:
                    if class_rep in class_groups:
                        class_groups[class_rep].append(index - max_offset)
                    else:
                        class_groups[class_rep] = [index - max_offset]

            if self.balanced_dataloader:
                # Determine the desired number of samples per class (pseudo-balanced). We use the size of the less represented class.
                min_samples_class = min(
                    len(indices) for indices in class_groups.values()
                )
                if min_samples_class > 5000:
                    min_samples_class = 5000
                balanced_sample_size = min_samples_class

            # We randomly select indices from each class to create the subsample.
            subsample_indices = []
            for class_rep, indices in class_groups.items():
                random.seed(self.shuffling_seed)
                sample_size = (
                    balanced_sample_size
                    if self.balanced_dataloader
                    else int(len(indices) * 0.1)
                )
                subsample_indices.extend(random.sample(indices, int(sample_size)))

            random.seed(self.shuffling_seed)
            random.shuffle(subsample_indices)
            self.glob_indices.extend(subsample_indices)

    @abstractmethod
    def get_max_offset(self):
        """Returns the maximum offset needed for the specific dataset type (window size or max lag)"""
        pass

    @abstractmethod
    def get_window_data(self, cache_idx, start_idx):
        """Returns the window data for the specific dataset type"""
        pass

    def __len__(self):
        return self.cumulative_lengths[-1]

    @classmethod
    def sort_key(cls, c):
        match = re.match(r"(ASKs|ASKp|BIDs|BIDp)(\d+)", c)
        if match:
            prefix, level = match.groups()
            return (int(level), cls.ORDER.index(prefix))
        return (9999, 9999)  # non-matching go to the end (but we won't use this)

    @classmethod
    def get_sorted_columns_df(cls, df):
        cols = df.columns.tolist()
        lob_cols = [c for c in cols if re.match(r"(ASKs|ASKp|BIDs|BIDp)\d+$", c)]

        lob_cols_sorted = sorted(lob_cols, key=cls.sort_key)

        # Build final column list
        final_cols = [c if c not in lob_cols else None for c in cols]  # placeholders
        lob_iter = iter(lob_cols_sorted)
        final_cols = [next(lob_iter) if c is None else c for c in final_cols]

        # Apply to dataframe
        df = df[final_cols].copy()

        return df

    def cache_dataset(self, dataset_index):
        if self.current_cache_index >= 0:
            self.cache_data[self.current_cache_index] = None
            self.cache_indices[self.current_cache_index] = None

        self.current_cache_index = random.randint(0, self.cache_size - 1)
        df = pl.read_csv(self.csv_files[dataset_index]).to_pandas()
        df = self.get_sorted_columns_df(df)
        self.cache_data[self.current_cache_index] = df.values[:, 1:].astype(np.float32)
        self.cache_indices[self.current_cache_index] = dataset_index

    def __getitem__(self, index):
        try:
            dataset_index = 0
            while index >= self.cumulative_lengths[dataset_index + 1]:
                dataset_index += 1

            if self.cache_indices[self.current_cache_index] != dataset_index:
                self.cache_dataset(dataset_index)

            start_index = (
                index
                if dataset_index == 0
                else index - self.cumulative_lengths[dataset_index]
            )
            window_data = self.get_window_data(self.current_cache_index, start_index)

            position = next(
                (
                    i
                    for i, v in enumerate(self.all_horizons)
                    if v == self.prediction_horizon
                ),
                None,
            )
            label = self.cache_data[self.current_cache_index][
                start_index + self.get_max_offset(), 40:
            ][position]

            if self.backtest is False:
                if label > self.threshold:
                    label = 2
                elif label < -self.threshold:
                    label = 0
                else:
                    label = 1

            return torch.tensor(window_data).unsqueeze(0), torch.tensor(label)
        except Exception as e:
            print(f"Exception in DataLoader worker: {e}")
            raise e
