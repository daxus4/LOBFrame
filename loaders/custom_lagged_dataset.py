import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import tqdm
import matplotlib.pyplot as plt

from utils import detect_changing_points
from loaders.custom_dataset import CustomDataset

class CustomLaggedDataset(CustomDataset):
    def __init__(
        self,
        dataset,
        learning_stage,
        lags,
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
        target_stocks=None
    ):
        self.lags = lags
        self.max_lag = np.max(lags)
        super().__init__(
            dataset=dataset,
            learning_stage=learning_stage,
            shuffling_seed=shuffling_seed,
            cache_size=cache_size,
            lighten=lighten,
            threshold=threshold,
            all_horizons=all_horizons,
            prediction_horizon=prediction_horizon,
            targets_type=targets_type,
            balanced_dataloader=balanced_dataloader,
            backtest=backtest,
            training_stocks=training_stocks,
            validation_stocks=validation_stocks,
            target_stocks=target_stocks
        )

    def get_max_offset(self):
        return self.max_lag

    def get_window_data(self, cache_idx, start_idx):
        lag_idxs = [start_idx]
        lag_idxs.extend([start_idx + lag for lag in self.lags])
        if self.lighten:
            return self.cache_data[cache_idx][lag_idxs, :20]
        return self.cache_data[cache_idx][lag_idxs, :40]


'''
if __name__ == "__main__":
    # Create dataset and DataLoader with random shuffling
    dataset = CustomLaggedDataset(
        dataset="nasdaq",
        learning_stage="training",
        lags=[1,2,3,5,10,17,30,50,80],
        shuffling_seed=42,
        cache_size=1,
        lighten=True,
        threshold=32,
        targets_type="raw",
        all_horizons=[5, 10, 30, 50],
        prediction_horizon=10,
        balanced_dataloader=False,
        training_stocks=["IWM"],
        validation_stocks=["IWM"],
        target_stocks=["IWM"]
    )

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=True, sampler=dataset.glob_indices
    )

    print(len(dataloader))

    complete_list = []
    # Example usage of the DataLoader
    for batch_data, batch_labels in dataloader:
        # Train your model using batch_data and batch_labels
        # print(batch_labels.tolist())
        complete_list.extend(batch_labels.tolist())
        #print(batch_data.shape, batch_labels.shape)

    plt.hist(complete_list)
    plt.show()
'''