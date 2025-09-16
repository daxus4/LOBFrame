import pandas as pd

from loaders.custom_dataset import CustomDataset


class CustomWindowedDataset(CustomDataset):
    def __init__(
        self,
        dataset,
        learning_stage,
        windows_limits: list[tuple[int, int]],
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
        """
        windows_limits is a list of tuples specifying the start and end indices for each window.
        It must be ordered. End index must be excluded. If I have [(0,2), (2,5), (5,10)], it means I have three windows:
        - Window 1: 0,1
        - Window 2: 2,3,4
        - Window 3: 5,6,7,8,9
        """
        self.windows_limits = windows_limits
        self.last_lag = windows_limits[-1][1]

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
            target_stocks=target_stocks,
        )

    def get_max_offset(self):
        return self.last_lag

    def get_window_data(self, cache_idx, start_idx):
        columns_number = 20 if self.lighten else 40
        window_means = [
            self._get_single_window_mean(
                start_idx, cache_idx, start_lag_window, end_lag_window, columns_number
            )
            for (start_lag_window, end_lag_window) in self.windows_limits
        ]

        # Stack into new DataFrame, reverse order (decreasing by window number)
        result = pd.DataFrame(
            window_means[::-1],
            columns=self.cache_data[cache_idx].columns[:columns_number],
        )
        result.index = range(len(result))  # reset index
        return result

    def _get_single_window_mean(
        self,
        start_idx: int,
        cache_idx: int,
        start_lag_window: int,
        end_lag_window: int,
        columns_number: int,
    ) -> pd.Series:
        # + 1 in necessary; otherwise the window will be shifted by one in the past
        start_window_idx = start_idx - end_lag_window + 1
        end_window_idx = start_idx - start_lag_window + 1

        window_df = self.cache_data[cache_idx].iloc[
            start_window_idx:end_window_idx, :columns_number
        ]

        return window_df.mean()


"""if __name__ == "__main__":
    # Create dataset and DataLoader with random shuffling
    dataset = CustomWindowedDataset(
        dataset="nasdaq",
        learning_stage="training",
        windows_limits=[(0, 1), (1, 3), (3, 7), (7, 15), (15, 31)],
        shuffling_seed=42,
        cache_size=1,
        lighten=True,
        threshold=32,
        targets_type="raw",
        all_horizons=[5, 10, 30, 50, 100],
        prediction_horizon=100,
        balanced_dataloader=False,
        training_stocks=["CSCO"],
        validation_stocks=["CSCO"],
        target_stocks=["CSCO"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        sampler=dataset.glob_indices,
    )

    print(len(dataloader))

    complete_list = []
    # Example usage of the DataLoader
    for batch_data, batch_labels in dataloader:
        # Train your model using batch_data and batch_labels
        # print(batch_labels.tolist())
        complete_list.extend(batch_labels.tolist())
        # print(batch_data.shape, batch_labels.shape)

    plt.hist(complete_list)
    plt.show()
"""
