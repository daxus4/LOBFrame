import numpy as np

from loaders.custom_dataset import CustomDataset


class CustomWindowedDataset(CustomDataset):
    def __init__(
        self,
        dataset,
        learning_stage,
        windows_limits: list[tuple[int, int]],
        window_index_cols_map: dict[int, np.ndarray],
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
        self.last_lag = windows_limits[-1][
            1
        ]  # probabilmente qui va aggiunto un piu 2. pero devo controllare poi cosa succede in process_df di custom_dataset.
        # Attenzione che forse ci sono stronzate nel codice di daniel, quindi provo a guardare anche quello di anto
        self.window_index_cols_map = window_index_cols_map

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
        window_means = [
            self._get_single_window_mean(
                start_idx,
                cache_idx,
                start_lag_window,
                end_lag_window,
                window_index,
            )
            for window_index, (start_lag_window, end_lag_window) in enumerate(
                self.windows_limits
            )
        ]

        # Stack into new array ready for st_hnn shape = (1, sum (for i in windows) num_features(i))
        result = np.concatenate(window_means)
        return result

    def _get_single_window_mean(
        self,
        start_idx: int,
        cache_idx: int,
        start_lag_window: int,
        end_lag_window: int,
        window_index: int,
    ) -> np.ndarray:
        start_window_idx = start_idx + (self.last_lag - end_lag_window)
        end_window_idx = start_idx + (self.last_lag - start_lag_window)

        mask = self.window_index_cols_map[window_index]

        columns_number = 20 if self.lighten else 40

        window_df = self.cache_data[cache_idx][
            start_window_idx:end_window_idx, :columns_number
        ]
        window_df = window_df[:, mask]

        if window_df.size <= 0:
            raise ValueError(
                f"Window df has shape 0. start_window_idx: {start_window_idx}, end_window_idx: {end_window_idx}, mask: {mask}, window_index: {window_index}, start_idx: {start_idx}, cache_idx: {cache_idx}"
            )
        return window_df.mean(axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Create dataset and DataLoader with random shuffling
    dataset = CustomWindowedDataset(
        dataset="nasdaq",
        learning_stage="training",
        windows_limits=[(0, 1), (1, 3), (3, 7), (7, 15), (15, 31)],
        window_index_cols_map={
            0: np.arange(20),  # First window uses all columns
            1: np.arange(18),  # Second window uses all columns
            2: np.arange(16),  # Third window uses all columns
            3: np.arange(14),  # Fourth window uses all columns
            4: np.arange(12),  # Fifth window uses all columns
        },
        shuffling_seed=42,
        cache_size=1,
        lighten=True,
        threshold=0.01,
        targets_type="raw",
        all_horizons=[10, 50, 100],
        prediction_horizon=100,
        balanced_dataloader=True,
        training_stocks=["CSCO"],
        validation_stocks=["CSCO"],
        target_stocks=["CSCO"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
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
    complete_list = np.array(complete_list)
    print("Labels histogram:")
    print(np.bincount(complete_list))
    plt.savefig("histogram.png")
