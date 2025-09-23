import pickle
import shutil

import torch
from torch.utils.data import DataLoader

from data_processing.spatiotemporal_utils.constants import (
    INTERMEDIATE_FILES_SUBFOLDER_NAME,
    SAVING_FOLDER_NAME,
)
from loaders.custom_dataset import CustomDataset
from loaders.custom_windowed_dataset import CustomWindowedDataset
from loggers import logger
from models.AxialLob.axiallob import AxialLOB
from models.CNN1.cnn1 import CNN1
from models.CNN2.cnn2 import CNN2
from models.CompleteHCNN.complete_hcnn import Complete_HCNN
from models.DeepLob.deeplob import DeepLOB
from models.DLA.DLA import DLA
from models.HNN.spatio_temporal_hnn import SpatioTemporalHNN
from models.iTransformer.itransformer import ITransformer
from models.LobTransformer.lobtransformer import LobTransformer
from models.TABL.bin_tabl import BiN_BTABL, BiN_CTABL
from models.Transformer.transformer import Transformer
from optimizers.lightning_batch_gd import BatchGDManager
from utils import create_tree, get_training_test_stocks_as_string


class Executor:
    def __init__(
        self,
        experiment_id,
        general_hyperparameters,
        model_hyperparameters,
        torch_dataset_preparation=False,
        torch_dataset_preparation_backtest=False,
    ):
        self.manager = None
        self.model = None
        self.experiment_id = experiment_id
        self.torch_dataset_preparation = torch_dataset_preparation
        self.torch_dataset_preparation_backtest = torch_dataset_preparation_backtest

        self.training_stocks_string, self.test_stocks_string = (
            get_training_test_stocks_as_string(general_hyperparameters)
        )

        if self.torch_dataset_preparation:
            create_tree(
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/"
            )

        spatiotemporal_executor = general_hyperparameters["model"] == "sthnn"
        training_dataset_name = (
            "training_dataset_sthnn.pt"
            if spatiotemporal_executor
            else "training_dataset.pt"
        )
        validation_dataset_name = (
            "validation_dataset_sthnn.pt"
            if spatiotemporal_executor
            else "validation_dataset.pt"
        )
        test_dataset_backtest_name = (
            "test_dataset_backtest_sthnn.pt"
            if spatiotemporal_executor
            else "test_dataset_backtest.pt"
        )
        test_dataset_name = (
            "test_dataset_sthnn.pt" if spatiotemporal_executor else "test_dataset.pt"
        )

        if general_hyperparameters["model"] == "deeplob":
            self.model = DeepLOB(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "transformer":
            self.model = Transformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "itransformer":
            self.model = ITransformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "lobtransformer":
            self.model = LobTransformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "dla":
            self.model = DLA(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "cnn1":
            self.model = CNN1()
        elif general_hyperparameters["model"] == "cnn2":
            self.model = CNN2()
        elif general_hyperparameters["model"] == "binbtabl":
            self.model = BiN_BTABL(120, 40, 100, 5, 3, 1)
        elif general_hyperparameters["model"] == "binctabl":
            self.model = BiN_CTABL(120, 40, 100, 5, 120, 5, 3, 1)
        elif general_hyperparameters["model"] == "axiallob":
            self.model = AxialLOB()
        elif general_hyperparameters["model"] == "hlob":
            homological_structures = torch.load(
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/complete_homological_structures.pt"
            )
            self.model = Complete_HCNN(
                lighten=model_hyperparameters["lighten"],
                homological_structures=homological_structures,
            )

        elif general_hyperparameters["model"] == "sthnn":
            homological_structures = torch.load(
                f"./{SAVING_FOLDER_NAME}/{self.training_stocks_string}/experiment_id_{experiment_id}/st_hnn_homological_structure.pt"
            )
            end_excluded_interval_windows = pickle.load(
                open(
                    f"./{SAVING_FOLDER_NAME}/{self.training_stocks_string}/experiment_id_{experiment_id}/{INTERMEDIATE_FILES_SUBFOLDER_NAME}/interval_lags.pkl",
                    "rb",
                )
            )
            self.model = SpatioTemporalHNN(
                homological_structure=homological_structures,
                num_convolutional_channels=model_hyperparameters[
                    "num_convolutional_channels"
                ],
                lighten=model_hyperparameters["lighten"],
                num_classes=len(general_hyperparameters["targets_type"]),
            )

        if self.torch_dataset_preparation:
            # Prepare the training dataloader.
            if spatiotemporal_executor:
                dataset = CustomWindowedDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="training",
                    windows_limits=end_excluded_interval_windows,
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    targets_type=general_hyperparameters["targets_type"],
                    balanced_dataloader=model_hyperparameters["balanced_sampling"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )

            else:
                dataset = CustomDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="training",
                    window_size=model_hyperparameters["history_length"],
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    targets_type=general_hyperparameters["targets_type"],
                    balanced_dataloader=model_hyperparameters["balanced_sampling"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            torch.save(
                dataset,
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{training_dataset_name}",
            )
        elif (
            self.torch_dataset_preparation is False
            and self.torch_dataset_preparation_backtest is False
        ):
            dataset = torch.load(
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{training_dataset_name}"
            )
            self.train_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
                sampler=dataset.glob_indices,
            )

        if self.torch_dataset_preparation:
            # Prepare the validation dataloader.
            if spatiotemporal_executor:
                dataset = CustomWindowedDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="validation",
                    windows_limits=end_excluded_interval_windows,
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            else:
                dataset = CustomDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="validation",
                    window_size=model_hyperparameters["history_length"],
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )

            torch.save(
                dataset,
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{validation_dataset_name}",
            )
        elif (
            self.torch_dataset_preparation is False
            and self.torch_dataset_preparation_backtest is False
        ):
            dataset = torch.load(
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{validation_dataset_name}"
            )
            self.val_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
            )

        if (
            self.torch_dataset_preparation is False
            and self.torch_dataset_preparation_backtest
        ):
            if spatiotemporal_executor:
                dataset = CustomWindowedDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="test",
                    windows_limits=end_excluded_interval_windows,
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    backtest=True,
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            else:
                dataset = CustomDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="test",
                    window_size=model_hyperparameters["history_length"],
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    backtest=True,
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            torch.save(
                dataset,
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{test_dataset_backtest_name}",
            )
        elif (
            self.torch_dataset_preparation
            and self.torch_dataset_preparation_backtest is False
        ):
            if spatiotemporal_executor:
                dataset = CustomWindowedDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="test",
                    windows_limits=end_excluded_interval_windows,
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            else:
                dataset = CustomDataset(
                    dataset=general_hyperparameters["dataset"],
                    learning_stage="test",
                    window_size=model_hyperparameters["history_length"],
                    shuffling_seed=model_hyperparameters["shuffling_seed"],
                    cache_size=1,
                    lighten=model_hyperparameters["lighten"],
                    threshold=model_hyperparameters["threshold"],
                    all_horizons=general_hyperparameters["horizons"],
                    targets_type=general_hyperparameters["targets_type"],
                    prediction_horizon=model_hyperparameters["prediction_horizon"],
                    training_stocks=general_hyperparameters["training_stocks"],
                    validation_stocks=general_hyperparameters["target_stocks"],
                    target_stocks=general_hyperparameters["target_stocks"],
                )
            torch.save(
                dataset,
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{test_dataset_name}",
            )
        elif (
            self.torch_dataset_preparation is False
            and self.torch_dataset_preparation_backtest is False
        ):
            dataset = torch.load(
                f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/{test_dataset_name}"
            )
            self.test_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
            )

        if (
            self.torch_dataset_preparation is False
            and self.torch_dataset_preparation_backtest is False
        ):
            self.manager = BatchGDManager(
                experiment_id=experiment_id,
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                epochs=model_hyperparameters["epochs"],
                learning_rate=model_hyperparameters["learning_rate"],
                patience=model_hyperparameters["patience"],
                general_hyperparameters=general_hyperparameters,
                model_hyperparameters=model_hyperparameters,
            )

    def execute_training(self):
        self.manager.train()

    def execute_testing(self):
        self.manager.test()

    def logger_clean_up(self):
        folder_path = f"{logger.find_save_path(self.experiment_id)}/wandb/"
        try:
            shutil.rmtree(folder_path)
        except:
            pass
