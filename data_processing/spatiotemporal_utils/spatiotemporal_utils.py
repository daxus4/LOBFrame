import os
import pickle
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from fast_tmfg import TMFG

from data_processing.complete_homological_utils import extract_components
from data_processing.spatiotemporal_utils.candidate_lags_extraction import (
    get_candidates_lags,
)
from data_processing.spatiotemporal_utils.constants import (
    IMAGES_SUBFOLDER_NAME,
    INTERMEDIATE_FILES_SUBFOLDER_NAME,
    NUM_BINS,
    NUM_FILES_FOR_CHECKPOINT,
    SAVING_FOLDER_NAME,
)
from data_processing.spatiotemporal_utils.lag_mi_df_map_computation import (
    compute_lag_mi_df_map,
)
from data_processing.spatiotemporal_utils.low_information_features_removal import (
    get_lag_mi_df_map_without_low_correlated_features,
)
from data_processing.spatiotemporal_utils.spatiotemporal_matrix_creation import (
    get_spatiotemporal_df,
)
from models.HNN.hnn import GraphHomologicalStructure
from utils import get_training_test_stocks_as_string


@dataclass
class SpatiotemporalMatrixPaths:
    lag_mi_df_path: Path
    logging_file_path: Path
    homological_structure_path: Path
    interval_lags_path: Path | None = None
    lag_candidates_path: Path | None = None
    pruned_lag_mi_df_path: Path | None = None
    spatiotemporal_matrix_path: Path | None = None
    images_folder_path: Path | None = None


def get_spatiotemporal_mi_matrix(
    lob_files_paths: list[str],
    initial_lags: list[int],
    num_bins: int,
    num_files_for_checkpoint: int,
    num_lags_to_select: int,
    saving_paths: SpatiotemporalMatrixPaths,
) -> pd.DataFrame:
    # Compute lag mi df map for initial lags to check auto-MI
    compute_lag_mi_df_map(
        lob_files_paths,
        initial_lags,
        saving_paths.logging_file_path,
        saving_paths.lag_mi_df_path,
        num_bins,
        num_files_for_checkpoint,
    )

    # Extract candidate lags
    with open(saving_paths.lag_mi_df_path, "rb") as f:
        lag_mi_df_map = pickle.load(f)
    lag_mi_df_map = lag_mi_df_map["final"]

    end_excluded_intervals, candidate_lags = get_candidates_lags(
        lag_mi_df_map, num_lags_to_select, initial_lags, saving_paths.images_folder_path
    )

    if saving_paths.interval_lags_path is not None:
        with open(saving_paths.interval_lags_path, "wb") as f:
            pickle.dump(end_excluded_intervals, f)
    if saving_paths.lag_candidates_path is not None:
        with open(saving_paths.lag_candidates_path, "wb") as f:
            pickle.dump(candidate_lags, f)

    # Compute lag mi df map for candidate lags
    lags = initial_lags.copy()
    lags.extend(candidate_lags)
    lags = sorted(set(lags), reverse=True)

    compute_lag_mi_df_map(
        lob_files_paths,
        lags,
        saving_paths.logging_file_path,
        saving_paths.lag_mi_df_path,
        num_bins,
        num_files_for_checkpoint,
    )

    # Remove low correlated features
    with open(saving_paths.lag_mi_df_path, "rb") as f:
        lag_mi_df_map = pickle.load(f)
    lag_mi_df_map = lag_mi_df_map["final"]

    pruned_lag_mi_df_map = get_lag_mi_df_map_without_low_correlated_features(
        lag_mi_df_map, candidate_lags
    )
    if saving_paths.pruned_lag_mi_df_path is not None:
        with open(saving_paths.pruned_lag_mi_df_path, "wb") as f:
            pickle.dump(pruned_lag_mi_df_map, f)

    # Create spatiotemporal matrix
    spatiotemporal_df = get_spatiotemporal_df(pruned_lag_mi_df_map)

    return spatiotemporal_df


def get_spatiotemporal_tmfg(
    lob_files_paths: list[str],
    initial_lags: list[int],
    num_bins: int,
    num_files_for_checkpoint: int,
    num_lags_to_select: int,
    saving_paths: SpatiotemporalMatrixPaths,
) -> Tuple[GraphHomologicalStructure, List, List, np.ndarray]:
    spatiotemporal_df = get_spatiotemporal_mi_matrix(
        lob_files_paths,
        initial_lags,
        num_bins,
        num_files_for_checkpoint,
        num_lags_to_select,
        saving_paths,
    )
    if saving_paths.spatiotemporal_matrix_path is not None:
        with open(saving_paths.spatiotemporal_matrix_path, "wb") as f:
            pickle.dump(spatiotemporal_df, f)

    model_all = TMFG()
    cliques_all, seps_all, adj_matrix_all = model_all.fit_transform(
        spatiotemporal_df, output="weighted_sparse_W_matrix"
    )

    c4, c3, c2 = extract_components(cliques_all, seps_all, adj_matrix_all)
    c4 = list(chain.from_iterable(c4))
    c3 = list(chain.from_iterable(c3))
    c2 = list(chain.from_iterable(c2))

    original_cliques_all = list(chain.from_iterable(cliques_all))
    original_seps_all = list(chain.from_iterable(seps_all))

    homological_structure = GraphHomologicalStructure(
        nodes_to_edges_connections=c2,
        edges_to_triangles_connections=c3,
        triangles_to_tetrahedra_connections=c4,
    )

    return (
        homological_structure,
        original_cliques_all,
        original_seps_all,
        adj_matrix_all,
    )


def execute_spatiotemporal_tmfg_pipeline(
    general_hyperparameters: Dict[str, Any],
    model_hyperparameters: Dict[str, Any],
):
    lob_files_paths = sorted(
        [
            os.path.join(
                ".",
                "data",
                general_hyperparameters["dataset"],
                "unscaled_data",
                "training",
                f"*{element}*.csv",
            )
            for element in general_hyperparameters["training_stocks"]
        ]
    )

    training_stocks_string, _ = get_training_test_stocks_as_string(
        general_hyperparameters
    )

    experiment_id = general_hyperparameters["experiment_id"]

    saving_path_dir = os.path.join(
        os.getcwd(),
        SAVING_FOLDER_NAME,
        f"{training_stocks_string}",
        f"experiment_id_{experiment_id}",
    )
    intermediate_files_path_dir = os.path.join(
        saving_path_dir, INTERMEDIATE_FILES_SUBFOLDER_NAME
    )
    images_folder_path_dir = os.path.join(
        intermediate_files_path_dir, IMAGES_SUBFOLDER_NAME
    )
    homological_structure_path = os.path.join(
        saving_path_dir, "st_hnn_homological_structure.pkl"
    )

    os.makedirs(saving_path_dir, exist_ok=True)
    os.makedirs(intermediate_files_path_dir, exist_ok=True)
    os.makedirs(images_folder_path_dir, exist_ok=True)

    saving_paths = SpatiotemporalMatrixPaths(
        lag_mi_df_path=os.path.join(intermediate_files_path_dir, "lag_mi_df.pkl"),
        logging_file_path=os.path.join(intermediate_files_path_dir, "logging.txt"),
        homological_structure_path=homological_structure_path,
        interval_lags_path=os.path.join(
            intermediate_files_path_dir, "interval_lags.pkl"
        ),
        lag_candidates_path=os.path.join(
            intermediate_files_path_dir, "lag_candidates.pkl"
        ),
        pruned_lag_mi_df_path=os.path.join(
            intermediate_files_path_dir, "pruned_lag_mi_df.pkl"
        ),
        spatiotemporal_matrix_path=os.path.join(
            intermediate_files_path_dir, "spatiotemporal_matrix.pkl"
        ),
        images_folder_path=images_folder_path_dir,
    )

    # POI ATTENZIONE CHE CODICE PER SELEZIONARE I VALIDATION DAYS GIUSTI NON FUNZIONA.
    # DEVO PROVARE A RILANCIARE PRIMA IL CODICE PER CREARE I DATASET SCALED E UNSCALED (FACENDOLO STOPPARE PRIMA CHE CHIAMA LA FUNZIONE DI DIVISIONE DEI DF)
    # E POI DEBUGGARE IL CODICE DI SELEZIONE DEI FILES DI VALIDATION

    homological_structure, original_cliques_all, original_seps_all, adj_matrix_all = (
        get_spatiotemporal_tmfg(
            lob_files_paths,
            model_hyperparameters["st_hnn_initial_lags"],
            NUM_BINS,
            NUM_FILES_FOR_CHECKPOINT,
            model_hyperparameters["st_hnn_number_past_lags"],
            saving_paths,
        )
    )

    homological_structure_dataset = {
        "homological_structure": homological_structure,
        "original_cliques_all": original_cliques_all,
        "original_seps_all": original_seps_all,
        "adj_matrix_all": adj_matrix_all,
    }

    torch.save(homological_structure_dataset, homological_structure_path)
    print("Spatiotemporal homological structures have been saved.")
