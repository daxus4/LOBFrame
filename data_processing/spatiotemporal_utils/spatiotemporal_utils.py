import glob
import json
import os
import re
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from fast_tmfg import TMFG

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
from utils import (
    dump_yaml_with_tuple,
    get_intervals_as_native_int,
    get_training_test_stocks_as_string,
    load_int_df,
    load_yaml_with_tuple,
    save_nested_parquet,
)


@dataclass
class SpatiotemporalMatrixPaths:
    lag_mi_df_checkpoint_path: Path
    lag_mi_df_final_path: Path
    logging_file_path: Path
    homological_structure_path: Path
    interval_lags_path: Path | None = None
    lag_candidates_path: Path | None = None
    pruned_lag_mi_df_path: Path | None = None
    spatiotemporal_matrix_path: Path | None = None
    images_folder_path: Path | None = None


def get_cliques(g: nx.Graph) -> Tuple[List[Set], List[Set], List[Set], List[Set]]:
    clique_1 = []
    clique_2 = []
    clique_3 = []
    clique_4 = []
    for clique in nx.enumerate_all_cliques(g):
        clique = set(clique)
        if len(clique) == 1:
            clique_1.append(clique)
        elif len(clique) == 2:
            clique_2.append(clique)
        elif len(clique) == 3:
            clique_3.append(clique)
        elif len(clique) == 4:
            clique_4.append(clique)
    return clique_1, clique_2, clique_3, clique_4


def get_cliques_connections(
    clique_last: List[Set], clique_next: List[Set]
) -> List[List]:
    connection_list = [[], []]
    component_mapping = {i: x for i, x in enumerate(clique_last)}
    for i, clique in enumerate(clique_next):
        component = [set(x) for x in combinations(clique, len(clique) - 1)]
        index_last = [
            list(component_mapping.keys())[list(component_mapping.values()).index(x)]
            for x in component
        ]
        for j in index_last:
            connection_list[0].append(j)
            connection_list[1].append(i)

    return connection_list


def parse_col(c):
    lag = re.search(r"_lag(\d+)", c)
    lag_num = int(lag.group(1)) if lag else 0
    c_clean = c.split("_lag")[0]
    match = re.match(r"(ASKs|ASKp|BIDs|BIDp)(\d+)", c_clean)
    if match:
        prefix, level = match.groups()
        return prefix, int(level), lag_num
    return c, 0, lag_num


def sort_key(c):
    if "Target" in c:
        return (99, 99, 99)  # Put target columns at the end

    order = ["ASKs", "ASKp", "BIDs", "BIDp"]
    prefix, level, lag_num = parse_col(c)
    return (
        lag_num,  # non-lagged first
        level,  # by level number
        order.index(prefix) if prefix in order else 99,  # by prefix order
    )


def get_spatiotemporal_mi_matrix(
    lob_files_paths: list[Path],
    initial_lags: list[int],
    num_bins: int,
    num_files_for_checkpoint: int,
    num_lags_to_select: int,
    saving_paths: SpatiotemporalMatrixPaths,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    # Compute lag mi df map for initial lags to check auto-MI
    lag_mi_df_map_manifest_file_path = (
        saving_paths.lag_mi_df_final_path / "manifest.json"
    )
    if not lag_mi_df_map_manifest_file_path.is_file():
        compute_lag_mi_df_map(
            lob_files_paths,
            initial_lags,
            saving_paths.logging_file_path,
            saving_paths.lag_mi_df_final_path,
            saving_paths.lag_mi_df_checkpoint_path,
            num_bins,
            num_files_for_checkpoint,
        )

    # Extract candidate lags
    if (
        saving_paths.interval_lags_path is None
        or not saving_paths.interval_lags_path.exists()
        or saving_paths.lag_candidates_path is None
        or not saving_paths.lag_candidates_path.exists()
    ):
        lag_mi_df_map = load_int_df(saving_paths.lag_mi_df_final_path)

        end_excluded_intervals, candidate_lags = get_candidates_lags(
            lag_mi_df_map,
            num_lags_to_select,
            initial_lags,
            saving_paths.images_folder_path,
        )

        end_excluded_intervals = get_intervals_as_native_int(end_excluded_intervals)
        candidate_lags = [int(lag) for lag in candidate_lags]

        if saving_paths.interval_lags_path is not None:
            dump_yaml_with_tuple(
                end_excluded_intervals, saving_paths.interval_lags_path
            )
        if saving_paths.lag_candidates_path is not None:
            dump_yaml_with_tuple(candidate_lags, saving_paths.lag_candidates_path)
    else:
        candidate_lags = load_yaml_with_tuple(saving_paths.lag_candidates_path)

    # Compute lag mi df map for candidate lags
    lags = initial_lags.copy()
    lags.extend(candidate_lags)
    lags = sorted(set(lags), reverse=True)

    compute_lag_mi_df_map(
        lob_files_paths,
        lags,
        saving_paths.logging_file_path,
        saving_paths.lag_mi_df_final_path,
        saving_paths.lag_mi_df_checkpoint_path,
        num_bins,
        num_files_for_checkpoint,
    )

    # Remove low correlated features
    lag_mi_df_map = load_int_df(saving_paths.lag_mi_df_final_path)

    index_lag_column_names_map = dict()
    for i, candidate_lag in enumerate(sorted(candidate_lags)):
        index_lag_column_names_map[i] = sorted(
            lag_mi_df_map[candidate_lag].columns.tolist(), key=sort_key
        )

    pruned_lag_mi_df_map = get_lag_mi_df_map_without_low_correlated_features(
        lag_mi_df_map, candidate_lags
    )
    if saving_paths.pruned_lag_mi_df_path is not None:
        save_nested_parquet(pruned_lag_mi_df_map, saving_paths.pruned_lag_mi_df_path)

    # get numpy array true/false for all columns if are pruned or not for each candidate lag
    index_lag_not_pruned_cols_map = dict()
    for i, candidate_lag in enumerate(sorted(candidate_lags)):
        pruned_cols = sorted(
            pruned_lag_mi_df_map[candidate_lag].columns.tolist(), key=sort_key
        )
        not_pruned_cols_mask = [
            col in pruned_cols for col in index_lag_column_names_map[i]
        ]
        index_lag_not_pruned_cols_map[i] = np.array(not_pruned_cols_mask)

    # Create spatiotemporal matrix
    spatiotemporal_df = get_spatiotemporal_df(pruned_lag_mi_df_map)

    return spatiotemporal_df, index_lag_not_pruned_cols_map


def get_spatiotemporal_tmfg(
    lob_files_paths: list[str],
    initial_lags: list[int],
    num_bins: int,
    num_files_for_checkpoint: int,
    num_lags_to_select: int,
    saving_paths: SpatiotemporalMatrixPaths,
) -> Tuple[GraphHomologicalStructure, List, List, np.ndarray, dict[int, np.ndarray]]:
    spatiotemporal_df, index_lag_not_pruned_cols_map = get_spatiotemporal_mi_matrix(
        lob_files_paths,
        initial_lags,
        num_bins,
        num_files_for_checkpoint,
        num_lags_to_select,
        saving_paths,
    )
    if saving_paths.spatiotemporal_matrix_path is not None:
        spatiotemporal_df.to_csv(
            saving_paths.spatiotemporal_matrix_path, sep="\t", index=True
        )

    spatiotemporal_df = spatiotemporal_df[
        sorted(spatiotemporal_df.columns, key=sort_key)
    ].copy()

    spatiotemporal_df = spatiotemporal_df.reindex(
        sorted(spatiotemporal_df.index, key=sort_key)
    )

    model_all = TMFG()
    cliques_all, seps_all, adj_matrix_all = model_all.fit_transform(
        spatiotemporal_df, output="weighted_sparse_W_matrix"
    )

    spatiotemporal_graph = nx.from_numpy_array(adj_matrix_all)
    clique_1, clique_2, clique_3, clique_4 = get_cliques(spatiotemporal_graph)

    connection_1 = get_cliques_connections(clique_1, clique_2)
    connection_2 = get_cliques_connections(clique_2, clique_3)
    connection_3 = get_cliques_connections(clique_3, clique_4)

    original_cliques_all = list(chain.from_iterable(cliques_all))
    original_seps_all = list(chain.from_iterable(seps_all))

    homological_structure = GraphHomologicalStructure(
        nodes_to_edges_connections=connection_1,
        edges_to_triangles_connections=connection_2,
        triangles_to_tetrahedra_connections=connection_3,
    )

    return (
        homological_structure,
        original_cliques_all,
        original_seps_all,
        adj_matrix_all,
        index_lag_not_pruned_cols_map,
    )


def execute_spatiotemporal_tmfg_pipeline(
    general_hyperparameters: Dict[str, Any],
    model_hyperparameters: Dict[str, Any],
):
    lob_folders_paths = sorted(
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

    lob_files_paths = list(chain.from_iterable(map(glob.glob, lob_folders_paths)))
    lob_files_paths = sorted([Path(lob_file) for lob_file in lob_files_paths])

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
    lag_mi_df_checkpoint_path = os.path.join(
        intermediate_files_path_dir, "lag_mi_df_checkpoint"
    )
    lag_mi_df_final_path = os.path.join(intermediate_files_path_dir, "lag_mi_df_final")
    images_folder_path_dir = os.path.join(
        intermediate_files_path_dir, IMAGES_SUBFOLDER_NAME
    )
    homological_structure_path = os.path.join(
        saving_path_dir, "st_hnn_homological_structure.yml"
    )

    os.makedirs(saving_path_dir, exist_ok=True)
    os.makedirs(intermediate_files_path_dir, exist_ok=True)
    os.makedirs(lag_mi_df_checkpoint_path, exist_ok=True)
    os.makedirs(lag_mi_df_final_path, exist_ok=True)
    os.makedirs(images_folder_path_dir, exist_ok=True)

    saving_paths = SpatiotemporalMatrixPaths(
        lag_mi_df_checkpoint_path=Path(lag_mi_df_checkpoint_path),
        lag_mi_df_final_path=Path(lag_mi_df_final_path),
        logging_file_path=Path(
            os.path.join(intermediate_files_path_dir, "logging.txt")
        ),
        homological_structure_path=homological_structure_path,
        interval_lags_path=Path(
            os.path.join(intermediate_files_path_dir, "interval_lags.yml")
        ),
        lag_candidates_path=Path(
            os.path.join(intermediate_files_path_dir, "lag_candidates.yml")
        ),
        pruned_lag_mi_df_path=Path(
            os.path.join(intermediate_files_path_dir, "pruned_lag_mi_df")
        ),
        spatiotemporal_matrix_path=Path(
            os.path.join(intermediate_files_path_dir, "spatiotemporal_matrix.tsv")
        ),
        images_folder_path=Path(images_folder_path_dir),
    )

    (
        homological_structure,
        original_cliques_all,
        original_seps_all,
        adj_matrix_all,
        index_lag_not_pruned_cols_map,
    ) = get_spatiotemporal_tmfg(
        lob_files_paths,
        model_hyperparameters["st_hnn_initial_lags"],
        NUM_BINS,
        NUM_FILES_FOR_CHECKPOINT,
        model_hyperparameters["st_hnn_number_past_lags"],
        saving_paths,
    )

    homological_structure_dict = homological_structure.to_dict()

    homological_structure_dataset = {
        "homological_structure": homological_structure_dict,
        "original_cliques_all": original_cliques_all,
        "original_seps_all": original_seps_all,
        "adj_matrix_all": adj_matrix_all,
        "window_index_cols_map": index_lag_not_pruned_cols_map,
    }

    dump_yaml_with_tuple(homological_structure_dataset, homological_structure_path)
    print("Spatiotemporal homological structures have been saved.")


if __name__ == "__main__":
    # Example usage
    general_hyperparameters = {
        "dataset": "nasdaq",
        "training_stocks": ["CSCO"],
        "target_stocks": ["CSCO"],
        "experiment_id": 1,
    }
    model_hyperparameters = {
        "st_hnn_initial_lags": [
            1000,
            500,
            400,
            300,
            200,
            100,
            90,
            80,
            70,
            60,
            50,
            40,
            30,
            25,
            20,
            15,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
        ],
        "st_hnn_number_past_lags": 20,
    }
    execute_spatiotemporal_tmfg_pipeline(general_hyperparameters, model_hyperparameters)
