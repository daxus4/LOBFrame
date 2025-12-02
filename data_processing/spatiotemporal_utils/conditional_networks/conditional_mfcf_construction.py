import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from data_processing.spatiotemporal_utils.conditional_networks.fast_fast_mfcf import (
    MFCF,
    Clique,
)
from data_processing.spatiotemporal_utils.conditional_networks.lag_conditioned_mi_df_map_computation import (
    IOHandler,
)


@dataclass
class SpatiotemporalMatrixBuilder:
    def build_spatiotemporal_matrix(
        self, mi_dict_for_class: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Build the spatiotemporal symmetric MI matrix.
        mi_dict_for_class: {lag → small DataFrame}
        """

        lags = sorted(mi_dict_for_class.keys(), reverse=True)
        n_lags = len(lags)
        n_features = mi_dict_for_class[lags[0]].shape[0]

        # Build index labels directly from the DF names
        index_labels = []
        for lag in lags:
            index_labels.extend(mi_dict_for_class[lag].index.tolist())

        size = n_lags * n_features
        big = np.zeros((size, size))

        # Single loop, fill block rows
        for block_i, lag in enumerate(lags):
            mat = mi_dict_for_class[lag].values
            r0 = block_i * n_features
            r1 = r0 + n_features

            for block_j in range(block_i + 1):
                c0 = block_j * n_features
                c1 = c0 + n_features

                if block_i == block_j:
                    big[r0:r1, c0:c1] = mat
                else:
                    big[r0:r1, c0:c1] = mat
                    big[c0:c1, r0:r1] = mat.T

        return pd.DataFrame(big, index=index_labels, columns=index_labels)


class MCFCSparsifier:
    @staticmethod
    def node_graph_from_cliques(cliques):
        G = nx.Graph()
        for clq in cliques:
            nodes = list(clq)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    G.add_edge(nodes[i], nodes[j])
        return G

    def __init__(self, mfcf_model: MFCF):
        self.mfcf = mfcf_model

    def get_sparsified_adjacency_matrix(
        self, similarity_matrix: pd.DataFrame
    ) -> tuple[pd.DataFrame, List[Clique]]:
        """
        Apply MFCF sparsification to the similarity matrix.
        """
        cliques, separators_count, peo, J_logo = self.mfcf.run(similarity_matrix.values)

        graph = self.node_graph_from_cliques(cliques)
        sparse_adj = nx.to_numpy_array(
            graph, nodelist=range(len(similarity_matrix.index))
        )
        sparse_adj = pd.DataFrame(
            sparse_adj, index=similarity_matrix.index, columns=similarity_matrix.columns
        )
        return sparse_adj, cliques


@dataclass
class Pipeline:
    builder: SpatiotemporalMatrixBuilder
    sparsifier: MCFCSparsifier

    def run(
        self, class_lag_mi_matrices_map: Dict[int, Dict[int, pd.DataFrame]]
    ) -> tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        adjancency_matrices_map, cliques_map = self.get_class_adjacency_matrix_map(
            class_lag_mi_matrices_map
        )

        final_sum = sum(list(adjancency_matrices_map.values()))
        final_sum = (final_sum > 0).astype(int)

        return final_sum, adjancency_matrices_map

    def get_class_adjacency_matrix_map(
        self, class_lag_mi_matrices_map: Dict[int, Dict[int, pd.DataFrame]]
    ) -> tuple[Dict[int, pd.DataFrame], Dict[int, List[Clique]]]:
        cliques_map = {}
        adjancency_matrices_map = {}

        # Build and sparsify per class
        for cls, lag_dict in class_lag_mi_matrices_map.items():
            spatiotemporal_sim_matrix = self.builder.build_spatiotemporal_matrix(
                lag_dict
            )
            adjancency_matrix, cliques = (
                self.sparsifier.get_sparsified_adjacency_matrix(
                    spatiotemporal_sim_matrix
                )
            )

            adjancency_matrices_map[cls] = adjancency_matrix
            cliques_map[cls] = cliques

        return adjancency_matrices_map, cliques_map


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute MFCF sparsified adjacency matrix from class-lag MI matrices."
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Folder containing .npz files with MI matrices.",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output folder for the results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold value for MFCF sparsification.",
    )

    parser.add_argument(
        "--min_clique_size",
        type=int,
        default=4,
        help="Minimum clique size to consider (default: 4).",
    )

    parser.add_argument(
        "--max_clique_size",
        type=int,
        default=4,
        help="Maximum clique size to consider (default: 4).",
    )

    parser.add_argument(
        "--coordination_number",
        type=int,
        default=np.inf,
        help="Coordination number for node degree thresholding (default: infinite).",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    class_lag_mi_matrices_map = IOHandler.get_class_lag_mi_matrices_map(
        args.input_folder
    )

    builder = SpatiotemporalMatrixBuilder()
    mfcf_model = MFCF(
        threshold=args.threshold,
        min_clique_size=args.min_clique_size,
        max_clique_size=args.max_clique_size,
        coordination_number=args.coordination_number,
    )
    sparsifier = MCFCSparsifier(mfcf_model)

    pipeline = Pipeline(builder, sparsifier)
    final_adjacency_matrix, conditional_adjancency_matrices_map = pipeline.run(
        class_lag_mi_matrices_map
    )

    if not args.output_path.parent.exists():
        args.output_path.parent.mkdir(parents=True, exist_ok=True)

    final_adjacency_matrix.to_csv(
        args.output_path / "final_adjacency_matrix.tsv",
        index=True,
        header=True,
        sep="\t",
    )

    for cls, adj_matrix in conditional_adjancency_matrices_map.items():
        adj_matrix.to_csv(
            args.output_path / f"conditional_adjacency_matrix_class_{cls}.tsv",
            index=True,
            header=True,
            sep="\t",
        )
