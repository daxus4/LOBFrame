from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HNN.sparse_linear import SparseLinear


@dataclass
class GraphHomologicalStructure:
    nodes_to_edges_connections: tuple
    edges_to_triangles_connections: tuple
    triangles_to_tetrahedra_connections: tuple

    @property
    def num_nodes(self) -> int:
        return max(self.nodes_to_edges_connections[0]) + 1

    @property
    def num_edges(self) -> int:
        return max(self.edges_to_triangles_connections[0]) + 1

    @property
    def num_triangles(self) -> int:
        return (
            max(self.triangles_to_tetrahedra_connections[0]) + 1
            if self.triangles_to_tetrahedra_connections
            else 0
        )

    @property
    def num_tetrahedra(self) -> int:
        return (
            max(self.triangles_to_tetrahedra_connections[1]) + 1
            if self.triangles_to_tetrahedra_connections
            else 0
        )

    def get_nodes_to_edges_connections_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.nodes_to_edges_connections[1],
                self.nodes_to_edges_connections[0],
            ],
            dtype=torch.int64,
        )

    def get_edges_to_triangles_connections_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.edges_to_triangles_connections[1],
                self.edges_to_triangles_connections[0],
            ],
            dtype=torch.int64,
        )

    def get_triangles_to_tetrahedra_connections_tensor(self) -> torch.Tensor:
        return (
            torch.tensor(
                [
                    self.triangles_to_tetrahedra_connections[1],
                    self.triangles_to_tetrahedra_connections[0],
                ],
                dtype=torch.int64,
            )
            if self.triangles_to_tetrahedra_connections
            else torch.empty((2, 0), dtype=torch.int64)
        )

    def __deepcopy__(self, memo):
        return GraphHomologicalStructure(
            nodes_to_edges_connections=deepcopy(self.nodes_to_edges_connections, memo),
            edges_to_triangles_connections=deepcopy(
                self.edges_to_triangles_connections, memo
            ),
            triangles_to_tetrahedra_connections=deepcopy(
                self.triangles_to_tetrahedra_connections, memo
            ),
        )


class HNN(nn.Module):
    def __init__(
        self,
        homological_structure: GraphHomologicalStructure,
    ):
        super(HNN, self).__init__()
        self.homological_structure = homological_structure

        self.sparse_layer_edges = SparseLinear(
            homological_structure.num_nodes,
            homological_structure.num_edges,
            connectivity=self.homological_structure.get_nodes_to_edges_connections_tensor(),
        )

        self.sparse_layer_triangles = SparseLinear(
            self.homological_structure.num_edges,
            self.homological_structure.num_triangles,
            connectivity=self.homological_structure.get_edges_to_triangles_connections_tensor(),
        )

        if len(self.homological_structure.triangles_to_tetrahedra_connections[0]) != 0:
            self.sparse_layer_tetrahedra = SparseLinear(
                self.homological_structure.num_triangles,
                self.homological_structure.num_tetrahedra,
                connectivity=self.homological_structure.get_triangles_to_tetrahedra_connections_tensor(),
            )

        else:
            self.sparse_layer_tetrahedra = None

    def forward(self, x):
        x_s1 = F.relu(self.sparse_layer_edges(x))

        x_s2 = F.relu(self.sparse_layer_triangles(x_s1))

        if len(self.homological_structure.triangles_to_tetrahedra_connections[0]) != 0:
            x_s3 = F.relu(self.sparse_layer_tetrahedra(x_s2))

            return torch.cat([x_s1, x_s2, x_s3], 1)

        else:

            return torch.cat([x_s1, x_s2], 1)
