import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HNN.sparse_linear import SparseLinear


class HNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        num_triangles: int,
        num_tetrahedra: int,
        nodes_to_edges_connections: tuple,
        edges_to_triangles_connections: tuple,
        triangles_to_tetrahedra_connections: tuple,
    ):
        """
        nodes_to_edges_connections: tuple of two lists, where the first list contains the indices of the edges
        and the second list contains the indices of the nodes connected to those edges, such that the i-th node
        in the first list is a member of the i-th edge in the second list.

        Same for edges_to_triangles_connections and triangles_to_tetrahedra_connections
        """
        super(HNN, self).__init__()
        self.sparse_layer_edges = SparseLinear(
            num_nodes,
            num_edges,
            connectivity=torch.tensor(
                [nodes_to_edges_connections[1], nodes_to_edges_connections[0]],
                dtype=torch.int64,
            ),
        )

        self.sparse_layer_triangles = SparseLinear(
            num_edges,
            num_triangles,
            connectivity=torch.tensor(
                [edges_to_triangles_connections[1], edges_to_triangles_connections[0]],
                dtype=torch.int64,
            ),
        )

        self.triangles_to_tetrahedra_connections = triangles_to_tetrahedra_connections

        if len(self.triangles_to_tetrahedra_connections[0]) != 0:
            self.sparse_layer_tetrahedra = SparseLinear(
                num_triangles,
                num_tetrahedra,
                connectivity=torch.tensor(
                    [
                        triangles_to_tetrahedra_connections[1],
                        triangles_to_tetrahedra_connections[0],
                    ],
                    dtype=torch.int64,
                ),
            )

        else:
            self.sparse_layer_tetrahedra = None

    def forward(self, x):
        x_s1 = F.relu(self.sparse_layer_edges(x))

        x_s2 = F.relu(self.sparse_layer_triangles(x_s1))

        if len(self.triangles_to_tetrahedra_connections[0]) != 0:
            x_s3 = F.relu(self.sparse_layer_tetrahedra(x_s2))

            return torch.cat([x_s1, x_s2, x_s3], 1)

        else:

            return torch.cat([x_s1, x_s2], 1)
