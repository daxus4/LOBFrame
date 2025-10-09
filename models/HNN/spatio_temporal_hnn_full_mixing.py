from copy import deepcopy

import torch
import torch.nn as nn

from models.HNN.hnn import HNN, GraphHomologicalStructure


class SpatioTemporalHNNFullMixing(nn.Module):
    @staticmethod
    def get_connections_for_convoluted_mixing_hnn(
        connections: tuple,
        num_convolutional_channels: int,
    ) -> tuple:
        """
        This function modifies the connections for the convoluted mixing HNN.
        It expands the nodes_to_edges_connections to account for the convolutional channels.
        """
        convoluted_connections = ([], [])

        for connection_index in range(len(connections[0])):
            for n_iter_over_nodes in range(num_convolutional_channels):
                for n_iter_over_edges in range(num_convolutional_channels):
                    convoluted_connections[0].append(
                        connections[0][connection_index] * num_convolutional_channels
                        + n_iter_over_nodes
                    )
                    convoluted_connections[1].append(
                        connections[1][connection_index] * num_convolutional_channels
                        + n_iter_over_edges
                    )

        return convoluted_connections

    def __init__(
        self,
        homological_structure: GraphHomologicalStructure,
        num_convolutional_channels: int,
        num_classes: int = 3,
        lighten: bool = False,
    ):
        super(SpatioTemporalHNNFullMixing, self).__init__()
        self.name = "sthnnfm"
        if lighten:
            self.name += "-lighten"

        self.homological_structure = homological_structure
        self.num_classes = num_classes
        self.num_convolutional_channels = num_convolutional_channels

        self.conv_layer_price_vol = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.num_convolutional_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
        )

        convoluted_nodes_to_edges_connections = (
            self.get_connections_for_convoluted_mixing_hnn(
                homological_structure.nodes_to_edges_connections,
                self.num_convolutional_channels,
            )
        )
        convoluted_edges_to_triangles_connections = (
            self.get_connections_for_convoluted_mixing_hnn(
                homological_structure.edges_to_triangles_connections,
                self.num_convolutional_channels,
            )
        )
        convoluted_triangles_to_tetrahedra_connections = (
            self.get_connections_for_convoluted_mixing_hnn(
                homological_structure.triangles_to_tetrahedra_connections,
                self.num_convolutional_channels,
            )
        )

        self.convoluted_homological_structure = deepcopy(homological_structure)
        self.convoluted_homological_structure.nodes_to_edges_connections = (
            convoluted_nodes_to_edges_connections
        )
        self.convoluted_homological_structure.edges_to_triangles_connections = (
            convoluted_edges_to_triangles_connections
        )
        self.convoluted_homological_structure.triangles_to_tetrahedra_connections = (
            convoluted_triangles_to_tetrahedra_connections
        )

        self.hnn = HNN(self.convoluted_homological_structure)

        self.readout_layer = nn.Linear(
            in_features=(
                homological_structure.num_edges
                + homological_structure.num_triangles
                + homological_structure.num_tetrahedra
            )
            * self.num_convolutional_channels,
            out_features=num_classes,
        )

    def forward(self, x):
        # x.shape = (batch_size, 1, num_spatiotemporal_features_already_pruned)
        # after conv_layer_price_vol -> x.shape = (batch_size, num_convolutional_channels, num_features // 2)
        x = self.conv_layer_price_vol(x)

        # after flatten -> # x.shape = (batch_size, num_convolutional_channels * num_features // 2)
        # Permute to have channels first, then flatten. so the columns will be feature_channel1, feature_channel2, ..., feature_channelN
        x = x.permute(0, 2, 1).flatten(start_dim=1)

        x = self.hnn(x)

        # after hnn -> x.shape = (batch_size, (num_edges + num_triangles + num_tetrahedra) * num_convolutional_channels)
        x = self.readout_layer(x)  # x.shape = (batch_size, num_classes)

        return x
