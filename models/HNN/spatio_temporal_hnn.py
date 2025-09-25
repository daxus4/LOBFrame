from copy import deepcopy

import torch
import torch.nn as nn

from models.HNN.hnn import HNN, GraphHomologicalStructure


class SpatioTemporalHNN(nn.Module):
    @staticmethod
    def get_connections_for_convoluted_mixing_hnn(
        nodes_to_edges_connections: tuple,
        num_convolutional_channels: int,
    ) -> tuple:
        """
        This function modifies the connections for the convoluted mixing HNN.
        It expands the nodes_to_edges_connections to account for the convolutional channels.
        """
        new_nodes_to_edges_connections = ([], [])
        for connection_index in range(len(nodes_to_edges_connections[0])):
            node_index = nodes_to_edges_connections[0][connection_index]
            edge_index = nodes_to_edges_connections[1][connection_index]

            for channel in range(num_convolutional_channels):
                new_nodes_to_edges_connections[0].append(
                    node_index * num_convolutional_channels + channel
                )
                new_nodes_to_edges_connections[1].append(edge_index)

        return new_nodes_to_edges_connections

    def __init__(
        self,
        homological_structure: GraphHomologicalStructure,
        num_convolutional_channels: int,
        num_classes: int = 3,
        lighten: bool = False,
    ):
        super(SpatioTemporalHNN, self).__init__()
        self.name = "hcnn"
        if lighten:
            self.name += "-lighten"

        self.homological_structure = homological_structure
        self.num_classes = num_classes

        self.conv_layer_price_vol = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=num_convolutional_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
        )

        convoluted_nodes_to_edges_connections = (
            self.get_connections_for_convoluted_mixing_hnn(
                homological_structure.nodes_to_edges_connections,
                num_convolutional_channels,
            )
        )

        self.convoluted_homological_structure = deepcopy(homological_structure)
        self.convoluted_homological_structure.nodes_to_edges_connections = (
            convoluted_nodes_to_edges_connections
        )

        self.hnn = HNN(self.convoluted_homological_structure)

        self.readout_layer = nn.Linear(
            in_features=homological_structure.num_edges
            + homological_structure.num_triangles
            + homological_structure.num_tetrahedra,
            out_features=num_classes,
        )

    def forward(self, x):
        # x.shape = (batch_size, 1, num_spatiotemporal_features_already_pruned)

        # DO NOT WATCH THIS PART, IT IS OLD AND NOT USED ANYMORE
        # x.shape = (batch_size, 1, num_window_lags, num_spatial_features) num_spatial_features è della dimensione di tutti i nodi (nodi nel senso di spaziali quindi senza lag) * 2 perche c'è price and volume
        # After these -> x.shape = (batch_size, 1, num_features) num_features è della dimensione di tutti i nodi (nodi nel senso di spazio-temporali, quindi vol1ask_lag0, vol1ask_lag1, ...) * 2 perche c'è price and volume
        # x = torch.flip(x, dims=[2])
        # x = x.reshape(x.shape[0], 1, -1)
        # END DO NOT WATCH

        # after conv_layer_price_vol -> x.shape = (batch_size, num_convolutional_channels, num_features // 2)
        x = self.conv_layer_price_vol(x)

        # after flatten -> # x.shape = (batch_size, num_convolutional_channels * num_features // 2)
        # Permute to have channels first, then flatten. so the columns will be feature_channel1, feature_channel2, ..., feature_channelN
        x = x.permute(0, 2, 1).flatten(start_dim=1)

        x = self.hnn(x)  # x.shape = (batch_size, num_classes)

        # after hnn -> x.shape = (batch_size, num_edges + num_triangles + num_tetrahedra)
        x = self.readout_layer(x)  # x.shape = (batch_size, num

        return x
