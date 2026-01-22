from copy import deepcopy

import torch.nn as nn

from models.simple_hnn.simple_hnn_builder import SimpleHNN


class SimpleSTHNN(nn.Module):
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
        hnn: SimpleHNN,
        num_convolutional_channels: int,
        lighten: bool = False,
    ):
        super(SimpleSTHNN, self).__init__()
        self.name = "sthnn"
        if lighten:
            self.name += "-lighten"

        self.hnn = hnn
        self.num_convolutional_channels = num_convolutional_channels

        self.conv_layer_price_vol = nn.Sequential(
            nn.Conv1d(
                in_channels=2,  # price, volume
                out_channels=num_convolutional_channels,
                kernel_size=1,  # mix price & volume only
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        # x.shape = (batch_size, 1, times, spatial_features * 2)
        # where spatial_features * 2 is for price and volume ordered as: [ask_price_1, ask_volume_1, bid_price_1, bid_volume_1, ...]
        # times is the number of time lags. The row 0 is the less recent, the row -1 is the most recent.

        B, _, T, S2 = x.shape
        S = S2 // 2  # number of spatial levels

        # 1. Remove the singleton channel
        x = x.squeeze(1)  # (B, T, S2)
        # 2. Separate price and volume
        x = x.view(B, T, S, 2)  # (B, T, S, 2)
        # last dim: [price, volume]
        # 3. Move feature dim to channels
        x = x.permute(0, 3, 1, 2)  # (B, 2, T, S)
        # 4. Collapse (time, spatial) into length
        # Ordered as [ask_*_1_lag100, bid_*_1_lag100, ask_*_2_lag100..., ask_*_1_lag0, bid_*_1_lag0, ask_*_2_lag0...]
        x = x.reshape(B, 2, T * S)  # (B, 2, T*S)

        # Convolution of price and volume. C = num_convolutional_channels
        x = self.conv_layer_price_vol(x)  # (B, C, T * S)

        # Flatten the convoluted features
        # Ordered as [ask_conv_1_lag100_chan1, ask_conv_1_lag100_chan2, ..., bid_conv_1_lag100_chan1, ...,
        # ask_conv_1_lag0_chan1, ask_conv_1_lag0_chan2, ..., bid_conv_1_lag0_chan1, ... ]
        x = x.permute(0, 2, 1).reshape(B, T * S * self.num_convolutional_channels)

        x = self.hnn(x)

        return x
