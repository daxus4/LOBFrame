import torch
import torch.nn as nn

from models.HNN.hnn import HNN


class ConvolutedMixingHNN(nn.Module):
    @staticmethod
    def get_connections_for_convoluted_mixing_hnn(
        nodes_to_edges_connections
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        num_triangles: int,
        num_tetrahedra: int,
        nodes_to_edges_connections: tuple,
        edges_to_triangles_connections: tuple,
        triangles_to_tetrahedra_connections: tuple,
        num_convolutional_channels: int,
        num_classes: int,
    ):
        super(ConvolutedMixingHNN, self).__init__()

        self.conv_layer_price_vol = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_convolutional_channels,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.ReLU(),
        )

        self.hnn = HNN(  # questa HNN bisogna inizializzarla con i parametri dopo la convoluzione
            num_nodes * num_convolutional_channels,
            num_edges * num_convolutional_channels,
            num_triangles * num_convolutional_channels,
            num_tetrahedra * num_convolutional_channels,
            nodes_to_edges_connections,
            edges_to_triangles_connections,
            triangles_to_tetrahedra_connections,
            num_classes,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        if self.bias is not None:
            x += self.bias.view(1, -1, 1, 1)
        return x
