import torch
import torch.nn as nn


class SparseLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        connectivity: user defined sparsity matrix
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        coalesce_device: device to coalesce the sparse matrix on
            Default: 'gpu'
        max_size (int): maximum number of entries allowed before chunking occurrs
            Default: 1e8

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples:

        >>> m = nn.SparseLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features,
        out_features,
        connectivity,
        bias=True,
        coalesce_device="cuda",
        max_size=1e8,
    ):
        assert in_features < 2**31 and out_features < 2**31
        if connectivity is not None:
            assert isinstance(connectivity, torch.LongTensor) or isinstance(
                connectivity,
                torch.cuda.LongTensor,
            ), "Connectivity must be a Long Tensor"
            assert (
                connectivity.shape[0] == 2 and connectivity.shape[1] > 0
            ), "Input shape for connectivity should be (2,nnz)"
            assert (
                connectivity.shape[1] <= in_features * out_features
            ), "Nnz can't be bigger than the weight matrix"
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity = connectivity
        self.max_size = max_size

        nnz = connectivity.shape[1]
        connectivity = connectivity.to(device=coalesce_device)
        indices = connectivity

        values = torch.empty(nnz, device=coalesce_device)

        self.register_buffer("indices", indices.cpu())
        self.weights = nn.Parameter(values.cpu())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        """returns a torch.sparse_coo_tensor view of the underlying weight matrix
        This is only for inspection purposes and should not be modified or used in any autograd operations
        """
        weight = torch.sparse_coo_tensor(
            self.indices,
            self.weights,
            (self.out_features, self.in_features),
        )
        return weight.coalesce().detach()

    def forward(self, inputs):
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        if len(output_shape) == 1:
            inputs = inputs.view(1, -1)
        inputs = inputs.flatten(end_dim=-2)

        target = torch.sparse_coo_tensor(
            self.indices,
            self.weights,
            torch.Size([self.out_features, self.in_features]),
        )
        output = torch.sparse.mm(target, inputs.t()).t()

        if self.bias is not None:
            output += self.bias

        return output.view(output_shape)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, connectivity={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.connectivity,
        )


if __name__ == "__main__":
    # Example usage
    sl = SparseLinear(
        4,
        4,
        connectivity=torch.tensor(
            [[[0, 1, 1, 2, 1, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]]]
        ),
    )
    x = torch.randn(2, 4)
    output = sl(x)
    print(output)
