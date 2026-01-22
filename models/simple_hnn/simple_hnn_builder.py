"""
Simplified HNN builder with constrained sparse layers and a lightweight readout.

Differences vs hnn_builder.py:
  * Readout offers three modes: 'all', 'last', 'orphan'.
      - 'all': each produced layer feeds its own dedicated readout neuron.
      - 'last': only the final layer connects to the readout (single neuron before output).
      - 'orphan': only units without outgoing incidences feed their layer's readout neuron.
  * Readout width equals the number of produced layers (one neuron per layer) before the
    final projection to `out_dim`. This highlights which simplicial order contributed.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "simple_hnn_builder.py requires networkx. Install via `pip install networkx`."
    ) from exc


def _all_cliques_up_to(
    G: nx.Graph,
    max_order: int,
    *,
    max_total_cliques: Optional[int] = None,
) -> Dict[int, List[Tuple[int, ...]]]:
    """Enumerate all cliques up to `max_order` (1 = vertices)."""
    assert max_order >= 1
    out: Dict[int, List[Tuple[int, ...]]] = {1: [(v,) for v in sorted(G.nodes())]}
    if max_order == 1:
        return out

    maximal = list(nx.find_cliques(G))
    total = len(out[1])

    for k in range(2, max_order + 1):
        seen: set[Tuple[int, ...]] = set()
        for clique in maximal:
            if len(clique) < k:
                continue
            nodes = tuple(sorted(clique))
            if len(nodes) == k:
                if nodes not in seen:
                    seen.add(nodes)
                    total += 1
            else:
                for sub in itertools.combinations(nodes, k):
                    if sub not in seen:
                        seen.add(sub)
                        total += 1
                        if max_total_cliques and total > max_total_cliques:
                            out[k] = sorted(seen)
                            return out
        if seen:
            out[k] = sorted(seen)
        else:
            break
    return out


def _incidence_pairs(
    lower: List[Tuple[int, ...]], higher: List[Tuple[int, ...]]
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    higher_sets = [set(h) for h in higher]
    for i, low in enumerate(lower):
        low_set = set(low)
        for j, high_set in enumerate(higher_sets):
            if low_set.issubset(high_set):
                pairs.append((i, j))
    return pairs


class SparseLinear(nn.Module):
    """Linear map constrained by provided (src,dst) connectivity."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: Sequence[Tuple[int, int]],
        *,
        use_sparse_mm: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_sparse_mm = use_sparse_mm

        if len(connectivity) == 0:
            src_idx = torch.empty(0, dtype=torch.long, device=device)
            dst_idx = torch.empty(0, dtype=torch.long, device=device)
        else:
            src_idx = torch.tensor(
                [s for (s, _) in connectivity], dtype=torch.long, device=device
            )
            dst_idx = torch.tensor(
                [t for (_, t) in connectivity], dtype=torch.long, device=device
            )
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        self.weight = nn.Parameter(
            torch.empty(len(self.src_idx), device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )

        if self.use_sparse_mm and len(self.src_idx) > 0:
            indices = torch.stack([self.dst_idx, self.src_idx], dim=0)
            self.register_buffer("_sparse_indices", indices, persistent=False)
        else:
            self.register_buffer(
                "_sparse_indices",
                torch.empty((2, 0), dtype=torch.long),
                persistent=False,
            )

        self.reset_parameters()

    # Initialize weights and biases
    def reset_parameters(self):
        if self.weight.numel() > 0:
            counts = torch.bincount(self.dst_idx, minlength=self.out_features)
            fan_in = max(1, counts.max().item())
            bound = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(self.weight, -bound, bound)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    # Fully sparse forward pass, not relying on dense intermediates followed by masking
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.weight.numel() == 0:
            y = x.new_zeros((B, self.out_features))
        elif self.use_sparse_mm:
            wt = torch.sparse_coo_tensor(
                self._sparse_indices,
                self.weight,
                (self.out_features, self.in_features),
                device=x.device,
                dtype=x.dtype,
            )
            y = torch.sparse.mm(wt, x.transpose(0, 1)).transpose(0, 1)
        else:
            edge_vals = x.index_select(1, self.src_idx) * self.weight
            y = x.new_zeros((B, self.out_features))
            y.scatter_add_(1, self.dst_idx.unsqueeze(0).expand(B, -1), edge_vals)
        if self.bias is not None:
            y = y + self.bias
        return y


# Helper to compute orphan masks per produced layer (useful only if MFCF has threshold)
def _compute_orphan_masks(
    layer_sizes: List[int], connections: List[List[Tuple[int, int]]]
) -> List[np.ndarray]:
    """
    Return boolean masks (per produced layer) marking units without outgoing connections.
    Length equals len(layer_sizes) - 1 (exclude input layer).
    """
    produced = []
    num_layers = len(layer_sizes)
    for idx in range(1, num_layers):
        size = layer_sizes[idx]
        if idx == num_layers - 1:
            mask = np.ones(size, dtype=bool)
        else:
            conn = connections[idx]
            connected = set(src for (src, _) in conn)
            mask = np.ones(size, dtype=bool)
            if connected:
                mask[list(connected)] = False
        produced.append(mask)
    return produced


class SimpleHNN(nn.Module):
    """
    Simplified HNN with per-layer sparse mappings and configurable readout modes:
        - all: aggregate every layer (excluding input) into its dedicated readout neuron.
        - last: use only the final layer to predict.
        - orphan: aggregate only orphan neurons (no outgoing edges) per layer.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        connections: List[List[Tuple[int, int]]],
        *,
        activation: str = "relu",
        readout_mode: str = "all",
        out_dim: int = 1,
        use_sparse_mm: bool = False,
        dropout: float = 0.0,
        layer_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need input and at least one produced layer."
        valid_modes = {"all", "last", "orphan"}
        if readout_mode not in valid_modes:
            raise ValueError(
                f"readout_mode must be one of {valid_modes}; got {readout_mode!r}"
            )
        self.layer_sizes = layer_sizes
        self.readout_mode = readout_mode
        self.out_dim = int(out_dim)

        self.sparse_layers = nn.ModuleList(
            [
                SparseLinear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    connections[i],
                    use_sparse_mm=use_sparse_mm,
                    device=device,
                    dtype=dtype,
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )

        self._use_layer_norm = layer_norm
        if layer_norm:
            self.layer_norms = nn.ModuleList(
                [
                    nn.LayerNorm(layer_sizes[i], device=device, dtype=dtype)
                    for i in range(1, len(layer_sizes))
                ]
            )
        else:
            self.layer_norms = None

        self._dropout_p = float(dropout)
        self._use_dropout = self._dropout_p > 0
        if self._use_dropout:
            self.dropout = nn.Dropout(self._dropout_p)

        self.act = getattr(F, activation) if hasattr(F, activation) else F.relu

        self._produced_sizes = layer_sizes[1:]
        self._num_produced = len(self._produced_sizes)
        self.register_buffer(
            "_dummy",
            torch.tensor(0),
            persistent=False,
        )  # placeholder to keep dtype/device defaults handy

        self._orphan_masks = _compute_orphan_masks(layer_sizes, connections)
        self._init_readout_params(device=device, dtype=dtype)

    def _init_readout_params(self, device=None, dtype=None):
        if self.readout_mode == "last":
            last_size = self._produced_sizes[-1]
            self.last_linear = nn.Linear(
                last_size, self.out_dim, device=device, dtype=dtype
            )  # last layer directly to output
        else:
            # One aggregator per produced layer (could be unused if no orphans)
            aggregators = []
            for sz, mask in zip(self._produced_sizes, self._orphan_masks):
                if self.readout_mode == "all":
                    aggregators.append(nn.Linear(sz, 1, device=device, dtype=dtype))
                else:  # orphan mode
                    orphan_count = int(mask.sum())
                    if orphan_count > 0:
                        aggregators.append(
                            nn.Linear(orphan_count, 1, device=device, dtype=dtype)
                        )
                    else:
                        aggregators.append(None)
            self.layer_aggregators = nn.ModuleList(
                [agg for agg in aggregators if agg is not None]
            )
            # Keep index mapping since ModuleList cannot contain None
            self._agg_indices: List[Optional[int]] = []
            cursor = 0
            for agg in aggregators:
                if agg is None:
                    self._agg_indices.append(None)
                else:
                    self._agg_indices.append(cursor)
                    cursor += 1
            self.final_linear = nn.Linear(
                self._num_produced, self.out_dim, device=device, dtype=dtype
            )

    # Helper to aggregate a produced layer's features into its readout contribution
    def _aggregate_layer(self, layer_idx: int, feats: torch.Tensor) -> torch.Tensor:
        """Return (B,1) contribution for the given produced layer."""
        if self.readout_mode == "all":
            agg = self.layer_aggregators[self._agg_indices[layer_idx]]  # type: ignore[index]
            return agg(feats)
        # orphan mode
        mask = self._orphan_masks[layer_idx]
        if mask.sum() == 0:
            return torch.zeros(
                feats.shape[0], 1, device=feats.device, dtype=feats.dtype
            )
        agg_idx = self._agg_indices[layer_idx]
        assert agg_idx is not None, "Aggregator index mismatch for orphan mode."
        agg = self.layer_aggregators[agg_idx]
        sel = feats[:, torch.from_numpy(mask).to(feats.device)]
        return agg(sel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        h = x
        for idx, layer in enumerate(self.sparse_layers):
            h = layer(h)
            if self._use_layer_norm:
                h = self.layer_norms[idx](h)
            h = self.act(h)
            if self._use_dropout:
                h = self.dropout(h)
            activations.append(h)

        produced = activations  # includes all non-input layers in order

        if self.readout_mode == "last":
            logits = self.last_linear(produced[-1])
        else:
            per_layer = [
                self._aggregate_layer(i, feats) for i, feats in enumerate(produced)
            ]
            stacked = torch.cat(per_layer, dim=1)
            logits = self.final_linear(stacked)

        if self.out_dim == 1:
            return logits.view(-1)
        return logits


def _build_layer_data_from_cliques(
    cliques: Dict[int, List[Tuple[int, ...]]],
) -> Tuple[List[int], List[List[Tuple[int, int]]]]:
    if 1 not in cliques:
        raise ValueError("Clique dictionary must include vertices under key 1.")
    layer_sizes = [len(cliques[1])]
    connections: List[List[Tuple[int, int]]] = []

    # iterate through higher orders in ascending order, but only build a layer
    # when the previous order exists to provide connectivity
    sorted_orders = sorted(k for k in cliques.keys() if k >= 2 and cliques[k])
    for k in sorted_orders:
        lower = k - 1
        if lower not in cliques or not cliques[lower]:
            raise ValueError(
                f"Missing cliques of size {lower} required to connect to order {k}."
            )
        layer_sizes.append(len(cliques[k]))
        connections.append(_incidence_pairs(cliques[lower], cliques[k]))
    if len(layer_sizes) < 2:
        raise ValueError(
            "Need at least one higher-order clique layer to build the network."
        )
    return layer_sizes, connections


def build_simple_hnn_from_adjacency(
    J: np.ndarray | torch.Tensor,
    *,
    max_order: int = 3,
    max_total_cliques: Optional[int] = None,
    use_abs_threshold: Optional[float] = None,
    activation: str = "relu",
    readout_mode: str = "all",
    out_dim: int = 1,
    use_sparse_mm: bool = False,
    dropout: float = 0.0,
    layer_norm: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[SimpleHNN, Dict]:
    """Build a SimpleHNN from a (symmetric) adjacency/precision matrix."""
    J_np = J.detach().cpu().numpy() if isinstance(J, torch.Tensor) else np.asarray(J)
    assert (
        J_np.ndim == 2 and J_np.shape[0] == J_np.shape[1]
    ), "Adjacency must be square."

    A = J_np.copy()
    if use_abs_threshold is not None:
        A[np.abs(A) < use_abs_threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)

    G = nx.from_numpy_array(A)
    cliques = _all_cliques_up_to(
        G, max_order=max_order, max_total_cliques=max_total_cliques
    )
    layer_sizes, connections = _build_layer_data_from_cliques(cliques)

    model = SimpleHNN(
        layer_sizes=layer_sizes,
        connections=connections,
        activation=activation,
        readout_mode=readout_mode,
        out_dim=out_dim,
        use_sparse_mm=use_sparse_mm,
        dropout=dropout,
        layer_norm=layer_norm,
        device=device,
        dtype=dtype,
    )
    meta = {
        "layer_sizes": layer_sizes,
        "connections": connections,
        "cliques": cliques,
        "graph_nx": G,
    }
    return model, meta


# can take either clique dict or list of maximal cliques(from MFCF directly)
def build_simple_hnn_from_cliques(
    cliques: Dict[int, List[Tuple[int, ...]]] | Sequence[Sequence[int]],
    *,
    activation: str = "relu",
    readout_mode: str = "all",
    out_dim: int = 1,
) -> Tuple[SimpleHNN, Dict]:
    """Construct SimpleHNN directly from a clique dictionary."""
    if isinstance(cliques, list) or isinstance(cliques, tuple):
        cliques = expand_maximal_cliques(cliques)
    layer_sizes, connections = _build_layer_data_from_cliques(cliques)
    model = SimpleHNN(
        layer_sizes=layer_sizes,
        connections=connections,
        activation=activation,
        readout_mode=readout_mode,
        out_dim=out_dim,
    )
    meta = {"layer_sizes": layer_sizes, "connections": connections, "cliques": cliques}
    return model, meta


# Helper to expand maximal cliques into full clique dictionary for builder
def expand_maximal_cliques(
    maximal_cliques: Sequence[Sequence[int]],
    *,
    n_vertices: Optional[int] = None,
    max_order: Optional[int] = None,
) -> Dict[int, List[Tuple[int, ...]]]:
    """
    Expand a list of maximal cliques into the clique dictionary expected by the builder.
    """
    from collections import defaultdict

    if n_vertices is None:
        if not maximal_cliques:
            raise ValueError("Cannot infer vertex count from empty clique list.")
        n_vertices = max(max(clique) for clique in maximal_cliques) + 1

    layers: Dict[int, set[Tuple[int, ...]]] = defaultdict(set)
    for v in range(n_vertices):
        layers[1].add((v,))

    for clique in maximal_cliques:
        ordered = tuple(sorted(clique))
        max_sz = len(ordered)
        if max_order is not None:
            max_sz = min(max_sz, max_order)
        for r in range(2, max_sz + 1):
            for subset in itertools.combinations(ordered, r):
                layers[r].add(subset)

    return {k: sorted(v) for k, v in layers.items() if v}


__all__ = [
    "SimpleHNN",
    "SparseLinear",
    "build_simple_hnn_from_adjacency",
    "build_simple_hnn_from_cliques",
    "expand_maximal_cliques",
]
