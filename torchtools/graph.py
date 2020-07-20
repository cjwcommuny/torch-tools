from typing import Tuple

import torch
from torch import Tensor


def full_connected_graph_edges(num_nodes: int) -> Tuple[Tensor, Tensor]:
    sources = torch.arange(num_nodes)
    destinations = torch.arange(num_nodes)
    edges = torch.cartesian_prod(sources, destinations)
    return edges[:, 0], edges[:, 1]

def half_connected_graph_edges(num_nodes: int) -> Tuple[Tensor, Tensor]:
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i <= j]
    edges = torch.tensor(edges, dtype=torch.long)
    return edges[:, 0], edges[:, 1]

def bidirectional_graph_edges(num_nodes: int) -> Tuple[Tensor, Tensor]:
    edges = [[i - 1, i] for i in range(1, num_nodes)]
    edges.extend([[i, i - 1] for i in range(1, num_nodes)])
    edges.extend([i, i] for i in range(num_nodes))
    edges = torch.tensor(edges, dtype=torch.long)
    return edges[:, 0], edges[:, 1]