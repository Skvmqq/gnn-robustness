import torch
from torch_geometric.utils import dropout_edge
import random
import numpy as np
import torch.nn.functional as F


def _probabilistic_remove_edges(edge_index, percent, scores, is_undirected):
    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float, device=edge_index.device)
    else:
        scores = scores.to(device=edge_index.device, dtype=torch.float)

    row, col = edge_index
    edge_scores = (scores[row] + scores[col]) / 2.0

    if is_undirected:
        mask = row < col
        unique_edges = edge_index[:, mask]
        unique_scores = edge_scores[mask]
    else:
        unique_edges = edge_index
        unique_scores = edge_scores

    if unique_edges.size(1) == 0:
        return edge_index

    if unique_scores.sum() <= 0:
        probs = torch.ones_like(unique_scores) / unique_scores.numel()
    else:
        probs = unique_scores / unique_scores.sum()

    num_edges = unique_edges.size(1)
    num_remove = int(num_edges * percent)
    num_remove = min(max(num_remove, 0), num_edges)

    if num_remove > 0:
        remove_idx = torch.multinomial(probs, num_remove, replacement=False)
        edges_to_remove = unique_edges[:, remove_idx]
    else:
        return edge_index

    num_nodes = int(edge_index.max().item()) + 1
    if is_undirected:
        all_row, all_col = edge_index
        all_u = torch.minimum(all_row, all_col)
        all_v = torch.maximum(all_row, all_col)
        all_keys = all_u * num_nodes + all_v

        rem_row, rem_col = edges_to_remove
        rem_u = torch.minimum(rem_row, rem_col)
        rem_v = torch.maximum(rem_row, rem_col)
        rem_keys = rem_u * num_nodes + rem_v
    else:
        all_row, all_col = edge_index
        all_keys = all_row * num_nodes + all_col
        rem_row, rem_col = edges_to_remove
        rem_keys = rem_row * num_nodes + rem_col

    keep_mask = ~torch.isin(all_keys, rem_keys)
    new_edge_index = edge_index[:, keep_mask]

    if is_undirected:
        row, col = new_edge_index
        rev_edges = torch.stack([col, row], dim=0)
        new_edge_index = torch.cat([new_edge_index, rev_edges], dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

    return new_edge_index


def feature_similarity_rewiring(
    x,
    edge_index,
    mode="fixed",
    threshold=0.5,
    drop_ratio=0.1,
):
    row, col = edge_index
    x_src = x[row]
    x_dst = x[col]
    sim_scores = F.cosine_similarity(x_src, x_dst, dim=1, eps=1e-8)

    if mode == "fixed":
        keep_mask = sim_scores > threshold
        return edge_index[:, keep_mask]

    if mode == "adaptive":
        num_edges = edge_index.size(1)
        if num_edges == 0 or drop_ratio <= 0:
            return edge_index

        num_drop = int(num_edges * drop_ratio)
        num_drop = min(max(num_drop, 0), num_edges - 1)
        num_keep = num_edges - num_drop

        if num_keep >= num_edges:
            return edge_index

        _, keep_idx = torch.topk(sim_scores, k=num_keep, largest=True)
        return edge_index[:, keep_idx]

    raise ValueError(f"Unknown rewiring mode: {mode}")


def augment_features(x, mode="noise", percent=0.1):
    x_augmented = x.clone()

    if mode == "noise":
        noise = torch.randn_like(x) * percent
        return x + noise

    elif mode == "missing":
        mask = torch.rand_like(x) < percent
        x_augmented[mask] = 0
        return x_augmented

    elif mode == "flip":
        mask = torch.rand_like(x) < percent
        x_augmented[mask] = 1 - x_augmented[mask]
        return x_augmented

    else:
        raise ValueError(f"Unknown mode: {mode}")


def augment_edges(
    edge_index,
    mode="dropout",
    percent=0.1,
    k=None,
    scores=None,
    hub_nodes=None,
    is_undirected=True,
):
    if mode == "dropout":
        edge_index, _ = dropout_edge(
            edge_index,
            p=percent,
            force_undirected=is_undirected,
        )
        return edge_index

    elif mode == "high_deg":
        if k is None or k <= 0:
            raise ValueError("For mode='high_deg', provide k > 0")

        degree = torch.bincount(edge_index.view(-1))
        _, high_deg_nodes = torch.topk(degree, k)

        mask_row = ~torch.isin(edge_index[0], high_deg_nodes)
        mask_col = ~torch.isin(edge_index[1], high_deg_nodes)
        mask = mask_row & mask_col
        new_edge_index = edge_index[:, mask]

        if is_undirected:
            row, col = new_edge_index
            rev_edges = torch.stack([col, row], dim=0)
            new_edge_index = torch.cat([new_edge_index, rev_edges], dim=1)
            new_edge_index = torch.unique(new_edge_index, dim=1)

        return new_edge_index

    elif mode == "hub_nodes":
        if hub_nodes is None:
            raise ValueError("For mode='hub_nodes', provide hub_nodes")

        if not torch.is_tensor(hub_nodes):
            hub_nodes = torch.tensor(hub_nodes, dtype=edge_index.dtype, device=edge_index.device)
        else:
            hub_nodes = hub_nodes.to(device=edge_index.device, dtype=edge_index.dtype)

        mask_row = ~torch.isin(edge_index[0], hub_nodes)
        mask_col = ~torch.isin(edge_index[1], hub_nodes)
        mask = mask_row & mask_col
        new_edge_index = edge_index[:, mask]

        if is_undirected:
            row, col = new_edge_index
            rev_edges = torch.stack([col, row], dim=0)
            new_edge_index = torch.cat([new_edge_index, rev_edges], dim=1)
            new_edge_index = torch.unique(new_edge_index, dim=1)

        return new_edge_index


    elif mode in {"bw_prob", "pagerank_prob", "degree_prob", "closeness_prob", "eigenvector_prob"}:
        if scores is None:
            raise ValueError(f"For mode='{mode}', provide scores")
        return _probabilistic_remove_edges(edge_index, percent, scores, is_undirected)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def masked_accuracy(logits, y, mask):
    pred = logits.argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()


