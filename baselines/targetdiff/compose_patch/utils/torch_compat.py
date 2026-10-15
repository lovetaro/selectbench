"""Pure-PyTorch replacements for torch_scatter / torch_sparse functions used in TargetDiff."""

import torch


def _expand_index(index, src, dim):
    shape = [1] * src.dim()
    shape[dim] = -1
    return index.view(shape).expand_as(src)


def scatter_sum(src, index, dim=0, dim_size=None, out=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    if out is None:
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = _expand_index(index, src, dim)
    out.scatter_add_(dim, idx, src)
    return out


# scatter_add is an alias for scatter_sum
scatter_add = scatter_sum


def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = scatter_sum(src, index, dim=dim, dim_size=dim_size)
    count_1d = torch.bincount(index, minlength=dim_size).to(dtype=src.dtype, device=src.device)
    shape = [1] * src.dim()
    shape[dim] = dim_size
    count = count_1d.view(shape)
    return out / count.clamp(min=1)


def scatter_softmax(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    idx = _expand_index(index, src, dim)
    # numerically stable: subtract per-group max
    max_vals = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
    max_vals.scatter_reduce_(dim, idx, src, reduce='amax', include_self=True)
    src_shifted = src - max_vals.gather(dim, idx)
    src_exp = src_shifted.exp()
    sum_exp = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    sum_exp.scatter_add_(dim, idx, src_exp)
    return src_exp / sum_exp.gather(dim, idx).clamp(min=1e-16)


def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):
    """Generic scatter matching the torch_scatter.scatter interface."""
    if reduce in ('sum', 'add'):
        return scatter_sum(src, index, dim=dim, dim_size=dim_size, out=out)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim=dim, dim_size=dim_size)
    elif reduce in ('min', 'amin'):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        result = torch.full(out_shape, float('inf'), dtype=src.dtype, device=src.device)
        idx = _expand_index(index, src, dim)
        result.scatter_reduce_(dim, idx, src, reduce='amin', include_self=True)
        return result
    elif reduce in ('max', 'amax'):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        result = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
        idx = _expand_index(index, src, dim)
        result.scatter_reduce_(dim, idx, src, reduce='amax', include_self=True)
        return result
    else:
        raise ValueError(f'Unsupported reduce: {reduce}')


def segment_coo(src, index, out=None, dim_size=None, reduce='sum'):
    """Pure-PyTorch replacement for torch_scatter.segment_coo.

    segment_coo assumes a sorted index (contiguous segments).  We delegate to
    scatter_sum / scatter which are strictly more general.
    """
    return scatter(src, index, dim=0, out=out, dim_size=dim_size, reduce=reduce)


# ---------------------------------------------------------------------------
# Pure-PyTorch replacements for torch_cluster / torch_geometric.nn graph ops
# ---------------------------------------------------------------------------

def knn_graph(x, k, batch=None, flow='source_to_target', loop=False, cosine=False):
    """Pure-PyTorch replacement for torch_geometric.nn.knn_graph / torch_cluster.knn_graph.

    For each node i, finds its k nearest neighbours j within the same batch
    element (excluding self unless loop=True).

    flow='source_to_target': edge_index = [neighbours_j, query_i]   (j -> i)
    flow='target_to_source': edge_index = [query_i, neighbours_j]   (i -> j)
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    num_batches = int(batch.max().item()) + 1

    rows, cols = [], []
    for b in range(num_batches):
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        x_b = x[idx]
        n_b = x_b.size(0)
        actual_k = min(k + (0 if loop else 1), n_b)
        if actual_k <= 0:
            continue

        if cosine:
            x_norm = torch.nn.functional.normalize(x_b, p=2, dim=-1)
            dist = 1.0 - x_norm @ x_norm.t()
        else:
            dist = torch.cdist(x_b, x_b)

        _, top_idx = dist.topk(actual_k, dim=-1, largest=False)

        if not loop:
            # top_idx[:, 0] is always self (dist=0); skip it
            top_idx = top_idx[:, 1:]

        real_k = top_idx.size(1)
        # src_local: neighbour indices (j), dst_local: query indices (i)
        src_local = top_idx.reshape(-1)
        dst_local = torch.arange(n_b, device=x.device).unsqueeze(1).expand(-1, real_k).reshape(-1)

        rows.append(idx[src_local])  # j in global index
        cols.append(idx[dst_local])  # i in global index

    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    j = torch.cat(rows)
    i = torch.cat(cols)
    # flow='source_to_target': message travels j->i, so edge_index = [j, i]
    if flow == 'source_to_target':
        return torch.stack([j, i], dim=0)
    else:  # 'target_to_source': edge_index = [i, j]
        return torch.stack([i, j], dim=0)


def radius_graph(x, r, batch=None, loop=False, flow='source_to_target'):
    """Pure-PyTorch replacement for torch_geometric.nn.radius_graph / torch_cluster.radius_graph.

    Connects each node to all neighbours within radius r in the same batch element.

    flow='source_to_target': edge_index = [neighbour_j, query_i]
    flow='target_to_source': edge_index = [query_i, neighbour_j]
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    num_batches = int(batch.max().item()) + 1

    rows, cols = [], []
    for b in range(num_batches):
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        x_b = x[idx]
        dist = torch.cdist(x_b, x_b)  # (n_b, n_b)
        mask = dist < r
        if not loop:
            mask.fill_diagonal_(False)
        # src_local=j (neighbour), dst_local=i (query)
        dst_local, src_local = mask.nonzero(as_tuple=True)
        rows.append(idx[src_local])
        cols.append(idx[dst_local])

    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    j = torch.cat(rows)
    i = torch.cat(cols)
    if flow == 'source_to_target':
        return torch.stack([j, i], dim=0)
    else:
        return torch.stack([i, j], dim=0)
