"""Pure-PyTorch replacements for torch_scatter functions used in DecompDiff."""

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


def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = scatter_sum(src, index, dim=dim, dim_size=dim_size)
    count_1d = torch.bincount(index, minlength=dim_size).to(dtype=src.dtype, device=src.device)
    shape = [1] * src.dim()
    shape[dim] = dim_size
    count = count_1d.view(shape)
    return out / count.clamp(min=1)


def scatter_min(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.full(out_shape, float('inf'), dtype=src.dtype, device=src.device)
    idx = _expand_index(index, src, dim)
    out.scatter_reduce_(dim, idx, src, reduce='amin', include_self=True)
    # return (values, argmin) to match torch_scatter.scatter_min signature
    argmin = torch.zeros(out_shape, dtype=torch.long, device=src.device)
    return out, argmin


def knn_graph(x, k, batch=None, flow='source_to_target', loop=False, cosine=False):
    """Pure-PyTorch replacement for torch_cluster.knn_graph.

    For each node i, finds its k nearest neighbours j within the same batch
    element (excluding self unless loop=True).

    Returns edge_index of shape [2, N*k] where:
      flow='source_to_target': edge_index = [neighbours, query_nodes]
      flow='target_to_source': edge_index = [query_nodes, neighbours]
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
        src_local = top_idx.reshape(-1)
        dst_local = torch.arange(n_b, device=x.device).unsqueeze(1).expand(-1, real_k).reshape(-1)

        rows.append(idx[src_local])
        cols.append(idx[dst_local])

    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    src = torch.cat(rows)
    dst = torch.cat(cols)
    return torch.stack([src, dst] if flow == 'source_to_target' else [dst, src], dim=0)


def edge_triplets(edge_index, num_nodes):
    """Pure-PyTorch replacement for the SparseTensor-based triplet computation in BondUpdateLayer.

    Given directed edges j->i (edge_index[0]=j, edge_index[1]=i), returns all
    k->j->i triplets (k != i).  Matches the return signature of the original
    torch_sparse implementation:
        col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
    where idx_kj / idx_ji are edge indices into the original edge_index.
    """
    row, col = edge_index  # row=j (source), col=i (destination)
    E = row.size(0)
    device = row.device

    # For each node v, collect incoming edge indices (edges e where col[e] == v).
    sort_by_dest = torch.argsort(col)
    deg_in = torch.zeros(num_nodes, dtype=torch.long, device=device)
    deg_in.scatter_add_(0, col, torch.ones(E, dtype=torch.long, device=device))
    ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    ptr[1:] = deg_in.cumsum(0)

    # Number of triplets produced by each edge e = deg_in[j_e].
    num_triplets_per_edge = deg_in[row]  # [E]
    T = int(num_triplets_per_edge.sum().item())

    # For each triplet, which edge e (j->i) does it extend?
    edge_of_triplet = torch.arange(E, device=device).repeat_interleave(num_triplets_per_edge)

    # Local 0-based offset within each edge's group of triplets.
    group_start = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        num_triplets_per_edge.cumsum(0)[:-1],
    ])[edge_of_triplet]
    local_pos = torch.arange(T, device=device) - group_start

    # Retrieve the k->j incoming-edge index and its source node k.
    base_ptr = ptr[row[edge_of_triplet]]
    idx_kj = sort_by_dest[base_ptr + local_pos]
    idx_k = row[idx_kj]

    idx_i = col[edge_of_triplet]
    idx_j = row[edge_of_triplet]
    idx_ji = edge_of_triplet

    # Remove self-loops in the triplet (i == k).
    mask = idx_i != idx_k
    return col, row, idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]


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
