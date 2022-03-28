import torch


def groupby_reduce(sample_indexer: torch.Tensor, sample_values: torch.Tensor, op: str = "max"):
    """
    Group-By and Reduce sample_values according to their indices, the reduction operation is defined in `op`.
    :param sample_indexer: (N,). An index, must start from 0 and go to the (max-1), can be obtained using torch.unique.
    :param sample_values: (N, L)
    :param op: have to be in 'max', 'mean'
    :return: reduced values: (C, L)
    """
    C = sample_indexer.max() + 1
    n_samples = sample_indexer.size(0)

    assert n_samples == sample_values.size(0), "Indexer and Values must agree on sample count!"

    sample_values = sample_values.contiguous()
    sample_indexer = sample_indexer.contiguous()
    if op == 'mean':
        from src.ext import groupby_sum
        values_sum, values_count = groupby_sum(sample_values, sample_indexer, C)
        return values_sum / values_count.unsqueeze(-1)
    elif op == 'sum':
        from src.ext import groupby_sum
        values_sum, _ = groupby_sum(sample_values, sample_indexer, C)
        return values_sum
    else:
        raise NotImplementedError


def get_samples(r: int, device: torch.device, a: float = 0.0, b: float = None):
    """
    Get samples within a cube, the voxel size is (b-a)/(r-1). range is from [a, b]
    :param r: num samples
    :param a: bound min
    :param b: bound max
    :return: (r*r*r, 3)
    """
    overall_index = torch.arange(0, r ** 3, 1, device=device, dtype=torch.long)
    r = int(r)

    if b is None:
        b = 1. - 1. / r

    vsize = (b - a) / (r - 1)
    samples = torch.zeros(r ** 3, 3, device=device, dtype=torch.float32)
    samples[:, 0] = (overall_index // (r * r)) * vsize + a
    samples[:, 1] = ((overall_index // r) % r) * vsize + a
    samples[:, 2] = (overall_index % r) * vsize + a

    return samples