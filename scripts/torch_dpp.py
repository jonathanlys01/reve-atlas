"""GPU-backed greedy MAP helpers for large offline display populations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def greedy_map(
    similarity: Any,
    utility: np.ndarray,
    method: str,
    interaction: float,
    selection_k: int,
    epsilon: float,
    tie_order: np.ndarray | None = None,
    tie_tolerance: float | None = None,
) -> tuple[list[int], list[float], float]:
    """Run deterministic single-start greedy MAP on a full per-index pool.

    The candidate Gram matrix is still the complete display-index matrix, but
    Cholesky residuals are updated on the GPU without materializing a second
    candidate-by-candidate workspace for every greedy step. ``tie_order``/``tie_tolerance``
    apply the same near-tie rule as ``real_atlas._greedy_select``.
    """
    import torch

    matrix = similarity
    candidate_count = int(matrix.shape[0])
    if matrix.ndim != 2 or matrix.shape[1] != candidate_count:
        raise ValueError("DPP similarity must be square")
    if not 0 < selection_k <= candidate_count:
        raise ValueError(f"selection_k={selection_k} must be in [1, {candidate_count}]")
    utility_tensor = torch.as_tensor(utility, dtype=matrix.dtype, device=matrix.device)
    if method == "multiplicative":
        weights = torch.exp(float(interaction) * (utility_tensor - torch.max(utility_tensor)))
        kernel = weights[:, None] * matrix * weights[None, :]
    elif method == "additive":
        kernel = torch.diag(utility_tensor + float(epsilon)) + float(interaction) * matrix
    else:
        raise ValueError(f"Unknown DPP kernel method: {method}")
    kernel = (kernel + kernel.T) * 0.5

    residual = torch.diagonal(kernel).clone()
    factors = torch.zeros((selection_k, candidate_count), dtype=matrix.dtype, device=matrix.device)
    selected_mask = torch.zeros(candidate_count, dtype=torch.bool, device=matrix.device)
    order_tensor = (
        None
        if tie_order is None or tie_tolerance is None
        else torch.as_tensor(np.asarray(tie_order), dtype=torch.float64, device=matrix.device)
    )
    selected: list[int] = []
    gains: list[float] = []
    for step in range(selection_k):
        scores = residual.masked_fill(selected_mask, -torch.inf)
        winner = int(torch.argmax(scores).item())
        if order_tensor is not None:
            best = scores[winner]
            tied = scores >= best - torch.abs(best) * float(tie_tolerance)
            if int(tied.sum().item()) > 1:
                winner = int(torch.argmin(order_tensor.masked_fill(~tied, torch.inf)).item())
        pivot = torch.sqrt(torch.clamp(residual[winner], min=float(epsilon)))
        row = kernel[winner, :].clone()
        if step:
            row -= torch.mv(factors[:step, :].T, factors[:step, winner])
        row /= pivot
        factors[step, :] = row
        residual -= row.square()
        residual[winner] = -torch.inf
        selected_mask[winner] = True
        selected.append(winner)
        gains.append(float(torch.log(torch.clamp(pivot.square(), min=float(epsilon))).item()))

    positions = torch.as_tensor(selected, dtype=torch.long, device=matrix.device)
    final = kernel.index_select(0, positions).index_select(1, positions)
    sign, logdet_tensor = torch.linalg.slogdet((final + final.T) * 0.5 + float(epsilon) * torch.eye(selection_k, device=matrix.device, dtype=matrix.dtype))
    if float(sign.item()) <= 0 or not math.isfinite(float(logdet_tensor.item())):
        raise ValueError("DPP kernel produced a non-positive final determinant")
    return selected, gains, float(logdet_tensor.item())
