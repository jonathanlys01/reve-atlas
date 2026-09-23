import numpy as np
import pytest

from scripts.build_selections import allocate
from scripts.real_atlas import cosine_gram, greedy_map_lazy


def _problem(seed: int, count: int = 300, dim: int = 24):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(count, dim)).astype(np.float32)
    losses = rng.gamma(2.0, size=count)
    utility = (losses - losses.min()) / (losses.max() - losses.min())
    tie_order = rng.permutation(count)
    return cosine_gram(embeddings), utility, tie_order


def test_flat_allocation_water_fills_to_target():
    sizes = {0: 5, 1: 50, 2: 500, 3: 5000}
    counts = allocate(sizes, 0.10, "flat")
    assert sum(counts.values()) == round(0.10 * sum(sizes.values()))
    assert counts[0] == 5 and counts[1] == 50
    assert abs(counts[2] - counts[3]) <= 1
    assert all(1 <= counts[group] <= sizes[group] for group in sizes)


def test_proportional_allocation_matches_published_rule():
    sizes = {0: 389, 1: 1636, 2: 12940}
    assert allocate(sizes, 0.01, "proportional") == {0: 4, 1: 17, 2: 130}


def test_tie_break_prefers_smallest_order_among_equal_residuals():
    # Four identical rows: every residual is tied, so the path is fixed by tie_order alone.
    similarity = cosine_gram(np.ones((4, 3), dtype=np.float32))
    utility = np.ones(4)
    selected, _, _ = greedy_map_lazy(
        similarity, utility, "multiplicative", 0.0, 2, 1e-6, tie_order=np.array([3, 1, 0, 2]), tie_tolerance=1e-9
    )
    assert selected[0] == 2


@pytest.mark.parametrize("method,interaction", [("multiplicative", 0.0), ("multiplicative", 8.0), ("additive", 0.0), ("additive", 1.0)])
def test_numpy_and_torch_agree_with_tie_break(method, interaction):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.torch_dpp import greedy_map as torch_greedy_map

    for seed in range(5):
        similarity, utility, tie_order = _problem(seed)
        numpy_path, numpy_gains, _ = greedy_map_lazy(
            similarity, utility, method, interaction, 40, 1e-6, tie_order=tie_order, tie_tolerance=1e-9
        )
        torch_path, torch_gains, _ = torch_greedy_map(
            torch.as_tensor(similarity, dtype=torch.float64, device="cuda"),
            utility, method, interaction, 40, 1e-6, tie_order=tie_order, tie_tolerance=1e-9,
        )
        assert numpy_path == torch_path
        np.testing.assert_allclose(numpy_gains, torch_gains, atol=1e-8)


def test_recursive_dbscan_accepts_separated_blobs_and_attaches_noise():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuml")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import pyarrow as pa

    from scripts.hier_groups import Grouper, Params, check, groups_table

    rng = np.random.default_rng(0)
    centers = np.linalg.qr(rng.normal(size=(64, 3)))[0].T  # orthonormal: blobs ~1.41 apart
    blobs = [center + 0.02 * rng.normal(size=(1500, 64)) for center in centers]
    noise = rng.normal(size=(40, 64))
    points = np.concatenate([*blobs, noise]).astype(np.float32)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    params = Params(n_max=2000, n_min=500, eps0=0.4, min_samples=5, gamma=0.85)
    grouper = Grouper(torch.as_tensor(points, device="cuda"), params, None, log=lambda _: None)
    grouper.run()
    window_ids = np.asarray([f"w{i:05d}" for i in range(len(points))], dtype=object)
    table = groups_table(grouper, window_ids)
    check(table, params, len(points))
    assert len(grouper.leaves) == 3
    assert {leaf.split for leaf in grouper.leaves} == {"dbscan"}
    assert int(grouper.attached.sum()) == 40 and grouper.attached[-40:].all()
    group_of = table["group_id"].to_numpy()
    for blob in range(3):
        assert len(set(group_of[blob * 1500 : (blob + 1) * 1500].tolist())) == 1
    assert isinstance(table, pa.Table)
