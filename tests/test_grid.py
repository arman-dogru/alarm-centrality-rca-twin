"""rcatwin/grid.py design invariants (review r2 F3): eval twins never overlap dev twins, and the p_link sweep is a
real sweep."""
from rcatwin import grid


def test_eval_and_dev_seeds_are_disjoint():
    assert set(grid.EVAL_SEEDS).isdisjoint(grid.DEV_SEEDS)
    assert len(grid.EVAL_SEEDS) == 30 and set(grid.EVAL10) <= set(grid.EVAL_SEEDS)
    P = grid.phases()
    dev = {s for _, _, s, *_ in P["dev"]}
    ev = {s for ph in ("main", "plink", "poll", "ablate", "size") for _, _, s, *_ in P[ph]}
    assert dev == set(grid.DEV_SEEDS) and dev.isdisjoint(ev)


def test_plink_sweep_is_not_collapsed():
    assert grid.P_LINK == (0.3, 0.5, 0.7, 0.9)
    conds = grid.phases()["plink"][0][3]
    assert len({grid.cond_id(c) for c in conds}) == len(conds) == 10
    assert {c["p_link"] for c in conds} == set(grid.P_LINK)
    assert sum(c.get("link_model") == "upstream" for c in conds) == 2
