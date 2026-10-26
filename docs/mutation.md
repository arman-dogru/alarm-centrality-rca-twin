# Mutation testing

[`scripts/mutation_check.py`](../scripts/mutation_check.py) applies each single-line mutant to a temporary copy of
`rcatwin/`, `scripts/` and `tests/`, then runs `pytest -x`. Every committed mutant must make at least one test fail.
The result is in [`results/mutation_check.json`](../results/mutation_check.json).

The reviewer's independent round-2 set was also run
against this tree. The round-2 result is in [`results/mutation_reviewer_r2.json`](../results/mutation_reviewer_r2.json).

## True equivalents (not in the committed suite, because no test can kill them)

These mutants change no observable output of the functions. Their only effect is on running time.

| mutant | why it is equivalent |
|---|---|
| `rca.neighbourhood`: the BFS frontier is not pruned by `- nodes` | `nodes` is a set and the frontier is unioned into it, so re-adding already-visited nodes changes nothing. The loop still runs exactly `hops` times. The result is identical; only redundant work is added. |
| `rca.global_pagerank`: `cache.clear()` at function entry (cache disabled) | The cached value is the same deterministic PageRank vector that a fresh computation returns, so disabling the cache changes only speed. The cache *key* is a separate matter: it is pinned by `test_global_pagerank_cache_keyed_by_twin_id` and by the committed mutant "global PR cache keyed by id(G)". |
| `rca.global_pagerank`: `if key is None or key not in cache:` becomes `if key not in cache:` (reviewer r2) | When `key is None`, the next statement `if key is None: return val` returns the fresh vector before anything is stored. `None` therefore never enters the cache, and `None not in cache` is always true, so both conditions take the same branch for every input. |

## Not equivalent after all

- **`rank_of` rewritten as `(s >= r).sum() + (1 if optimistic else 0)`.** In round 1 I called this equivalent. It
  is equivalent for the pessimistic rank, but it breaks the optimistic rank, which becomes ties + 1 instead of
  better + 1. `test_rank_optimistic_and_expected_top1` kills it, and it is now in the committed suite as
  "rank rewrite (not equivalent: breaks optimistic)".

## Round-2 additions to the committed suite

These are the reviewer's surviving round-2 mutants that are not equivalent, plus two for the new grid guard. Each
is pinned by a test:

| mutant | killed by |
|---|---|
| `twin_means` returns sums | `test_twin_means_are_means_with_unequal_twins` |
| pooled bootstrap not stratified by topology | `test_metrics_twin_id_is_kind_and_seed_and_pooled_bootstrap_is_stratified` |
| twin id = seed only (merges topologies) | same test (twin count 4, not 2) |
| `twins_pos` counts zero-difference twins | `test_twins_pos_excludes_zero_twins` and the stratified test |
| pooled Holm family includes the exploratory comparison | `test_holm_families_exclude_exploratory_and_have_ten_members` |
| gate agreement counts a member's tied top | `test_confidence_member_tie_that_includes_the_ensemble_top_does_not_count` |
| eval seeds overlap dev seeds | `test_eval_and_dev_seeds_are_disjoint` |
| p_link sweep collapsed to one value | `test_plink_sweep_is_not_collapsed` |
| aggregate accepts an incomplete grid | `test_aggregate_refuses_an_incomplete_grid` |
| grid check only notices empty phases | `test_check_grid_complete_missing_and_unexpected` |
