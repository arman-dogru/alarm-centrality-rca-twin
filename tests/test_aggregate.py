"""scripts/aggregate.py on toy data frames: dev-only calibration, the triage rule, derived methods (oracle, blog
mapping), bootstrap metrics with unequal twins and the significance rule."""
import importlib.util, os

import numpy as np
import pandas as pd
import pytest

from rcatwin.triage import LABELS, SELECT

_spec = importlib.util.spec_from_file_location("aggregate", os.path.join(os.path.dirname(__file__), "..", "scripts", "aggregate.py"))
agg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(agg)

BASE = ["ppr", "alarm_degree", "alarm_closeness", "equal", "w_hierarchical", "w_star", "w_mesh", "alarm_degree_tb",
        "static_pagerank", "static_degree", "static_closeness", "earliest_alarm", "earliest_in_cascade", "global_pagerank",
        "random_candidate"] + [f"common_ancestor_{f:g}" for f in (0.3, 0.5, 0.7, 0.9)]


def toy(seeds, n_per=30, rng=None):
    """Per incident: true label from the features (cycle-rich -> mesh, uneven degrees -> star, else hierarchical);
    in mesh rows alarm-degree is right, in star rows PPR, in hierarchical rows alarm-closeness."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for s in seeds:
        for i in range(n_per):
            lab = LABELS[i % 3]
            cyc = {"mesh": 0.8, "star": 0.0, "hierarchical": 0.05}[lab] + rng.uniform(0, 0.05)
            cv = {"mesh": 0.3, "star": 3.5, "hierarchical": 0.8}[lab] + rng.uniform(0, 0.1)
            r = {f"rank:{m}": 5 for m in BASE}
            r.update({"rank:common_ancestor_0.7": 1 if i % 2 else 3})  # 0.7 is the dev-best coverage fraction
            win = {"mesh": "alarm_degree", "star": "ppr", "hierarchical": "alarm_closeness"}[lab]
            r[f"rank:{win}"] = 1
            r["rank:alarm_degree"] = 1 if lab == "mesh" else 4
            r.update({k.replace("rank:", "rankopt:"): v for k, v in list(r.items())})
            f = {f"{sc}:{k}": v for sc in ("feat", "featctx") for k, v in
                 dict(cycle_rank=cyc, deg_cv=cv, hub_share=0.1, depth=2, diameter=4, clustering=0.1, n=50).items()}
            rows.append(dict(seed=s, kind="mesh", incident=i, no_alarms=0, true_label=lab, **r, **f))
    return pd.DataFrame(rows)


def test_calibration_refuses_eval_twins_and_uses_dev():
    with pytest.raises(ValueError):
        agg.calibrate(toy([1, 100]))
    cal, tree = agg.calibrate(toy([1, 2]))
    assert cal["dev_seeds"] == [1, 2] and cal["n_dev_incidents"] == 60
    assert cal["common_ancestor_frac"]["chosen"] == "0.7"
    assert cal["rule"]["dev_balanced_acc"] == 1.0
    assert cal["dev_best_member"]["mesh"]["chosen"] == "alarm_degree"
    assert cal["dev_best_member"]["star"]["chosen"] == "ppr"


def test_main_reads_only_dev_for_calibration(tmp_path, monkeypatch):
    for ph, seeds in (("dev", [1]), ("main", [100])):
        os.makedirs(tmp_path / ph)
        toy(seeds).to_csv(tmp_path / ph / "x.csv.gz", index=False, compression="gzip")
    seen = []
    real = agg.calibrate
    monkeypatch.setattr(agg, "calibrate", lambda dev: (seen.append(sorted(dev.seed.unique())), real(dev))[1])
    monkeypatch.setattr(agg, "ROOT", str(tmp_path)); os.makedirs(tmp_path / "results")
    monkeypatch.setattr(agg, "groups_for", lambda phase, df: [("all", df)])
    monkeypatch.setattr(agg, "apply_holm", lambda comps: None)
    monkeypatch.setattr(agg, "METHODS", ["ppr", "alarm_degree", "equal"])
    monkeypatch.setattr(agg, "B", 20)
    agg.main(str(tmp_path), allow_partial=True)  # partial on purpose: only the calibration path is under test
    assert seen == [[1]]


def test_triage_rule_labels():
    df = toy([1], 9)
    lab = agg.rule(df, "feat", 0.4, 2.6)
    assert (lab == df.true_label.to_numpy()).all()
    assert list(agg.rule(pd.DataFrame({"feat:cycle_rank": [0.5, 0.1, 0.1], "feat:deg_cv": [3.0, 3.0, 1.0]}), "feat", 0.4, 2.6)) \
        == ["mesh", "star", "hierarchical"]


def test_derive_oracle_and_blog_mapping():
    dev = toy([1, 2]); cal, tree = agg.calibrate(dev)
    df = toy([100])
    df.loc[0, ["rank:ppr", "rank:alarm_degree", "rank:alarm_closeness"]] = [7, 2, 9]
    df.loc[0, ["rankopt:ppr", "rankopt:alarm_degree", "rankopt:alarm_closeness"]] = [7, 1, 9]
    out = agg.derive(df, cal, tree)
    assert out.at[0, "rank:oracle_incident"] == 2 and out.at[0, "rankopt:oracle_incident"] == 1  # best member, its own opt rank
    assert (out["rank:oracle_incident"] == out[["rank:ppr", "rank:alarm_degree", "rank:alarm_closeness"]].min(1)).all()
    for i, lab in enumerate(out.triage_rule):
        assert out.at[i, "rank:triage_blog_select"] == out.at[i, f"rank:{SELECT[lab]}"]
        assert out.at[i, "rank:triage_blog_weighted"] == out.at[i, f"rank:w_{lab}"]
        assert out.at[i, "rank:triage_dev_select"] == out.at[i, f"rank:{cal['dev_best_member'][lab]['chosen']}"]
    assert (out["rank:common_ancestor"] == out["rank:common_ancestor_0.7"]).all()


def test_metrics_unequal_twins_ci_and_twin_level_p():
    rng = np.random.default_rng(0)
    rows = []
    for s, n in zip(range(8), (5, 50, 10, 30, 7, 40, 20, 12)):  # unequal twin sizes
        for i in range(n):
            a, b = 1 if (i % 4) else 3, 2
            rows.append({"kind": "mesh", "seed": s, "rank:a": a, "rankopt:a": a, "rank:b": b, "rankopt:b": b})
    df = pd.DataFrame(rows)
    out = agg.metrics([("g", df)], ["a", "b"], rng, B=400, comparisons=[("a", "b")])["g"]
    m = out["methods"]["a"]["top1"]
    assert out["twins"] == 8 and abs(m["mean"] - (df["rank:a"] == 1).mean()) < 1e-12
    assert m["lo"] <= m["mean"] <= m["hi"] and m["hi"] - m["lo"] < 0.2
    c = out["comparisons"][0]
    assert c["twins_pos"] == 8 and c["twins_neg"] == 0
    assert c["p"] == pytest.approx(2 / 2 ** 8)  # exact twin-level sign-flip: all 8 twins positive
    assert c["p_incident"] < c["p"]  # the incident-level test is (wrongly) far more confident


def test_significance_needs_holm_and_ci():
    comps = [dict(p=1e-4, lo=0.01, hi=0.1), dict(p=1e-4, lo=-0.01, hi=0.1), dict(p=0.03, lo=0.01, hi=0.1),
             dict(p=0.03, lo=-0.2, hi=-0.01)]
    agg.apply_holm(comps)
    assert [c["significant"] for c in comps] == [True, False, False, False]  # raw p 0.03 < 0.05, but Holm 0.06
    assert comps[2]["p_holm"] == pytest.approx(0.06) and comps[0]["p_holm"] == pytest.approx(4e-4)


def test_metrics_bootstrap_is_twin_clustered():
    # 10 twins: 5 always right, 5 always wrong -> mean 0.5 with a wide twin-level CI (an incident bootstrap would be +-0.03)
    rows = [{"kind": "mesh", "seed": s, "rank:a": 1 if s < 5 else 2, "rankopt:a": 1 if s < 5 else 2} for s in range(10) for _ in range(100)]
    m = agg.metrics([("g", pd.DataFrame(rows))], ["a"], np.random.default_rng(0), B=400, comparisons=[])["g"]["methods"]["a"]["top1"]
    assert m["hi"] - m["lo"] > 0.4


# ---- review round 2 (F2, F3): grid guard, twin ids, stratification, twin counts, Holm families -------------------
def _two_topology_frame():
    """Seeds 100/101 exist in BOTH topologies (so a seed-only twin id would merge them); unequal sizes across
    topologies; within a topology every twin has the same size and the same outcome, so a topology-stratified
    bootstrap has zero variance while a pooled one does not."""
    rows = []
    for kind, n, a in (("mesh", 10, 1), ("star", 30, 2)):
        for s in (100, 101):
            for i in range(n):
                b = 1 if (kind == "mesh" and s == 100) else a  # mesh twin 100: a == b -> zero difference
                rows.append({"kind": kind, "seed": s, "rank:a": a, "rankopt:a": a, "rank:b": b, "rankopt:b": b})
    return pd.DataFrame(rows)


def test_metrics_twin_id_is_kind_and_seed_and_pooled_bootstrap_is_stratified():
    out = agg.metrics([("all", _two_topology_frame())], ["a", "b"], np.random.default_rng(0), B=300,
                      comparisons=[("b", "a")])["all"]
    assert out["twins"] == 4  # (mesh,100), (mesh,101), (star,100), (star,101)
    m = out["methods"]["a"]["top1"]
    assert m["mean"] == pytest.approx(20 / 80)
    assert m["lo"] == pytest.approx(m["mean"]) and m["hi"] == pytest.approx(m["mean"])  # stratified: no spread
    c = out["comparisons"][0]  # b == a in every row (the frame encodes b = 1 for mesh twin 100, which equals a there)
    assert c["twins_pos"] + c["twins_neg"] < out["twins"]  # zero-difference twins are counted on neither side
    assert c["twins_pos"] == 0 and c["twins_neg"] == 0


def test_twins_pos_excludes_zero_twins():
    df = _two_topology_frame()
    df.loc[(df.kind == "star"), "rank:b"] = 1  # star twins: b right, a wrong -> b - a = +1
    df["rankopt:b"] = df["rank:b"]
    c = agg.metrics([("all", df)], ["a", "b"], np.random.default_rng(0), B=50, comparisons=[("b", "a")])["all"]["comparisons"][0]
    assert (c["twins_pos"], c["twins_neg"]) == (2, 0)  # mesh twins: a == b == 1 -> zero, not positive


def _comp(a, b, p, planned=True):
    return {"a": a, "b": b, "p": p, "lo": 0.01, "hi": 0.1, "planned": planned}


def test_holm_families_exclude_exploratory_and_have_ten_members():
    M = {k: {"comparisons": [_comp(a, b, 0.004) for a, b in agg.COMPARISONS] + [_comp(*agg.EXPLORATORY[0], 1e-6, False)]}
         for k in ("all", "hierarchical", "star", "mesh", "hybrid")}
    agg.holm_families(M)
    for k in M:
        planned = [c for c in M[k]["comparisons"] if c["planned"]]
        assert len(planned) == 10 and all("p_holm" in c for c in planned)
        assert all("p_holm" not in c for c in M[k]["comparisons"] if not c["planned"])
    assert M["all"]["comparisons"][0]["p_holm"] == pytest.approx(0.04)    # pooled family: 10 x 0.004
    assert M["mesh"]["comparisons"][0]["p_holm"] == pytest.approx(0.16)   # topology family: 40 x 0.004


def test_aggregate_refuses_an_incomplete_grid(tmp_path, monkeypatch, capsys):
    os.makedirs(tmp_path / "dev"); os.makedirs(tmp_path / "main")
    some = sorted(agg.expected_files()["main"])[0]
    toy([100]).to_csv(tmp_path / "main" / some, index=False, compression="gzip")
    monkeypatch.setattr(agg, "ROOT", str(tmp_path)); os.makedirs(tmp_path / "results")
    with pytest.raises(SystemExit) as e:
        agg.main(str(tmp_path))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "incomplete" in err and "main: 1/1440" in err and "dev: 0/240" in err
    assert not os.path.exists(tmp_path / "results" / "summary.json")  # nothing overwritten


def test_check_grid_complete_missing_and_unexpected(tmp_path):
    exp = agg.expected_files()
    assert {k: len(v) for k, v in exp.items()} == {"dev": 240, "main": 1440, "plink": 1200, "poll": 360, "ablate": 40, "size": 60}
    for ph, names in exp.items():
        os.makedirs(tmp_path / ph)
        for f in names:
            (tmp_path / ph / f).touch()
    assert agg.check_grid(str(tmp_path)) == {}
    gone = sorted(exp["plink"])[3]
    os.remove(tmp_path / "plink" / gone)
    (tmp_path / "size" / "mesh_n9_s1_bogus.csv.gz").touch()
    prob = agg.check_grid(str(tmp_path))
    assert prob == {"plink": ([gone], []), "size": ([], ["mesh_n9_s1_bogus.csv.gz"])}
