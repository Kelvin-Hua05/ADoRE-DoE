import ast
import pathlib

import numpy as np
import pytest

import adore_doe as ad
from adore_doe import active_learning as al
from adore_doe import design
from adore_doe.active_learning import ActiveLearner, EvaluationSet, MultiResponseStopping

FS = [ad.Factor("A", 0, 10), ad.Factor("B", 0, 10)]
KW = dict(n_restarts=1, n_candidates=150, n_integration=128, cv={"n_splits": 3})


def sim_hard(P):
    return {"y1": np.sin(2 * P[:, 0]) + P[:, 1] ** 2, "y2": 1000 * (P[:, 0] * P[:, 1]) + 50}


def sim_easy(P):
    return {"y1": 2 * P[:, 0] - P[:, 1] + 3, "y2": 500 * P[:, 0] + 7 * P[:, 1]}


def make(sim, n0=8, **stop):
    ds = design.i_optimal(FS, "quadratic", n0, seed=1)
    v = sim(ds.coded)
    for r in ("y1", "y2"):
        ds.add_response(r, v[r])
    return ds


def learner(sim=sim_hard, n0=8, seed=3, strategy="ivr", stop=None, **kw):
    ds = make(sim, n0)
    st = stop or MultiResponseStopping(n_max=14, response_min_r2=0.9, consecutive=2, n_min=1)
    return ActiveLearner(ds, ["y1", "y2"], st, strategy=strategy, seed=seed, **{**KW, **kw})


# ------------------------------------------------------------------ the two ways to stop
def test_budget_exhausted_is_reported_honestly_and_budget_is_exact():
    L = learner(stop=MultiResponseStopping(n_max=12, response_min_r2=0.9999, consecutive=2, n_min=1))
    d = L.run(sim_hard)
    s = L.summary()
    assert d.reason == "budget_exhausted" and d.target_met is False and s["target_met"] is False
    assert L.n_runs == 12 == s["n_runs"] and s["stopped"] and "未達成" in s["message"]
    assert L.history[-1]["decision"].reason == "budget_exhausted" and len(L.history) == s["rounds"]


def test_target_reached_stops_before_the_budget():
    L = learner(sim_easy, stop=MultiResponseStopping(n_max=30, response_min_r2=0.5, consecutive=2, n_min=1))
    d = L.run(sim_easy)
    assert d.reason == "target_reached" and d.target_met and L.n_runs < 30
    assert d.streak >= 2 and L.history[-1]["round_met"]
    assert L.summary()["target_met"] is True


def test_last_batch_is_trimmed_so_that_n_max_is_never_exceeded():
    st = MultiResponseStopping(n_max=13, response_min_r2=0.9999, consecutive=2, n_min=1, batch_size=3)
    L = learner(stop=st)
    L.run(sim_hard)
    assert L.n_runs == 13
    sizes = np.diff([h["n_runs"] for h in L.history])
    assert list(sizes) == [3, 2] and max(sizes) <= 3
    with pytest.raises(ad.DesignError, match="exceed n_max"):
        pts = np.array([[0.1, 0.1], [0.2, 0.2]])
        L.update(pts, sim_hard(pts))


# ------------------------------------------------------------------ protocol
def test_protocol_order_and_no_double_counting_of_rounds():
    L = learner()
    with pytest.raises(ad.DesignError, match="assess"):
        L.recommend()
    d1 = L.assess()
    d2 = L.assess()                                       # unchanged data: same round, same decision
    assert d1 is d2 and L.round == 1 and len(L.history) == 1 and len(L.checks) == 1
    sel = L.recommend()
    assert L.pending is sel and len(sel) == 1
    L.update(sel.points, sim_hard(sel.points))
    assert L.pending is None and L.n_runs == 9
    with pytest.raises(ad.DesignError, match="assess"):
        L.recommend()                                     # new data -> must assess again
    L.assess()
    assert L.round == 2


def test_recommend_never_simulates_or_changes_the_dataset():
    L = learner()
    L.assess()
    n0, ids0 = L.ds.n_runs, L.ds.coded.copy()
    sel = L.recommend()
    assert L.ds.n_runs == n0 and np.array_equal(L.ds.coded, ids0)
    assert L.region.contains(sel.points).all()
    assert not any(np.allclose(sel.points[0], r) for r in ids0)           # completed runs are excluded
    ids = L.update(sel.points, sim_hard(sel.points))
    assert list(ids) == [n0 + 1] and L.ds.source[-1] == "AL-iter-1" and L.ds.point_type[-1] == "active_learning"
    assert L.ds.stage[-1] > 0 and not np.isnan(L.ds.get_response("y1")).any()


def test_recommend_after_a_stop_is_refused():
    L = learner(stop=MultiResponseStopping(n_max=8, consecutive=2, n_min=1))      # already at n_max
    d = L.assess()
    assert d.stop and d.reason == "budget_exhausted"
    with pytest.raises(ad.DesignError, match="budget_exhausted"):
        L.recommend()


def test_update_validation():
    L = learner()
    L.assess()
    P = np.array([[0.1, 0.2]])
    v = sim_hard(P)
    for pts, vals in (([[0.1, 0.2, 0.3]], v), ([[2.0, 0.0]], v), (P, {"y1": v["y1"]}),
                      (P, {"y1": v["y1"], "y2": np.array([np.nan])}), (P, {"y1": v["y1"], "y2": [1.0, 2.0]})):
        with pytest.raises(ad.DesignError):
            L.update(pts, vals)
    assert L.ds.n_runs == 8                                   # nothing was added by the failed calls


def test_incomplete_or_too_few_runs_are_reported():
    ds = make(sim_hard)
    ds.append_runs(np.array([[0.0, 0.0]]), source="manual")           # responses NaN
    L = ActiveLearner(ds, ["y1", "y2"], MultiResponseStopping(n_max=20), seed=1, **KW)
    with pytest.raises(ad.DesignError, match="missing responses"):
        L.assess()
    tiny = design.ccd([ad.Factor("A", 0, 1), ad.Factor("B", 0, 1)]).subset(np.arange(9) < 2)
    tiny.add_response("y1", [1.0, 2.0]); tiny.add_response("y2", [3.0, 5.0])
    with pytest.raises(ad.DesignError, match="at least 3"):
        ActiveLearner(tiny, ["y1", "y2"], MultiResponseStopping(n_max=20), seed=1, **KW).assess()


# ------------------------------------------------------------------ reproducibility and options
def test_same_seed_same_recommendations_different_seed_differs():
    def rec(seed):
        L = learner(seed=seed, stop=MultiResponseStopping(n_max=11, response_min_r2=0.9999, consecutive=2, n_min=1))
        L.run(sim_hard)
        return L.ds.coded[8:]
    a, b, c = rec(5), rec(5), rec(6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("strategy", ["ivr", "max_variance", "i_optimal", "random"])
def test_every_strategy_runs_a_full_loop(strategy):
    L = learner(strategy=strategy, stop=MultiResponseStopping(n_max=10, response_min_r2=0.9999, consecutive=2, n_min=1))
    d = L.run(sim_hard)
    assert d.stop and L.n_runs == 10 and all(h["strategy"] == strategy for h in L.history)


def test_constraints_and_min_distance_hold_in_every_round():
    cons = [({"A": 1, "B": 1}, "<=", 12)]
    ds = design.i_optimal(FS, "quadratic", 8, constraints=cons, seed=1)
    v = sim_hard(ds.coded)
    ds.add_response("y1", v["y1"]); ds.add_response("y2", v["y2"])
    L = ActiveLearner(ds, ["y1", "y2"], MultiResponseStopping(n_max=12, response_min_r2=0.9999, consecutive=2, n_min=1),
                      constraints=cons, min_distance=0.25, seed=2, **KW)
    L.run(sim_hard)
    new = L.ds.coded[8:]
    assert L.region.contains(new).all() and (L.ds.actual_df().iloc[8:].sum(axis=1) <= 12 + 1e-9).all()
    from adore_doe.active_learning import distances_to
    for i in range(len(new)):
        assert distances_to(new[i][None, :], L.ds.coded[:8 + i]).min() >= 0.25 - 1e-12


def test_hyperparameter_reestimation_option():
    stop = MultiResponseStopping(n_max=11, response_min_r2=0.9999, consecutive=2, n_min=1)
    fixed = learner(reestimate_hyperparameters=False, stop=stop); fixed.run(sim_hard)
    th = [h["hyperparameters"]["y1"]["theta"] for h in fixed.history]
    assert all(np.allclose(th[0], t) for t in th)                           # frozen after the first fit
    free = learner(reestimate_hyperparameters=True, stop=stop); free.run(sim_hard)
    th2 = [h["hyperparameters"]["y1"]["theta"] for h in free.history]
    assert not all(np.allclose(th2[0], t) for t in th2)


def test_responses_in_different_units_do_not_change_the_r2():
    a = learner(); a.assess()
    ds = make(sim_hard)
    ds.add_response("y2", ds.get_response("y2") * 1e6 + 5e6)
    b = ActiveLearner(ds, ["y1", "y2"], MultiResponseStopping(n_max=14, response_min_r2=0.9, consecutive=2, n_min=1),
                      seed=3, **KW)
    b.assess()
    for r in ("y1", "y2"):
        assert a.history[0]["r2"][r].value == pytest.approx(b.history[0]["r2"][r].value, abs=5e-3)


def test_validation_runs_are_not_training_runs_or_budget():
    L = learner()
    P = np.array([[3.0, 4.0], [6.0, 2.0]])
    v = sim_hard(P)
    L.ds.append_runs(P, source="validation", values=v)
    d = L.assess()
    assert L.n_runs == 8 and d.n_runs == 8
    assert L.ds.n_runs == 10 and L.history[0]["n_runs"] == 8


# ------------------------------------------------------------------ evaluation set integration
def test_evaluation_metrics_are_recorded_and_independent_of_training():
    es = EvaluationSet.build(FS, 200, seed=7).with_truth(lambda P: sim_hard(P))
    L = learner(eval_set=es, stop=MultiResponseStopping(n_max=11, response_min_r2=0.9999, consecutive=2, n_min=1))
    L.run(sim_hard)
    df = L.history_df()
    assert {"eval_rmse_y1", "eval_r2_y2", "r2_y1", "r2_status_y2", "aggregate_r2"} <= set(df.columns)
    assert df["eval_rmse_y1"].notna().all() and df["n_runs"].is_monotonic_increasing
    ds = make(sim_hard)
    overlap = EvaluationSet(np.vstack([ds.coded[:2], np.array([[0.3, -0.2]])]), 0)
    bad = ActiveLearner(ds, ["y1", "y2"], MultiResponseStopping(n_max=20), eval_set=overlap, seed=1, **KW)
    with pytest.raises(ad.DesignError, match="coincide"):
        bad.assess()


def test_history_table_has_counts_and_decision():
    L = learner(stop=MultiResponseStopping(n_max=10, response_min_r2=0.9999, consecutive=2, n_min=1))
    L.run(sim_hard)
    df = L.history_df()
    assert list(df["n_runs"]) == [8, 9, 10] and df["decision"].iloc[-1] == "budget_exhausted"
    assert (df["n_unique"] == df["n_runs"]).all() and (df["n_replicated"] == 0).all()
    assert df["r2_y1"].dtype.kind == "f"


def test_constructor_validation():
    ds = make(sim_hard)
    st = MultiResponseStopping(n_max=20)
    for kw in (dict(strategy="ei"), dict(responses=[]), dict(responses=["y1", "y1"]), dict(responses=["zzz"])):
        args = dict(responses=["y1", "y2"], strategy="ivr"); args.update(kw)
        with pytest.raises(ad.DesignError):
            ActiveLearner(ds, args.pop("responses"), st, **args)
    with pytest.raises(ad.DesignError):
        ActiveLearner(ds, ["y1"], "not a stopping rule")


# ------------------------------------------------------------------ ADR-005: independence
def test_active_learning_does_not_import_rsm_or_residual():
    root = pathlib.Path(al.__file__).parent
    bad = []
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [("." * node.level) + (node.module or "")] + [("." * node.level) + (node.module or "") + "." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                tail = m.replace("adore_doe", "").lstrip(".")
                if tail.split(".")[0] in ("rsm", "residual"):
                    bad.append((f.name, m))
    assert bad == []


def test_femto_scale_responses_work_end_to_end():
    """Regression: a response of order 1e-15 (e.g. capacitance in farad) must not be treated as constant."""
    def sim(P):
        return {"y1": 2e-15 * (1 + P[:, 0] - 0.5 * P[:, 1] + 0.4 * P[:, 0] * P[:, 1]), "y2": 100 + 30 * np.sin(2 * P[:, 0])}
    ds = design.i_optimal(FS, "quadratic", 8, seed=1)
    v = sim(ds.coded)
    for r in ("y1", "y2"):
        ds.add_response(r, v[r])
    L = ActiveLearner(ds, ["y1", "y2"], MultiResponseStopping(n_max=10, response_min_r2=0.9999, consecutive=2, n_min=1),
                      seed=3, **KW)
    L.run(sim)
    assert L.n_runs == 10 and all(h["r2"]["y1"].status == "ok" for h in L.history)
