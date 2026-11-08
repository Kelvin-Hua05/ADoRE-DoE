import numpy as np
import pytest

import adore_doe as ad
from adore_doe.active_learning.cv import R2Result, predictive_r2
from adore_doe.active_learning.stopping import (MultiResponseStopping, check_round, validate_weights)

N3 = ["R1", "R2", "R3"]


def r2(*v):
    return dict(zip(N3, v))


# ------------------------------------------------------------------ one round
def test_spec_example_third_response_below_threshold_blocks_stopping():
    c = check_round(r2(0.96, 0.95, 0.89), response_min_r2=0.90, aggregate_r2=0.95)
    assert not c.met and c.per_response["R3"]["met"] is False
    assert c.aggregate == pytest.approx(0.9333333333) and c.aggregate_met is False
    assert any("R3" in u for u in c.unmet)
    # the mean alone would also be below 0.95 here; make the mean pass and check the individual rule alone:
    c2 = check_round(r2(0.99, 0.99, 0.89), response_min_r2=0.90, aggregate_r2=0.90)
    assert c2.aggregate_met is True and c2.per_response["R3"]["met"] is False and not c2.met


def test_spec_example_mean_exactly_at_threshold_meets_the_target():
    c = check_round(r2(0.96, 0.95, 0.94), response_min_r2=0.90, aggregate_r2=0.95)
    assert c.aggregate == pytest.approx(0.95) and c.aggregate_met and c.met and c.unmet == []


def test_aggregate_never_replaces_individual_thresholds_and_vice_versa():
    assert not check_round(r2(0.99, 0.99, 0.80), response_min_r2=0.90, aggregate_r2=0.90).met   # mean .927 ok, one fails
    c = check_round(r2(0.91, 0.91, 0.91), response_min_r2=0.90, aggregate_r2=0.95)
    assert all(p["met"] for p in c.per_response.values()) and c.aggregate_met is False and not c.met
    assert check_round(r2(0.91, 0.92, 0.93), response_min_r2=0.90).met                          # no aggregate required
    assert check_round(r2(0.95, 0.95, 0.95)).met and not check_round(r2(0.95, 0.95, 0.9499)).met   # default tau = .95


def test_per_response_thresholds_and_weighted_mean():
    c = check_round(r2(0.92, 0.96, 0.88), response_min_r2={"R1": 0.90, "R2": 0.95, "R3": 0.85})
    assert c.met and c.per_response["R3"]["threshold"] == 0.85
    w = {"R1": 0.5, "R2": 0.3, "R3": 0.2}
    cw = check_round(r2(0.96, 0.95, 0.89), response_min_r2=0.88, aggregate_r2=0.94,
                     aggregate_method="weighted_mean", weights=w)
    assert cw.aggregate == pytest.approx(0.5 * 0.96 + 0.3 * 0.95 + 0.2 * 0.89) and cw.aggregate_met
    assert not check_round(r2(0.96, 0.95, 0.89), response_min_r2=0.88, aggregate_r2=0.95,
                           aggregate_method="weighted_mean", weights=w).aggregate_met


def test_weight_validation():
    names = N3
    assert validate_weights({"R1": 0.5, "R2": 0.3, "R3": 0.2}, names)["R1"] == 0.5
    for bad in ({"R1": 0.5, "R2": 0.3, "R3": 0.3}, {"R1": 1.2, "R2": -0.1, "R3": -0.1},
                {"R1": 0.5, "R2": 0.5}, {"R1": 0.5, "R2": 0.3, "R3": 0.2, "R4": 0.0}, {"R1": np.nan, "R2": 0.5, "R3": 0.5}):
        with pytest.raises(ad.DesignError):
            validate_weights(bad, names)
    with pytest.raises(ad.DesignError):
        validate_weights(None, names)


def test_negative_and_undefined_r2_are_kept_and_never_count_as_met():
    c = check_round(r2(-0.3, 0.97, 0.97), response_min_r2=0.9)
    assert c.per_response["R1"]["r2"] == pytest.approx(-0.3) and not c.per_response["R1"]["met"]   # not clipped to 0
    undefined = predictive_r2(np.full(6, 1.0), np.ones((1, 6)))
    c = check_round({"R1": undefined, "R2": 0.99, "R3": 0.99}, response_min_r2=0.9, aggregate_r2=0.9)
    assert not c.met and c.per_response["R1"]["status"] == "sst_zero" and np.isnan(c.per_response["R1"]["r2"])
    assert np.isnan(c.aggregate) and c.aggregate_met is False
    assert any("undefined" in u and "SST = 0" in u for u in c.unmet)
    assert not check_round({"R1": float("nan"), "R2": 0.99, "R3": 0.99}).met


def test_round_check_input_validation():
    for kw in (dict(aggregate_method="median"), dict(response_min_r2={"R1": 0.9}),):
        with pytest.raises(ad.DesignError):
            check_round(r2(0.9, 0.9, 0.9), **kw)
    with pytest.raises(ad.DesignError):
        check_round({})
    with pytest.raises(ad.DesignError):
        check_round(r2(0.9, 0.9, 0.9), aggregate_r2=0.9, aggregate_method="weighted_mean")


# ------------------------------------------------------------------ the stopping decision
def checks(*flags):
    return [check_round(r2(*(0.99 if f else 0.5,) * 3)) for f in flags]


def test_requires_consecutive_rounds_and_n_min():
    s = MultiResponseStopping(n_max=40, consecutive=3, n_min=10)
    d = s.evaluate(checks(True, True), n_runs=20, k=4)
    assert not d.stop and d.reason == "continue" and d.streak == 2 and "2/3" in d.message
    d = s.evaluate(checks(True, False, True, True), n_runs=20, k=4)
    assert not d.stop and d.streak == 2                                  # streak broken by the False round
    d = s.evaluate(checks(False, True, True, True), n_runs=20, k=4)
    assert d.stop and d.reason == "target_reached" and d.target_met and d.streak == 3
    d = s.evaluate(checks(True, True, True), n_runs=8, k=4)              # all met but fewer than N_min runs
    assert not d.stop and "N_min" in d.message


def test_default_n_min_is_k_choose_2_and_default_consecutive_is_3():
    s = MultiResponseStopping(n_max=50)
    assert s.consecutive == 3 and s.n_min_for(5) == 10 and s.n_min_for(3) == 3 and s.response_min_r2 == 0.95
    assert s.evaluate(checks(True, True, True), n_runs=9, k=5).stop is False       # 9 < 10
    assert s.evaluate(checks(True, True, True), n_runs=10, k=5).reason == "target_reached"
    assert MultiResponseStopping(n_max=50, n_min=0).n_min_for(7) == 0


def test_budget_exhausted_is_never_reported_as_success():
    s = MultiResponseStopping(n_max=20, consecutive=2, n_min=5)
    d = s.evaluate(checks(False, False), n_runs=20, k=3)
    assert d.stop and d.reason == "budget_exhausted" and d.target_met is False and "未達成" in d.message
    assert d.unmet and "R1" in d.unmet[0]
    d2 = s.evaluate(checks(True, True), n_runs=20, k=3)                  # target satisfied at the same time
    assert d2.reason == "target_reached" and d2.target_met
    assert s.evaluate(checks(False), n_runs=25, k=3).reason == "budget_exhausted"  # over budget


def test_stopping_with_aggregate_condition_in_the_loop():
    s = MultiResponseStopping(n_max=30, response_min_r2=0.90, aggregate_r2=0.95, consecutive=1, n_min=3)
    bad = [s.check(r2(0.96, 0.95, 0.89))]
    assert not s.evaluate(bad, 12, 3).stop                                # spec example 1: must continue
    good = [s.check(r2(0.96, 0.95, 0.94))]
    d = s.evaluate(good, 12, 3)
    assert d.stop and d.reason == "target_reached"                        # spec example 2: stop


def test_budget_bookkeeping_and_batch_trimming():
    s = MultiResponseStopping(n_max=20, batch_size=3)
    assert s.remaining(17) == 3 and s.remaining(25) == 0
    assert s.next_batch_size(10) == 3 and s.next_batch_size(18) == 2 and s.next_batch_size(20) == 0


def test_invalid_configuration_is_rejected():
    for kw in (dict(n_max=0), dict(n_max=2.5), dict(n_max=10, consecutive=0), dict(n_max=10, batch_size=0),
               dict(n_max=10, n_min=-1), dict(n_max=10, aggregate_method="max"),
               dict(n_max=10, response_min_r2=1.2), dict(n_max=10, aggregate_r2=2.0),
               dict(n_max=10, aggregate_r2=0.9, aggregate_method="weighted_mean"), dict(n_max=True)):
        with pytest.raises(ad.DesignError):
            MultiResponseStopping(**kw)
    with pytest.raises(TypeError):
        MultiResponseStopping()                                           # n_max has no default
    s = MultiResponseStopping(n_max=5, n_min=9)
    with pytest.raises(ad.DesignError, match="N_min"):
        s.evaluate(checks(True), 3, 4)
    with pytest.raises(ad.DesignError):
        MultiResponseStopping(n_max=10).evaluate([], 3, 3)


def test_accepts_r2result_objects_from_cross_validation():
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    good = predictive_r2(y, y[None, :] + 0.01)
    assert isinstance(good, R2Result)
    c = check_round({"a": good, "b": good}, response_min_r2=0.95)
    assert c.met and c.per_response["a"]["status"] == "ok"
