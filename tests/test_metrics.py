"""metrics.py against a hand-computed toy fixture."""

import math

from agentic_distill.metrics import bootstrap, group_by_task, metrics_of, pass_hat_k


def rec(task, trial, reward, calls=2, env_err=0, steps=5, term="done", cost=0.01, loop=False, parse=0):
    return {"task_id": task, "trial": trial, "reward": reward, "n_tool_calls": calls, "n_env_tool_errors": env_err,
            "n_parse_invalid": parse, "n_steps": steps, "terminated_by": term, "cost_usd": cost, "loop_detected": loop}


# A: 2/3 succeed, B: 3/3, C: 0/3
RECS = ([rec("A", 0, 1), rec("A", 1, 1), rec("A", 2, 0, env_err=1, term="abort", steps=3)]
        + [rec("B", t, 1) for t in range(3)] + [rec("C", t, 0, calls=4, env_err=2, loop=(t == 0)) for t in range(3)])


def test_pass_hat_k_hand_values():
    assert pass_hat_k(2, 3, 1) == 2 / 3 and pass_hat_k(2, 3, 2) == 1 / 3 and pass_hat_k(2, 3, 3) == 0
    assert math.isnan(pass_hat_k(1, 2, 3))


def test_metrics_hand_values():
    m = metrics_of(list(group_by_task(RECS).values()))
    assert abs(m["pass^1"] - 5 / 9) < 1e-12 and abs(m["pass^2"] - 4 / 9) < 1e-12 and abs(m["pass^3"] - 1 / 3) < 1e-12
    assert abs(m["cost_per_success"] - (9 * 0.01) / 5) < 1e-12          # 9 episodes, 5 successes
    # calls: A 2+2+2, B 6, C 12 = 24; env errors: 1 + 6 = 7
    assert abs(m["valid_call_rate"] - (1 - 7 / 24)) < 1e-12
    assert abs(m["loop_abort_rate"] - 2 / 9) < 1e-12                    # one abort, one loop


def test_steps_mean():
    m = metrics_of(list(group_by_task(RECS).values()))
    assert abs(m["steps"] - (5 + 5 + 3 + 15 + 15) / 9) < 1e-12


def test_bootstrap_reproducible_and_bracketing():
    g = group_by_task(RECS)
    p1, c1 = bootstrap(g, n_boot=300, seed=1)
    p2, c2 = bootstrap(g, n_boot=300, seed=1)
    assert c1 == c2 and c1["pass^1"][0] <= p1["pass^1"] <= c1["pass^1"][1]


def test_paired_identical_is_zero():
    g = group_by_task(RECS)
    point, cis = bootstrap(g, n_boot=200, seed=0, other=g)
    assert point["pass^1"] == 0 and cis["pass^1"] == (0, 0)
