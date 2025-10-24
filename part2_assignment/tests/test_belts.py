import math

from belts.main import BeltsSolver


def assert_close(actual, expected, tol=1e-6):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tol), (
        actual,
        expected,
    )


def test_belts_feasible_flow():
    data = {
        "edges": [
            {"from": "s1", "to": "mid", "hi": 10},
            {"from": "mid", "to": "sink", "hi": 10},
            {"from": "s2", "to": "sink", "hi": 7},
        ],
        "sources": {"s1": 5, "s2": 7},
        "sink": "sink",
    }
    solver = BeltsSolver(data)
    result = solver.solve()

    assert result["status"] == "ok"
    assert_close(result["max_flow_per_min"], 12.0)
    flows = {(f["from"], f["to"]): f["flow"] for f in result["flows"]}
    assert_close(flows[("s1", "mid")], 5.0)
    assert_close(flows[("mid", "sink")], 5.0)


def test_belts_infeasible_due_to_lower_bounds():
    data = {
        "edges": [
            {"from": "s", "to": "a", "lo": 5, "hi": 10},
            {"from": "a", "to": "sink", "hi": 3},
        ],
        "sources": {"s": 5},
        "sink": "sink",
    }
    solver = BeltsSolver(data)
    result = solver.solve()

    assert result["status"] == "infeasible"
    deficit = result["deficit"]
    assert_close(deficit["demand_balance"], 2.0)
    assert "a" in deficit["tight_nodes"]
