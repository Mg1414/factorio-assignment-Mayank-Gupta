import math

from factory.main import FactorySolver


def assert_close(actual, expected, tol=1e-6):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tol), (
        actual,
        expected,
    )


def test_factory_basic_production():
    data = {
        "machines": {
            "press": {"crafts_per_min": 30},
            "assembler": {"crafts_per_min": 30},
        },
        "recipes": {
            "plates": {
                "machine": "press",
                "time_s": 60.0,
                "in": {"ore": 1},
                "out": {"plate": 1},
            },
            "gears": {
                "machine": "assembler",
                "time_s": 60.0,
                "in": {"plate": 2},
                "out": {"gear": 1},
            },
        },
        "limits": {
            "raw_supply_per_min": {"ore": 100},
            "max_machines": {"press": 10, "assembler": 10},
        },
        "target": {"item": "gear", "rate_per_min": 30},
    }

    solver = FactorySolver(data)
    result = solver.solve()

    assert result["status"] == "ok"
    crafts = result["per_recipe_crafts_per_min"]
    assert_close(crafts["plates"], 60.0)
    assert_close(crafts["gears"], 30.0)

    machines = result["per_machine_counts"]
    assert_close(machines["press"], 2.0)
    assert_close(machines["assembler"], 1.0)

    raw_use = result["raw_consumption_per_min"]
    assert_close(raw_use["ore"], 60.0)


def test_factory_machine_bottleneck_reports():
    data = {
        "machines": {
            "press": {"crafts_per_min": 30},
            "assembler": {"crafts_per_min": 30},
        },
        "recipes": {
            "plates": {
                "machine": "press",
                "time_s": 60.0,
                "in": {"ore": 1},
                "out": {"plate": 1},
            },
            "gears": {
                "machine": "assembler",
                "time_s": 60.0,
                "in": {"plate": 2},
                "out": {"gear": 1},
            },
        },
        "limits": {
            "raw_supply_per_min": {"ore": 1000},
            "max_machines": {"press": 1, "assembler": 10},
        },
        "target": {"item": "gear", "rate_per_min": 30},
    }

    solver = FactorySolver(data)
    result = solver.solve()

    assert result["status"] == "infeasible"
    assert_close(result["max_feasible_target_per_min"], 15.0)
    hints = result["bottleneck_hint"]
    assert any("press cap" in hint for hint in hints)
