


import argparse
import json
import math
from typing import Dict

TOL = 1e-9


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def effective_crafts_per_min(machine_def: Dict, module_def: Dict, recipe_time_s: float) -> float:
    base_rate = float(machine_def["crafts_per_min"])
    speed = 1.0 + float(module_def.get("speed", 0.0))
    return base_rate * speed * 60.0 / recipe_time_s


def check_factory(input_data: Dict, output_data: Dict) -> None:
    machines = input_data.get("machines", {})
    recipes = input_data.get("recipes", {})
    modules = input_data.get("modules", {})
    limits = input_data.get("limits", {})
    raw_caps = {item: float(cap) for item, cap in limits.get("raw_supply_per_min", {}).items()}
    machine_caps = {m: float(cap) for m, cap in limits.get("max_machines", {}).items()}
    target_def = input_data.get("target", {})
    target_item = target_def.get("item")
    requested_rate = float(target_def.get("rate_per_min", 0.0))

    status = output_data.get("status")
    if status not in {"ok", "infeasible"}:
        raise AssertionError(f"Unexpected status: {status}")

    if status == "infeasible":
        max_rate = float(output_data.get("max_feasible_target_per_min", 0.0))
        if max_rate > requested_rate + 1e-6:
            raise AssertionError("max_feasible_target_per_min exceeds request")
        hints = output_data.get("bottleneck_hint")
        if not isinstance(hints, list):
            raise AssertionError("bottleneck_hint must be a list")
        return

    per_recipe = output_data.get("per_recipe_crafts_per_min", {})
    per_machine = output_data.get("per_machine_counts", {})
    raw_consumption = output_data.get("raw_consumption_per_min", {})

    recipe_rates = {name: float(rate) for name, rate in per_recipe.items()}

    produced_items = set()
    consumed_items = set()
    for recipe in recipes.values():
        produced_items.update(recipe.get("out", {}).keys())
        consumed_items.update(recipe.get("in", {}).keys())

    raw_items = set(raw_caps.keys())
    for item in consumed_items:
        if item not in produced_items:
            raw_items.add(item)
    raw_items.discard(target_item)

    item_balance = {}
    for recipe_name, recipe in recipes.items():
        machine_name = recipe["machine"]
        machine_def = machines[machine_name]
        mod = modules.get(machine_name, {})
        prod_multiplier = 1.0 + float(mod.get("prod", 0.0))
        rate = recipe_rates.get(recipe_name, 0.0)
        for item, qty in recipe.get("out", {}).items():
            item_balance[item] = item_balance.get(item, 0.0) + rate * float(qty) * prod_multiplier
        for item, qty in recipe.get("in", {}).items():
            item_balance[item] = item_balance.get(item, 0.0) - rate * float(qty)

    target_balance = item_balance.get(target_item, 0.0)
    if abs(target_balance - requested_rate) > 1e-6:
        raise AssertionError(f"Target balance mismatch: {target_balance} vs {requested_rate}")

    for item, balance in item_balance.items():
        if item == target_item:
            continue
        if item in raw_items:
            if balance > TOL:
                raise AssertionError(f"Raw item {item} has net production {balance}")
            cap = raw_caps.get(item)
            if cap is not None and -balance > cap + 1e-6:
                raise AssertionError(f"Raw item {item} consumption exceeds cap")
            expected_consumption = raw_consumption.get(item, 0.0)
            if abs(expected_consumption + balance) > 1e-6:
                raise AssertionError(f"Raw item {item} mismatch: reported {expected_consumption}, actual {-balance}")
        else:
            if abs(balance) > 1e-6:
                raise AssertionError(f"Intermediate {item} not balanced: {balance}")

    machine_usage = {}
    for recipe_name, recipe in recipes.items():
        machine_name = recipe["machine"]
        machine_usage.setdefault(machine_name, 0.0)
        time_s = float(recipe["time_s"])
        rate = recipe_rates.get(recipe_name, 0.0)
        machine_def = machines[machine_name]
        module_def = modules.get(machine_name, {})
        eff = effective_crafts_per_min(machine_def, module_def, time_s)
        if eff <= 0:
            raise AssertionError(f"Recipe {recipe_name} has invalid effective rate")
        machine_usage[machine_name] += rate / eff

    for machine, usage in machine_usage.items():
        cap = machine_caps.get(machine)
        if cap is not None and usage > cap + 1e-6:
            raise AssertionError(f"Machine {machine} exceeds cap: {usage} > {cap}")
        reported = float(per_machine.get(machine, 0.0))
        if abs(reported - usage) > 1e-6:
            raise AssertionError(f"Machine {machine} usage mismatch: reported {reported}, actual {usage}")

    # No unexpected raw consumption entries
    for item in raw_consumption:
        if item not in raw_items:
            raise AssertionError(f"Raw consumption lists non-raw item {item}")


def main():
    parser = argparse.ArgumentParser(description="Validate factory solver output against constraints.")
    parser.add_argument("input_json", help="Path to the factory input JSON.")
    parser.add_argument("output_json", help="Path to the solver output JSON.")
    args = parser.parse_args()

    input_data = load_json(args.input_json)
    output_data = load_json(args.output_json)
    check_factory(input_data, output_data)
    print("factory output valid.")


if __name__ == "__main__":
    main()
