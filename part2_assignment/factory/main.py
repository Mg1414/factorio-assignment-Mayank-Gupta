
import json
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9
LEX_EPS = 1e-9


def round_for_output(value: float) -> float:
    if abs(value) < TOL:
        return 0.0
    return float(round(value + 0.0, 10))


class FactorySolver:
    
    def __init__(self, data: Dict):
        self.data = data
        self.machines = data.get("machines", {})
        self.recipes = data.get("recipes", {})
        self.modules = data.get("modules", {})
        limits = data.get("limits", {})
        self.raw_caps = {
            item: float(cap)
            for item, cap in limits.get("raw_supply_per_min", {}).items()
        }
        self.machine_caps = {
            m: float(cap) for m, cap in limits.get("max_machines", {}).items()
        }
        target = data.get("target", {})
        self.target_item = target.get("item")
        self.requested_rate = float(target.get("rate_per_min", 0.0))

        self.recipe_names = sorted(self.recipes.keys())
        self.num_recipes = len(self.recipe_names)
        if self.num_recipes == 0:
            raise ValueError("No recipes provided.")

        (
            self.recipe_machine,
            self.recipe_eff_crafts,
            self.recipe_machine_usage_coeff,
            self.item_balance_matrix,
            self.produced_items,
            self.consumed_items,
        ) = self._prepare_recipe_data()

        self.raw_items = self._identify_raw_items()
        self.intermediate_items = self._identify_intermediate_items()

        (
            self.var_index,
            self.idx_target_var,
            self.raw_var_items,
        ) = self._build_variable_index()

        self.A_eq_base, self.b_eq_base = self._build_base_equalities()
        (
            self.A_ub,
            self.b_ub,
            self.bounds,
        ) = self._build_inequalities_and_bounds()

        self.c_min_machines = self._build_machine_objective()
        self.c_max_target = self._build_max_target_objective()

    def _prepare_recipe_data(self):
        recipe_machine: List[str] = []
        recipe_eff_crafts: List[float] = []
        recipe_machine_usage_coeff: List[float] = []
        item_balance: Dict[str, np.ndarray] = {}
        produced_items = set()
        consumed_items = set()

        for idx, recipe_name in enumerate(self.recipe_names):
            recipe = self.recipes[recipe_name]
            machine_name = recipe["machine"]
            machine_def = self.machines[machine_name]
            base_crafts_per_min = float(machine_def["crafts_per_min"])
            modules = self.modules.get(machine_name, {})
            speed_multiplier = 1.0 + float(modules.get("speed", 0.0))
            prod_multiplier = 1.0 + float(modules.get("prod", 0.0))
            time_seconds = float(recipe["time_s"])
            if time_seconds <= 0.0:
                raise ValueError(f"Recipe {recipe_name} has non-positive time.")

            eff_crafts_per_min = (
                base_crafts_per_min * speed_multiplier * 60.0 / time_seconds
            )
            if eff_crafts_per_min <= 0.0:
                raise ValueError(f"Recipe {recipe_name} has zero effective speed.")

            recipe_machine.append(machine_name)
            recipe_eff_crafts.append(eff_crafts_per_min)
            recipe_machine_usage_coeff.append(1.0 / eff_crafts_per_min)

            for item, qty in recipe.get("out", {}).items():
                produced_items.add(item)
                vec = item_balance.setdefault(item, np.zeros(self.num_recipes))
                vec[idx] += float(qty) * prod_multiplier
            for item, qty in recipe.get("in", {}).items():
                consumed_items.add(item)
                vec = item_balance.setdefault(item, np.zeros(self.num_recipes))
                vec[idx] -= float(qty)

        return (
            recipe_machine,
            recipe_eff_crafts,
            recipe_machine_usage_coeff,
            item_balance,
            produced_items,
            consumed_items,
        )

    def _identify_raw_items(self) -> Dict[str, Optional[float]]:
        raw_items: Dict[str, Optional[float]] = {
            item: cap for item, cap in self.raw_caps.items()
        }
        for item in self.consumed_items:
            if item not in self.produced_items:
                raw_items.setdefault(item, None)
        if self.target_item in raw_items:
            raw_items.pop(self.target_item, None)
        return raw_items

    def _identify_intermediate_items(self) -> List[str]:
        intermediate = []
        for item in sorted(self.item_balance_matrix.keys()):
            if item == self.target_item:
                continue
            if item in self.raw_items:
                continue
            produced = item in self.produced_items
            consumed = item in self.consumed_items
            if produced and consumed:
                intermediate.append(item)
        return intermediate

    def _build_variable_index(self):
        var_index: Dict[Tuple[str, Optional[str]], int] = {}
        for idx, name in enumerate(self.recipe_names):
            var_index[("recipe", name)] = idx
        raw_var_items = sorted(self.raw_items.keys())
        for item in raw_var_items:
            var_index[("raw", item)] = len(var_index)
        idx_target_var = len(var_index)
        var_index[("target", None)] = idx_target_var
        return var_index, idx_target_var, raw_var_items

    def _build_base_equalities(self):
        rows = []
        rhs = []
        for row, value in self._iter_balance_rows():
            rows.append(row)
            rhs.append(value)
        if rows:
            A_eq = np.vstack(rows)
        else:
            A_eq = np.zeros((0, len(self.var_index)))
        b_eq = np.array(rhs)
        return A_eq, b_eq

    def _iter_balance_rows(self):
        """Yield item balance rows one by one.

        This extra layer came from a later clarity pass when the raw and
        intermediate handling diverged in requirements.
        """
        yield (
            self._item_balance_row(self.target_item, include_raw=False, include_target=True),
            0.0,
        )
        for item in sorted(self.raw_items.keys()):
            yield (
                self._item_balance_row(item, include_raw=True, include_target=False),
                0.0,
            )
        for item in self.intermediate_items:
            yield (
                self._item_balance_row(item, include_raw=False, include_target=False),
                0.0,
            )

    def _item_balance_row(self, item: str, *, include_raw: bool, include_target: bool) -> np.ndarray:
        row = np.zeros(len(self.var_index))
        vec = self.item_balance_matrix.get(item)
        if vec is not None:
            row[: self.num_recipes] = vec
        if include_raw and item in self.raw_items:
            idx = self.var_index[("raw", item)]
            row[idx] = 1.0
        if include_target:
            row[self.idx_target_var] = -1.0
        return row

    def _build_inequalities_and_bounds(self):
        rows = []
        rhs = []
        bounds: List[Tuple[float, Optional[float]]] = []
        for name in self.recipe_names:
            bounds.append((0.0, None))
        for item in self.raw_var_items:
            cap = self.raw_items[item]
            if cap is None or math.isinf(cap):
                bounds.append((0.0, None))
            else:
                bounds.append((0.0, cap))
        bounds.append((0.0, None))  # target production variable

        machines_used = {}
        for recipe_idx, recipe_name in enumerate(self.recipe_names):
            machine = self.recipe_machine[recipe_idx]
            usage_coeff = self.recipe_machine_usage_coeff[recipe_idx]
            machines_used.setdefault(machine, np.zeros(self.num_recipes))
            machines_used[machine][recipe_idx] += usage_coeff

        for machine, coeff_vec in sorted(machines_used.items()):
            cap = self.machine_caps.get(machine)
            if cap is None or math.isinf(cap):
                continue
            row = np.zeros(len(self.var_index))
            row[: self.num_recipes] = coeff_vec
            rows.append(row)
            rhs.append(cap)

        if rows:
            A_ub = np.vstack(rows)
            b_ub = np.array(rhs)
        else:
            A_ub = None
            b_ub = None
        return A_ub, b_ub, bounds

    def _build_machine_objective(self):
        c = np.zeros(len(self.var_index))
        for idx, recipe_name in enumerate(self.recipe_names):
            machine_weight = self.recipe_machine_usage_coeff[idx]
            tie = (idx + 1) * LEX_EPS
            c[idx] = machine_weight + tie
        return c

    def _build_max_target_objective(self):
        c = np.zeros(len(self.var_index))
        c[self.idx_target_var] = -1.0
        return c

    def solve(self):
        max_rate, max_solution = self._maximize_target()
        if max_rate + TOL < self.requested_rate:
            hints = self._collect_bottleneck_hints(max_solution)
            result = {
                "status": "infeasible",
                "max_feasible_target_per_min": round_for_output(max_rate),
                "bottleneck_hint": hints,
            }
            return result

        target_solution = self._solve_for_target(self.requested_rate)
        if target_solution is None:
            # Fallback: scale maximum solution if numerically feasible.
            fallback = self._scale_solution_to_target(max_solution, self.requested_rate)
            if fallback is None:
                hints = self._collect_bottleneck_hints(max_solution)
                result = {
                    "status": "infeasible",
                    "max_feasible_target_per_min": round_for_output(max_rate),
                    "bottleneck_hint": hints,
                }
                return result
            target_solution = fallback

        per_recipe = self._extract_recipe_rates(target_solution)
        per_machine = self._extract_machine_usage(target_solution)
        raw_consumption = self._extract_raw_usage(target_solution)
        result = {
            "status": "ok",
            "per_recipe_crafts_per_min": per_recipe,
            "per_machine_counts": per_machine,
            "raw_consumption_per_min": raw_consumption,
        }
        return result

    def _maximize_target(self):
        res = linprog(
            c=self.c_max_target,
            A_eq=self.A_eq_base,
            b_eq=self.b_eq_base,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=self.bounds,
            method="highs",
        )
        if not res.success:
            # No feasible production at all; treat as zero solution.
            zero = np.zeros(len(self.var_index))
            return 0.0, zero
        max_rate = max(0.0, res.x[self.idx_target_var])
        return max_rate, res.x

    def _solve_for_target(self, target_rate: float) -> Optional[np.ndarray]:
        row = np.zeros(len(self.var_index))
        row[self.idx_target_var] = 1.0
        A_eq = np.vstack([self.A_eq_base, row])
        b_eq = np.concatenate([self.b_eq_base, np.array([target_rate])])
        res = linprog(
            c=self.c_min_machines,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=self.bounds,
            method="highs",
        )
        if not res.success:
            return None
        if res.x[self.idx_target_var] < target_rate - 1e-6:
            return None
        return res.x

    def _scale_solution_to_target(
        self, solution: np.ndarray, target_rate: float
    ) -> Optional[np.ndarray]:
        current_rate = solution[self.idx_target_var]
        if current_rate <= 0:
            return None
        scale = target_rate / current_rate
        if scale > 1.0 + 1e-6:
            return None
        scaled = solution.copy()
        scaled[: self.num_recipes] *= scale
        for item in self.raw_var_items:
            idx = self.var_index[("raw", item)]
            scaled[idx] *= scale
            cap = self.raw_items[item]
            if cap is not None and scaled[idx] > cap + 1e-6:
                return None
        for machine, cap in self.machine_caps.items():
            if cap is None or math.isinf(cap):
                continue
            usage = 0.0
            for recipe_idx, recipe_name in enumerate(self.recipe_names):
                if self.recipe_machine[recipe_idx] != machine:
                    continue
                usage += scaled[recipe_idx] * self.recipe_machine_usage_coeff[recipe_idx]
            if usage > cap + 1e-6:
                return None
        scaled[self.idx_target_var] = target_rate
        return scaled

    def _collect_bottleneck_hints(self, solution: np.ndarray) -> List[str]:
        hints = []
        if solution is None:
            solution = np.zeros(len(self.var_index))

        for machine, cap in sorted(self.machine_caps.items()):
            if cap is None or math.isinf(cap):
                continue
            usage = 0.0
            for recipe_idx, recipe_name in enumerate(self.recipe_names):
                if self.recipe_machine[recipe_idx] != machine:
                    continue
                usage += solution[recipe_idx] * self.recipe_machine_usage_coeff[recipe_idx]
            if cap <= 0.0:
                hints.append(f"{machine} cap")
                continue
            if usage >= cap - 1e-6:
                hints.append(f"{machine} cap")

        for item in self.raw_var_items:
            cap = self.raw_items[item]
            if cap is None or math.isinf(cap):
                continue
            idx = self.var_index[("raw", item)]
            amount = solution[idx]
            if cap <= 0.0:
                hints.append(f"{item} supply")
                continue
            if amount >= cap - 1e-6:
                hints.append(f"{item} supply")

        if not hints:
            if self.target_item not in self.produced_items:
                hints.append(f"no recipe for {self.target_item}")
            else:
                hints.append("unspecified bottleneck")
        return hints

    def _extract_recipe_rates(self, solution: np.ndarray) -> Dict[str, float]:
        per_recipe = {}
        for idx, name in enumerate(self.recipe_names):
            per_recipe[name] = round_for_output(solution[idx])
        return per_recipe

    def _extract_machine_usage(self, solution: np.ndarray) -> Dict[str, float]:
        usage_per_machine: Dict[str, float] = {}
        for idx, recipe_name in enumerate(self.recipe_names):
            machine = self.recipe_machine[idx]
            usage = solution[idx] * self.recipe_machine_usage_coeff[idx]
            usage_per_machine[machine] = usage_per_machine.get(machine, 0.0) + usage
        for machine in usage_per_machine:
            usage_per_machine[machine] = round_for_output(usage_per_machine[machine])
        return usage_per_machine

    def _extract_raw_usage(self, solution: np.ndarray) -> Dict[str, float]:
        raw_usage = {}
        for item in self.raw_var_items:
            idx = self.var_index[("raw", item)]
            val = round_for_output(solution[idx])
            if val > 0.0:
                raw_usage[item] = val
        return raw_usage


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON input: {exc}") from exc

    solver = FactorySolver(data)
    result = solver.solve()
    json.dump(result, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
