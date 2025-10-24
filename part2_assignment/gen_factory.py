import argparse
import json
import random
import sys
from typing import Dict


def generate_factory(stages: int, target_rate: float, seed: int) -> Dict:
    rnd = random.Random(seed)
    machines: Dict[str, Dict] = {}
    recipes: Dict[str, Dict] = {}
    modules: Dict[str, Dict] = {}
    raw_supplies: Dict[str, float] = {}
    max_machines: Dict[str, float] = {}

    raw_item = "raw_ore"
    raw_supplies[raw_item] = target_rate * 10

    prev_item = raw_item
    for idx in range(1, stages + 1):
        machine_name = f"assembler_{idx}"
        machines[machine_name] = {"crafts_per_min": rnd.uniform(20.0, 60.0)}
        modules[machine_name] = {
            "speed": rnd.uniform(0.0, 0.3),
            "prod": rnd.uniform(0.0, 0.2),
        }
        recipe_name = f"recipe_{idx}"
        out_item = f"item_{idx}"
        craft_time = rnd.uniform(0.2, 1.5)
        input_qty = rnd.randint(1, 3)
        output_qty = rnd.randint(1, 2)
        recipes[recipe_name] = {
            "machine": machine_name,
            "time_s": craft_time,
            "in": {prev_item: input_qty},
            "out": {out_item: output_qty},
        }
        max_machines[machine_name] = rnd.uniform(50.0, 120.0)
        prev_item = out_item

    target_item = prev_item
    limits = {
        "raw_supply_per_min": raw_supplies,
        "max_machines": max_machines,
    }
    return {
        "machines": machines,
        "recipes": recipes,
        "modules": modules,
        "limits": limits,
        "target": {"item": target_item, "rate_per_min": target_rate},
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a random factory scenario.")
    parser.add_argument("--stages", type=int, default=3, help="Number of recipe stages.")
    parser.add_argument("--rate", type=float, default=120.0, help="Target output rate per minute.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    data = generate_factory(args.stages, args.rate, args.seed)
    json.dump(data, fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
