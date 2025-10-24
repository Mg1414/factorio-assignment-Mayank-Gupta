import argparse
import json
import shlex
import subprocess
import sys


FACTORY_SAMPLE = {
    "machines": {
        "assembler": {"crafts_per_min": 30},
        "press": {"crafts_per_min": 30},
    },
    "recipes": {
        "plates": {
            "machine": "press",
            "time_s": 45.0,
            "in": {"ore": 1},
            "out": {"plate": 1},
        },
        "gears": {
            "machine": "assembler",
            "time_s": 30.0,
            "in": {"plate": 2},
            "out": {"gear": 1},
        },
    },
    "limits": {
        "raw_supply_per_min": {"ore": 500},
        "max_machines": {"assembler": 20, "press": 20},
    },
    "target": {"item": "gear", "rate_per_min": 90},
}

BELTS_SAMPLE = {
    "edges": [
        {"from": "s1", "to": "mid", "hi": 120},
        {"from": "mid", "to": "sink", "hi": 100},
        {"from": "s2", "to": "sink", "hi": 60},
    ],
    "sources": {"s1": 80, "s2": 40},
    "sink": "sink",
}


def run_command(command: str, payload: dict) -> dict:
    cmd = shlex.split(command)
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Run sample scenarios for factory and belts tools.")
    parser.add_argument("factory_cmd", help="Command to execute the factory CLI.")
    parser.add_argument("belts_cmd", help="Command to execute the belts CLI.")
    args = parser.parse_args()

    factory_output = run_command(args.factory_cmd, FACTORY_SAMPLE)
    belts_output = run_command(args.belts_cmd, BELTS_SAMPLE)

    json.dump(factory_output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    json.dump(belts_output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
