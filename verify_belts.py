
import argparse
import json
import math
from collections import defaultdict
from typing import Dict, Tuple

TOL = 1e-9
INF_CAP = 1e15


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_caps(caps_raw) -> Tuple[float, float]:
    cap_in = None
    cap_out = None
    if caps_raw is None:
        return cap_in, cap_out
    if isinstance(caps_raw, dict):
        if "in" in caps_raw:
            cap_in = float(caps_raw["in"])
        if "out" in caps_raw:
            cap_out = float(caps_raw["out"])
        throughput = caps_raw.get("throughput")
        if throughput is not None:
            throughput = float(throughput)
            cap_in = throughput if cap_in is None else min(cap_in, throughput)
            cap_out = throughput if cap_out is None else min(cap_out, throughput)
        if "cap" in caps_raw:
            cap_val = float(caps_raw["cap"])
            cap_in = cap_val if cap_in is None else min(cap_in, cap_val)
            cap_out = cap_val if cap_out is None else min(cap_out, cap_val)
    else:
        cap_val = float(caps_raw)
        cap_in = cap_val
        cap_out = cap_val
    return cap_in, cap_out


def check_ok(input_data: Dict, output_data: Dict) -> None:
    edges = input_data.get("edges", [])
    sources = {k: float(v) for k, v in input_data.get("sources", {}).items()}
    sink = input_data.get("sink")
    node_caps = input_data.get("node_caps", {})

    expected_flow = sum(sources.values())
    reported_flow = float(output_data.get("max_flow_per_min", 0.0))
    if abs(expected_flow - reported_flow) > 1e-6:
        raise AssertionError(f"max_flow_per_min mismatch: expected {expected_flow}, reported {reported_flow}")

    flow_entries = output_data.get("flows", [])
    edge_flow = defaultdict(float)
    for entry in flow_entries:
        frm = entry["from"]
        to = entry["to"]
        flow = float(entry["flow"])
        if flow < -1e-8:
            raise AssertionError("Negative flow reported.")
        edge_flow[(frm, to)] += flow

    lower = {}
    upper = {}
    for edge in edges:
        frm = edge["from"]
        to = edge["to"]
        lo = float(edge.get("lo", 0.0))
        hi_raw = edge.get("hi")
        hi = INF_CAP if hi_raw is None else float(hi_raw)
        lower[(frm, to)] = lo
        upper[(frm, to)] = hi

    for key, lo in lower.items():
        hi = upper[key]
        flow = edge_flow.get(key, 0.0)
        if flow < lo - 1e-6:
            raise AssertionError(f"Edge {key} below lower bound: {flow} < {lo}")
        if flow > hi + 1e-6:
            raise AssertionError(f"Edge {key} above upper bound: {flow} > {hi}")

    for key in edge_flow:
        if key not in lower:
            raise AssertionError(f"Output flow references missing edge {key}")

    node_in = defaultdict(float)
    node_out = defaultdict(float)
    for (frm, to), flow in edge_flow.items():
        node_out[frm] += flow
        node_in[to] += flow

    total_supply = sum(sources.values())
    for node in set(list(node_in.keys()) + list(node_out.keys()) + list(sources.keys()) + [sink]):
        supply = sources.get(node, 0.0)
        demand = total_supply if node == sink else 0.0
        lhs = node_in.get(node, 0.0) + supply
        rhs = node_out.get(node, 0.0) + demand
        if abs(lhs - rhs) > 1e-6:
            raise AssertionError(f"Conservation violated at {node}: in+supply={lhs}, out+demand={rhs}")

    for node, caps_raw in node_caps.items():
        cap_in, cap_out = parse_caps(caps_raw)
        inflow = node_in.get(node, 0.0)
        outflow = node_out.get(node, 0.0)
        if cap_in is not None and inflow > cap_in + 1e-6:
            raise AssertionError(f"Node {node} inbound cap exceeded: {inflow} > {cap_in}")
        if cap_out is not None and outflow > cap_out + 1e-6:
            raise AssertionError(f"Node {node} outbound cap exceeded: {outflow} > {cap_out}")


def check_infeasible(output_data: Dict) -> None:
    if "max_feasible_target_per_min" in output_data:
        raise AssertionError("Factory fields present in belts output.")
    deficit = output_data.get("deficit")
    if not isinstance(deficit, dict):
        raise AssertionError("deficit must be provided for infeasible instances.")
    _ = float(deficit.get("demand_balance", 0.0))
    _ = deficit.get("tight_nodes", [])
    _ = deficit.get("tight_edges", [])
    cut = output_data.get("cut_reachable", [])
    if not isinstance(cut, list):
        raise AssertionError("cut_reachable must be a list.")


def main():
    parser = argparse.ArgumentParser(description="Validate belts solver output.")
    parser.add_argument("input_json", help="Path to the belts input JSON.")
    parser.add_argument("output_json", help="Path to the solver output JSON.")
    args = parser.parse_args()

    input_data = load_json(args.input_json)
    output_data = load_json(args.output_json)
    status = output_data.get("status")
    if status == "ok":
        check_ok(input_data, output_data)
        print("belts output valid.")
    elif status == "infeasible":
        check_infeasible(output_data)
        print("belts infeasibility certificate looks well-formed.")
    else:
        raise AssertionError(f"Unexpected status: {status}")


if __name__ == "__main__":
    main()
