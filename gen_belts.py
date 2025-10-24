import argparse
import json
import random
import sys
from typing import Dict, List, Tuple


def generate_belts(num_nodes: int, num_edges: int, num_sources: int, seed: int) -> Dict:
    if num_nodes < 2:
        raise ValueError("Need at least sink plus one other node.")
    if num_sources >= num_nodes:
        raise ValueError("Number of sources must be less than total nodes.")

    rnd = random.Random(seed)
    nodes = [f"n{i}" for i in range(num_nodes)]
    sink = nodes[-1]

    other_nodes = nodes[:-1]
    rnd.shuffle(other_nodes)
    sources = other_nodes[:num_sources]

    edges: List[Dict] = []
    possible_pairs: List[Tuple[str, str]] = []
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if u == sink:
                continue  # sink has no outgoing edges
            if v in sources:
                continue  # restrict inbound to avoid source loops
            possible_pairs.append((u, v))
    rnd.shuffle(possible_pairs)
    if num_edges > len(possible_pairs):
        num_edges = len(possible_pairs)

    for idx in range(num_edges):
        u, v = possible_pairs[idx]
        lo = rnd.uniform(0.0, 10.0)
        hi = lo + rnd.uniform(5.0, 40.0)
        if rnd.random() < 0.2:
            lo = 0.0
        edge = {
            "from": u,
            "to": v,
            "lo": round(lo, 3),
            "hi": round(hi, 3),
        }
        edges.append(edge)

    source_supplies = {}
    total_supply = 0.0
    for source in sources:
        supply = rnd.uniform(10.0, 50.0)
        source_supplies[source] = round(supply, 3)
        total_supply += supply

    caps = {}
    for node in nodes:
        if node == sink or node in sources:
            continue
        if rnd.random() < 0.3:
            cap_val = rnd.uniform(total_supply * 0.3, total_supply * 0.8)
            caps[node] = {"throughput": round(cap_val, 3)}

    return {
        "edges": edges,
        "sources": source_supplies,
        "sink": sink,
        "node_caps": caps,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a random belts network scenario.")
    parser.add_argument("--nodes", type=int, default=6, help="Number of nodes including sink.")
    parser.add_argument("--edges", type=int, default=10, help="Number of directed edges.")
    parser.add_argument("--sources", type=int, default=2, help="Number of sources.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    data = generate_belts(args.nodes, args.edges, args.sources, args.seed)
    json.dump(data, fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
