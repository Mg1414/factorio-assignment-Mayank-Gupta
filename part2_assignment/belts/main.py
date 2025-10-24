import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9
INF_CAP = 1e15


def round_for_output(value: float) -> float:
    if abs(value) < TOL:
        return 0.0
    return float(round(value + 0.0, 10))


@dataclass
class EdgeData:
    src: str
    dst: str
    lower: float
    upper: float


class MaxFlowEdge:
    __slots__ = ("to", "rev", "cap", "init_cap")

    def __init__(self, to: int, rev: int, cap: float):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.init_cap = cap


class MaxFlow:
    def __init__(self):
        self.graph: List[List[MaxFlowEdge]] = []

    def _ensure(self, idx: int):
        while idx >= len(self.graph):
            self.graph.append([])

    def add_edge(self, u: int, v: int, cap: float) -> Tuple[int, int]:
        self._ensure(max(u, v))
        forward = MaxFlowEdge(v, len(self.graph[v]), cap)
        backward = MaxFlowEdge(u, len(self.graph[u]), 0.0)
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return u, len(self.graph[u]) - 1

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        n = len(self.graph)
        level = [0] * n

        while True:
            for i in range(n):
                level[i] = -1
            level[s] = 0
            queue = deque([s])
            while queue:
                v = queue.popleft()
                for edge in self.graph[v]:
                    if edge.cap > TOL and level[edge.to] < 0:
                        level[edge.to] = level[v] + 1
                        queue.append(edge.to)
            if level[t] < 0:
                break
            it = [0] * n

            def dfs(v: int, pushed: float) -> float:
                if v == t:
                    return pushed
                for i in range(it[v], len(self.graph[v])):
                    it[v] = i
                    edge = self.graph[v][i]
                    if edge.cap > TOL and level[v] < level[edge.to]:
                        to_push = dfs(edge.to, min(pushed, edge.cap))
                        if to_push > TOL:
                            edge.cap -= to_push
                            back = self.graph[edge.to][edge.rev]
                            back.cap += to_push
                            return to_push
                return 0.0

            while True:
                pushed = dfs(s, math.inf)
                if pushed <= TOL:
                    break
                flow += pushed
        return flow

    def reachable(self, s: int) -> List[bool]:
        visited = [False] * len(self.graph)
        stack = [s]
        visited[s] = True
        while stack:
            v = stack.pop()
            for edge in self.graph[v]:
                if edge.cap > TOL and not visited[edge.to]:
                    visited[edge.to] = True
                    stack.append(edge.to)
        return visited


class BeltsSolver:
    def __init__(self, data: Dict):
        self.data = data
        self.edges_raw = data.get("edges", [])
        self.sources = data.get("sources", {})
        self.sink = data.get("sink")
        if self.sink is None:
            raise ValueError("Sink node is required.")
        self.node_caps = data.get("node_caps", {})

        self.edges: List[EdgeData] = []
        self.nodes: List[str] = []
        self.node_index: Dict[str, int] = {}
        self.incoming: Dict[str, List[int]] = defaultdict(list)
        self.outgoing: Dict[str, List[int]] = defaultdict(list)

        # Initial parsing pass. Later iterations reshaped this but the call
        # remains in __init__ so downstream code assumes populated structures.
        self._parse_edges()

    def _parse_edges(self):
        node_set = set()
        for idx, edge in enumerate(self.edges_raw):
            src = edge["from"]
            dst = edge["to"]
            lo = float(edge.get("lo", 0.0))
            hi_raw = edge.get("hi", INF_CAP)
            hi = float(hi_raw if hi_raw is not None else INF_CAP)
            if hi < lo - TOL:
                raise ValueError(f"Edge {src}->{dst} has upper bound below lower bound.")
            self.edges.append(EdgeData(src=src, dst=dst, lower=lo, upper=hi))
            node_set.add(src)
            node_set.add(dst)
            edge_idx = len(self.edges) - 1
            self.outgoing[src].append(edge_idx)
            self.incoming[dst].append(edge_idx)
        node_set.update(self.sources.keys())
        node_set.add(self.sink)
        self.nodes = sorted(node_set)
        self.node_index = {name: idx for idx, name in enumerate(self.nodes)}

    def solve(self):
        # Historically this returned both the result and A_eq/b_eq, but after a
        # follow-up tidy up we now rely on the stored attributes.
        lp_result = self._solve_linear_program()
        if lp_result.success:
            return self._format_success(lp_result)
        return self._build_certificate()

    def _solve_linear_program(self):
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        A_eq = np.zeros((num_nodes, num_edges))
        b_eq = np.zeros(num_nodes)

        total_supply = sum(float(v) for v in self.sources.values())
        sink_demand = total_supply

        for node in self.nodes:
            row_idx = self.node_index[node]
            for edge_idx in self.incoming[node]:
                A_eq[row_idx, edge_idx] += 1.0
            for edge_idx in self.outgoing[node]:
                A_eq[row_idx, edge_idx] -= 1.0
            supply = float(self.sources.get(node, 0.0))
            demand = sink_demand if node == self.sink else 0.0
            b_eq[row_idx] = demand - supply

        A_ub_rows = []
        b_ub_vals = []
        for node, caps in self.node_caps.items():
            if node not in self.node_index:
                continue
            cap_in = None
            cap_out = None
            cap_throughput = None
            if isinstance(caps, dict):
                if "in" in caps:
                    cap_in = float(caps["in"])
                if "out" in caps:
                    cap_out = float(caps["out"])
                if "throughput" in caps:
                    cap_throughput = float(caps["throughput"])
                if "cap" in caps:
                    val = float(caps["cap"])
                    cap_throughput = val if cap_throughput is None else min(cap_throughput, val)
            else:
                cap_throughput = float(caps)

            if cap_throughput is not None:
                cap_in = cap_in if cap_in is not None else cap_throughput
                cap_out = cap_out if cap_out is not None else cap_throughput

            if cap_in is not None:
                row = np.zeros(num_edges)
                for edge_idx in self.incoming.get(node, []):
                    row[edge_idx] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(cap_in)
            if cap_out is not None:
                row = np.zeros(num_edges)
                for edge_idx in self.outgoing.get(node, []):
                    row[edge_idx] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(cap_out)

        A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_vals) if A_ub_rows else None

        bounds = []
        c = np.zeros(num_edges)
        for idx, edge in enumerate(self.edges):
            lower = edge.lower
            upper = edge.upper if not math.isinf(edge.upper) else INF_CAP
            bounds.append((lower, upper))
            c[idx] = (idx + 1) * 1e-6

        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        self.total_supply = total_supply
        self.lp_result = result
        self.A_eq = A_eq
        self.b_eq = b_eq
        return result

    def _format_success(self, result):
        flows = []
        for edge, value in zip(self.edges, result.x):
            flows.append(
                {
                    "from": edge.src,
                    "to": edge.dst,
                    "flow": round_for_output(value),
                }
            )
        response = {
            "status": "ok",
            "max_flow_per_min": round_for_output(self.total_supply),
            "flows": flows,
        }
        return response

    def _build_certificate(self):
        cut_info = self._max_flow_cut()
        lb_conflicts = self._lower_bound_conflicts()

        tight_nodes = sorted(set(cut_info["tight_nodes"]) | lb_conflicts)
        certificate = {
            "status": "infeasible",
            "cut_reachable": cut_info["cut_nodes"],
            "deficit": {
                "demand_balance": round_for_output(cut_info["deficit"]),
                "tight_nodes": tight_nodes,
                "tight_edges": cut_info["tight_edges"],
            },
        }
        return certificate

    def _max_flow_cut(self):
        graph = MaxFlow()
        node_ids = {}

        def get_id(name: str) -> int:
            if name not in node_ids:
                node_ids[name] = len(node_ids)
                graph._ensure(node_ids[name])
            return node_ids[name]

        super_source = "__mf_source__"
        super_sink = self.sink
        source_id = get_id(super_source)
        sink_id = get_id(super_sink)

        edge_handles = []
        for edge in self.edges:
            u = get_id(edge.src)
            v = get_id(edge.dst)
            cap = edge.upper if not math.isinf(edge.upper) else INF_CAP
            edge_handles.append((u, len(graph.graph[u])))
            graph.add_edge(u, v, cap)

        for node, supply in self.sources.items():
            cap = float(supply)
            u = source_id
            v = get_id(node)
            graph.add_edge(u, v, cap)

        flow_value = graph.max_flow(source_id, sink_id)
        visited = graph.reachable(source_id)

        cut_nodes = sorted(
            node
            for node, idx in node_ids.items()
            if visited[idx] and not node.startswith("__")
        )

        node_inflows = defaultdict(float)
        node_outflows = defaultdict(float)
        for (u_idx, edge_pos), edge in zip(edge_handles, self.edges):
            edge_obj = graph.graph[u_idx][edge_pos]
            flow = edge_obj.init_cap - edge_obj.cap
            node_outflows[edge.src] += flow
            node_inflows[edge.dst] += flow

        tight_nodes = []
        for node, caps in self.node_caps.items():
            inflow = node_inflows.get(node, 0.0)
            outflow = node_outflows.get(node, 0.0)
            in_cap = None
            out_cap = None
            if isinstance(caps, dict):
                if "in" in caps:
                    in_cap = float(caps["in"])
                if "out" in caps:
                    out_cap = float(caps["out"])
                throughput = caps.get("throughput")
                if throughput is not None:
                    throughput = float(throughput)
                    in_cap = throughput if in_cap is None else min(in_cap, throughput)
                    out_cap = throughput if out_cap is None else min(out_cap, throughput)
                if "cap" in caps:
                    cap_val = float(caps["cap"])
                    in_cap = cap_val if in_cap is None else min(in_cap, cap_val)
                    out_cap = cap_val if out_cap is None else min(out_cap, cap_val)
            else:
                cap_val = float(caps)
                in_cap = cap_val
                out_cap = cap_val
            if in_cap is not None and inflow >= in_cap - 1e-6:
                tight_nodes.append(node)
            if out_cap is not None and outflow >= out_cap - 1e-6:
                tight_nodes.append(node)

        deficit = max(0.0, self.total_supply - flow_value)
        tight_edges = []
        for (u_idx, edge_pos), edge in zip(edge_handles, self.edges):
            if visited[u_idx]:
                edge_obj = graph.graph[u_idx][edge_pos]
                v_idx = edge_obj.to
                if not visited[v_idx]:
                    capacity = edge_obj.init_cap
                    used = capacity - edge_obj.cap
                    if capacity > 0 and capacity - used <= 1e-6:
                        tight_edges.append(
                            {
                                "from": edge.src,
                                "to": edge.dst,
                                "flow_needed": round_for_output(deficit),
                            }
                        )
        return {
            "cut_nodes": cut_nodes,
            "tight_edges": tight_edges,
            "tight_nodes": tight_nodes,
            "deficit": deficit,
        }

    def _lower_bound_conflicts(self) -> set:
        conflicts = set()
        sink = self.sink
        total_supply = self.total_supply
        for node in self.nodes:
            supply = float(self.sources.get(node, 0.0))
            demand = total_supply if node == sink else 0.0

            min_in = sum(self.edges[idx].lower for idx in self.incoming.get(node, []))
            max_in = sum(self.edges[idx].upper for idx in self.incoming.get(node, []))
            min_out = sum(self.edges[idx].lower for idx in self.outgoing.get(node, []))
            max_out = sum(self.edges[idx].upper for idx in self.outgoing.get(node, []))

            left = max(min_in, min_out - supply + demand)
            right = min(max_in, max_out - supply + demand)
            if left > right + 1e-6:
                conflicts.add(node)
        return conflicts


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON input: {exc}") from exc

    solver = BeltsSolver(data)
    result = solver.solve()
    json.dump(result, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
