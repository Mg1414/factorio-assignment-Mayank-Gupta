## Factory Steady State Solver

### Formulation

Each recipe is assigned a decision variable representing crafts per minute. The effective outputs per craft apply the machine's productivity bonus, while inputs remain unscaled. We model steady state with linear equalities:

* Target item: total net production equals the requested rate.
* Intermediate items: inflow equals outflow (net zero).
* Raw items (either listed in `raw_supply_per_min` or never produced by any recipe): each gets an auxiliary import variable so that inflow + import = outflow. The import variable is bounded by the raw supply limit when present.

Machine usage constraints appear as linear inequalities of the form  
`∑ (crafts_r / effective_rate_r) ≤ max_machines[machine]`.

### Modules and Effective Rates

Every recipe inherits the productivity and speed bonuses of its machine type. Effective crafts per minute for recipe *r* is  
`base_crafts_per_min * (1 + speed_bonus) * 60 / recipe_time_s`.  
Productivity multiplies only outputs; imports and machine loads use the unmodified craft counts, so higher productivity reduces raw consumption as expected.

### Objective and Determinism

We minimise total machine counts, i.e. the sum over recipes of `crafts / effective_rate`. A lexicographic epsilon term on each variable breaks ties deterministically (recipes earlier in sorted order receive smaller weights). All optimisation is done with `scipy.optimize.linprog` using the HiGHS backend, which is deterministic for fixed inputs.

### Infeasibility Handling

We first solve a “max target” LP where the net target production is a free variable maximised by the objective. If the optimum is below the requested rate (within 1e-9), the instance is infeasible; we report that maximum rate and collect bottleneck hints from any binding machine or raw supply constraints. Otherwise, we re-solve with the target fixed to the requested rate while minimising machine usage. If HiGHS fails to return a feasible solution at the fixed target but the maximum rate matches, we scale the maximising solution (already within all inequality bounds) down to the requested rate.

### Numerical Choices

All constraints use an absolute tolerance of 1e-9; output values are rounded to 10 decimal places. Infinite bounds are represented with 1e15 inside the solver. Raw imports and machine counts below tolerance are reported as zero.

### Edge Cases

* Recipes with zero or negative cycle times raise errors early.
* Raw items without explicit limits are treated as having unbounded import.
* Cyclic recipe graphs are naturally handled by the equality system.
* If no recipe produces the target, the maximum feasible rate is zero with a “no recipe” hint.

## Belts Flow Solver

### Formulation

The belt problem is modelled as a linear program with one variable per edge. Lower and upper bounds translate directly to per-variable bounds. Node conservation uses `sum_in - sum_out = demand - supply`, where supplies are the fixed source rates and the sink carries demand equal to the total supply. Node throughput caps become inequality constraints on inbound and/or outbound sums; a single throughput value enforces both.

### Solving and Determinism

We minimise a tiny lexicographic objective (1e-6 times the edge index) purely to pick a deterministic feasible solution. Again, the HiGHS simplex interior solver from SciPy is deterministic for our inputs. Successful solutions report flows exactly as returned by the LP; the reported maximum flow equals the sum of supplies because conservation enforces it.

### Infeasibility Certificates

If the LP is infeasible, we build an informative certificate in two steps:

1. Run a standard max-flow on the network with capacities equal to the edges’ upper bounds and a super-source feeding the supplies. The residual cut defines `cut_reachable`, the deficit (shortfall from total supply), saturated edges (reported with `flow_needed` equal to the deficit), and node caps operating at their limits.
2. Independently, check per-node feasibility with lower bounds. For node *v*, we ensure the interval of feasible inflows that satisfies `min_in ≤ in ≤ max_in` and `min_out ≤ in + supply - demand ≤ max_out` is non-empty. Violations are appended to `tight_nodes`, flagging lower-bound inconsistencies even when pure capacity seems sufficient.

This combination yields useful hints whether the bottleneck is capacity, node caps, or lower-bound pressure.

### Numerical Details

The same 1e-9 absolute tolerance governs comparisons. Upper bounds set to infinity are replaced with `1e15`. Max-flow residual checks use an epsilon of 1e-6 to determine saturation.

### Edge Cases

* Empty flow networks succeed trivially with zero flow.
* Lower bounds exceeding upper bounds raise a `ValueError` during parsing.
* Node caps referencing unknown nodes are ignored gracefully.
* When total supply is zero the solver always succeeds with zero flows.

## Determinism and Runtime

Both tools rely exclusively on SciPy’s HiGHS solver and fixed tie-breaking weights, so identical inputs produce identical outputs. All helper routines avoid non-deterministic data structures by sorting node and edge lists during preprocessing. The workloads remain comfortably under the 2-second budget for moderate input sizes (<100 edges/recipes) on a laptop-class CPU.
