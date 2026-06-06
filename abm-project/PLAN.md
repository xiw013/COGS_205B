# abm-project2 — Implementation Plan

Sources: [PROMPT.md](PROMPT.md), [SKILL.md](SKILL.md)

## Project layout

```
abm-project2/
├── PROMPT.md
├── SKILL.md
├── PLAN.md
├── run_simulation.py      # ALL model + experiment + output code
├── test_resource_abm.py   # ALL verification tests
└── results/               # created at runtime
```

## Scientific goal

Study how **PredictiveAgent** proportion affects:
1. **Reward balance** — concentrated vs spread across agents (Gini, wealth std)
2. **Herding / oscillations** — synchronization and choice proportions over time

## Environment

- Actions: Zone A, Zone B, Stay Out
- `R_A ~ Uniform(90, 100)`, `R_B ~ Uniform(60, 80)`, Stay Out = 0
- `reward_per_agent = zone_pool / n_zone`
- N = 300, memory window = 3 rounds

## Agent rules (probabilistic via softmax)

| Agent | Rule |
|-------|------|
| RandomAgent | Uniform over 3 actions |
| MemoryAgent | Mean of last 3 public zone payoffs → softmax (τ=0.15) |
| PredictiveAgent | Models reactive flocking from public occupancy + payoff edges → softmax (τ=0.20) |

Predictive utility (public info only):
- `flock_A = mean_f_A + λ × max(0, mean_payoff_A − mean_payoff_B)`
- `u_A = mean_payoff_A − α × flock_A − β × max(0, Δf_A)`

Defaults: α=1.5, λ=2.5, β=1.0

## Conditions (4)

| ID | Random | Memory | Predictive | Counts |
|----|--------|--------|------------|--------|
| C1 | 10% | 90% | 0% | 30/270/0 |
| C2 | 10% | 60% | 30% | 30/180/90 |
| C3 | 10% | 30% | 60% | 30/90/180 |
| C4 | 10% | 0% | 90% | 30/0/270 |

4 × 20 seeds = 80 runs, 500 rounds each.

## Required outputs

### CSVs
- `seed_summary.csv` — one row per (condition, seed)
- `condition_summary.csv` — aggregated by condition
- `time_series_example.csv` — seed 0 per condition

### Figures (only these 4)

**Figure 1 — `average_reward_by_condition.png` (revised)**

Plot **mean reward by agent type** vs predictive proportion, not population-average reward.

- **X-axis:** predictive agent proportion (0%, 30%, 60%, 90%)
- **Y-axis:** average reward per round for each agent type
- **Series:** Random, Memory, Predictive — separate lines/points with error bars (std across seeds)
- **Why:** Directly shows whether predictive agents gain or lose advantage as their share grows, and how memory/random fare at each mix. Population average hides type-level effects because composition changes across conditions.
- **Edge cases:** C1 has no predictive agents (omit or leave Predictive series blank at 0%); C4 has no memory agents (omit Memory at 90%). Random is present in all conditions.

**Figure 2 — `reward_inequality_by_condition.png`**
- Gini coefficient (or wealth std) vs predictive proportion

**Figure 3 — `choice_proportions_over_time.png`**
- Zone A / B / Stay Out proportions over rounds (seed 0 per condition)

**Figure 4 — `synchronization_by_condition.png`**
- Mean majority-action proportion vs predictive proportion

## Verification (test_resource_abm.py)

- Population = 300 every round
- Non-negative zone counts
- No NaN/Inf in rewards or wealth
- Seed reproducibility
- Zone A/B reward pools fully distributed when n > 0; Stay Out = 0
- Dominant resource edge case: all-memory, R_B=0 → mostly Zone A after learning

## Run

```bash
python -m unittest test_resource_abm.py
python run_simulation.py
```

## Implementation order

1. `run_simulation.py` — constants, agents, environment, simulation, metrics, CSV/plot I/O, main()
2. `test_resource_abm.py` — unittest importing from run_simulation
3. Full run and verify outputs
