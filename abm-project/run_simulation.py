"""Competitive resource ABM: single-file model, experiment, and outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Constants ---
N = 300
T_ROUNDS = 500
N_SEEDS = 20
MEMORY_WINDOW = 3

ZONE_A = 0
ZONE_B = 1
STAY_OUT = 2

ZONE_A_LOW, ZONE_A_HIGH = 90.0, 100.0
ZONE_B_LOW, ZONE_B_HIGH = 60.0, 80.0
STAY_OUT_REWARD = 0.0

TAU_MEMORY = 0.15
TAU_PREDICTIVE = 0.20
CROWDING_ALPHA = 1.5
FLOCK_SENSITIVITY = 2.5
TREND_BETA = 1.0

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Condition:
    condition_id: int
    n_random: int
    n_memory: int
    n_predictive: int

    @property
    def pct_predictive(self) -> float:
        return self.n_predictive / N


CONDITIONS = [
    Condition(1, 30, 270, 0),
    Condition(2, 30, 255, 15),
    Condition(3, 30, 180, 90),
    Condition(4, 30, 135, 135),
    Condition(5, 30, 90, 180),
    Condition(6, 30, 15, 255),
    Condition(7, 30, 0, 270),
]


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def run_seed(condition_id: int, seed: int) -> int:
    return seed + condition_id * 10000


def draw_zone_pools(rng: np.random.Generator) -> tuple[float, float, float]:
    r_a = float(rng.uniform(ZONE_A_LOW, ZONE_A_HIGH))
    r_b = float(rng.uniform(ZONE_B_LOW, ZONE_B_HIGH))
    return r_a, r_b, STAY_OUT_REWARD


def allocate_rewards(
    actions: np.ndarray, r_a: float, r_b: float
) -> tuple[np.ndarray, int, int, int]:
    n_a = int(np.sum(actions == ZONE_A))
    n_b = int(np.sum(actions == ZONE_B))
    n_stay = int(np.sum(actions == STAY_OUT))
    rewards = np.zeros(len(actions), dtype=float)
    if n_a > 0:
        rewards[actions == ZONE_A] = r_a / n_a
    if n_b > 0:
        rewards[actions == ZONE_B] = r_b / n_b
    return rewards, n_a, n_b, n_stay


def empty_observation() -> dict:
    return {
        "round": -1,
        "n_a": 0,
        "n_b": 0,
        "n_stay": 0,
        "f_a": 0.0,
        "f_b": 0.0,
        "f_stay": 0.0,
        "payoff_a": 0.0,
        "payoff_b": 0.0,
        "r_a": 0.0,
        "r_b": 0.0,
    }


def build_observation(
    round_idx: int, n_a: int, n_b: int, n_stay: int, r_a: float, r_b: float
) -> dict:
    payoff_a = r_a / n_a if n_a > 0 else 0.0
    payoff_b = r_b / n_b if n_b > 0 else 0.0
    return {
        "round": round_idx,
        "n_a": n_a,
        "n_b": n_b,
        "n_stay": n_stay,
        "f_a": n_a / N,
        "f_b": n_b / N,
        "f_stay": n_stay / N,
        "payoff_a": payoff_a,
        "payoff_b": payoff_b,
        "r_a": r_a,
        "r_b": r_b,
    }


def softmax_sample(
    utilities: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    scaled = utilities / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(utilities), p=probs))


class RandomAgent:
    type_name = "random"

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 3))

    def update(self, reward: float, observation: dict) -> None:
        pass


class MemoryAgent:
    type_name = "memory"

    def __init__(self) -> None:
        self.payoff_history: deque[tuple[float, float]] = deque(maxlen=MEMORY_WINDOW)

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        if len(self.payoff_history) < MEMORY_WINDOW:
            return int(rng.integers(0, 3))
        payoffs = np.array(self.payoff_history, dtype=float)
        u_a = payoffs[:, 0].mean()
        u_b = payoffs[:, 1].mean()
        u_s = 0.0
        return softmax_sample(np.array([u_a, u_b, u_s]), TAU_MEMORY, rng)

    def update(self, reward: float, observation: dict) -> None:
        if observation.get("round", -1) >= 0:
            self.payoff_history.append(
                (observation["payoff_a"], observation["payoff_b"])
            )


class PredictiveAgent:
    type_name = "predictive"

    def __init__(self) -> None:
        self.payoff_history: deque[tuple[float, float]] = deque(maxlen=MEMORY_WINDOW)
        self.occupancy_history: deque[tuple[float, float]] = deque(maxlen=MEMORY_WINDOW)

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        if len(self.payoff_history) < MEMORY_WINDOW:
            return int(rng.integers(0, 3))

        payoffs = np.array(self.payoff_history, dtype=float)
        mean_payoff_a = payoffs[:, 0].mean()
        mean_payoff_b = payoffs[:, 1].mean()

        occ = np.array(self.occupancy_history, dtype=float)
        if len(occ) >= 2:
            mean_f_a = occ[:, 0].mean()
            mean_f_b = occ[:, 1].mean()
            delta_f_a = occ[-1, 0] - occ[-2, 0]
            delta_f_b = occ[-1, 1] - occ[-2, 1]
        elif len(occ) == 1:
            mean_f_a, mean_f_b = occ[-1]
            delta_f_a = delta_f_b = 0.0
        else:
            mean_f_a = observation.get("f_a", 0.0)
            mean_f_b = observation.get("f_b", 0.0)
            delta_f_a = delta_f_b = 0.0

        payoff_edge_a = max(0.0, mean_payoff_a - mean_payoff_b)
        payoff_edge_b = max(0.0, mean_payoff_b - mean_payoff_a)
        flock_a = mean_f_a + FLOCK_SENSITIVITY * payoff_edge_a
        flock_b = mean_f_b + FLOCK_SENSITIVITY * payoff_edge_b
        # u_a = mean_payoff_a - CROWDING_ALPHA * flock_a - TREND_BETA * max(0.0, delta_f_a)
        # u_b = mean_payoff_b - CROWDING_ALPHA * flock_b - TREND_BETA * max(0.0, delta_f_b)
        # u_s = 0.0
        crowding_a = 1.0 + CROWDING_ALPHA * flock_a + TREND_BETA * max(0.0, delta_f_a)
        crowding_b = 1.0 + CROWDING_ALPHA * flock_b + TREND_BETA * max(0.0, delta_f_b)

        u_a = mean_payoff_a / crowding_a
        u_b = mean_payoff_b / crowding_b
        u_s = 0.0
        return softmax_sample(np.array([u_a, u_b, u_s]), TAU_PREDICTIVE, rng)

    def update(self, reward: float, observation: dict) -> None:
        if observation.get("round", -1) >= 0:
            self.payoff_history.append(
                (observation["payoff_a"], observation["payoff_b"])
            )
            self.occupancy_history.append(
                (observation["f_a"], observation["f_b"])
            )


AGENT_CLASSES = {
    "random": RandomAgent,
    "memory": MemoryAgent,
    "predictive": PredictiveAgent,
}


def build_population(condition: Condition, rng: np.random.Generator):
    type_labels = (
        ["random"] * condition.n_random
        + ["memory"] * condition.n_memory
        + ["predictive"] * condition.n_predictive
    )
    agents = [AGENT_CLASSES[label]() for label in type_labels]
    types = np.array(type_labels, dtype=object)
    perm = rng.permutation(N)
    agents = [agents[i] for i in perm]
    types = types[perm]
    return agents, types


def run_one_simulation(
    condition: Condition,
    seed: int,
    t_rounds: int = T_ROUNDS,
    rng: np.random.Generator | None = None,
    zone_pool_fn=draw_zone_pools,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if rng is None:
        rng = make_rng(seed)

    agents, agent_types = build_population(condition, rng)
    wealth = np.zeros(N, dtype=float)
    observation = empty_observation()
    records = []

    for t in range(t_rounds):
        actions = np.array([a.choose(observation, rng) for a in agents], dtype=int)
        r_a, r_b, _ = zone_pool_fn(rng)
        rewards, n_a, n_b, n_stay = allocate_rewards(actions, r_a, r_b)
        wealth += rewards
        round_obs = build_observation(t, n_a, n_b, n_stay, r_a, r_b)
        for agent, reward in zip(agents, rewards):
            agent.update(reward, round_obs)

        sync = max(n_a, n_b, n_stay) / N
        records.append(
            {
                "round": t,
                "condition_id": condition.condition_id,
                "pct_predictive": condition.pct_predictive,
                "f_a": n_a / N,
                "f_b": n_b / N,
                "f_stay": n_stay / N,
                "mean_reward": float(rewards.mean()),
                "majority_action_proportion": sync,
                "n_a": n_a,
                "n_b": n_b,
                "n_stay": n_stay,
            }
        )
        observation = round_obs

    return pd.DataFrame(records), wealth, agent_types


def gini_coefficient(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n)


def compute_seed_metrics(
    timeseries: pd.DataFrame,
    wealth: np.ndarray,
    agent_types: np.ndarray,
    condition: Condition,
    seed: int,
    t_rounds: int,
) -> dict:
    avg_reward_population = float(timeseries["mean_reward"].mean())
    mean_f_a = float(timeseries["f_a"].mean())
    mean_f_b = float(timeseries["f_b"].mean())
    mean_f_stay = float(timeseries["f_stay"].mean())
    mean_sync = float(timeseries["majority_action_proportion"].mean())

    metrics = {
        "condition_id": condition.condition_id,
        "seed": seed,
        "pct_predictive": condition.pct_predictive,
        "mean_population_reward": avg_reward_population,
        "mean_prop_zone_a": mean_f_a,
        "mean_prop_zone_b": mean_f_b,
        "mean_prop_stay_out": mean_f_stay,
        "synchronization": mean_sync,
        "gini_wealth": gini_coefficient(wealth),
        "wealth_std": float(np.std(wealth)),
    }

    for agent_type in ("random", "memory", "predictive"):
        mask = agent_types == agent_type
        count = int(mask.sum())
        if count == 0:
            metrics[f"mean_reward_{agent_type}"] = np.nan
            metrics[f"total_wealth_{agent_type}"] = 0.0
        else:
            type_wealth = wealth[mask]
            metrics[f"total_wealth_{agent_type}"] = float(type_wealth.sum())
            metrics[f"mean_reward_{agent_type}"] = float(
                type_wealth.sum() / (count * t_rounds)
            )

    return metrics


def aggregate_condition_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in seed_summary.columns
        if c not in ("condition_id", "seed", "pct_predictive")
        and pd.api.types.is_numeric_dtype(seed_summary[c])
    ]
    grouped = seed_summary.groupby("condition_id")
    mean_df = grouped[numeric_cols].mean().add_suffix("_mean")
    std_df = grouped[numeric_cols].std().add_suffix("_std")
    pct = seed_summary.groupby("condition_id")["pct_predictive"].first()
    out = pd.concat([pct, mean_df, std_df], axis=1).reset_index()
    return out


def save_figures(seed_summary: pd.DataFrame, time_series_example: pd.DataFrame) -> None:
    pct_vals = sorted(seed_summary["pct_predictive"].unique())
    grouped = seed_summary.groupby("pct_predictive")

    # Figure 1: reward by agent type vs predictive proportion
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label, marker in [
        ("mean_reward_random", "Random", "s"),
        ("mean_reward_memory", "Memory", "^"),
        ("mean_reward_predictive", "Predictive", "D"),
    ]:
        xs, means, stds = [], [], []
        for p in pct_vals:
            sub = grouped.get_group(p)
            if col == "mean_reward_predictive" and p == 0.0:
                continue
            if col == "mean_reward_memory" and p == 0.9:
                continue
            val = sub[col]
            if val.notna().sum() == 0:
                continue
            xs.append(p)
            means.append(val.mean())
            stds.append(val.std())
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker=marker, label=label, capsize=3)
    ax.set_xlabel("Predictive agent proportion")
    ax.set_ylabel("Average reward per round (by agent type)")
    ax.set_title("Average Reward by Agent Type vs Predictive Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "average_reward_by_condition.png", dpi=150)
    plt.close(fig)

    # Figure 2: inequality
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [grouped.get_group(p)["gini_wealth"].mean() for p in pct_vals]
    stds = [grouped.get_group(p)["gini_wealth"].std() for p in pct_vals]
    ax.errorbar(pct_vals, means, yerr=stds, marker="o", capsize=3, color="C3")
    ax.set_xlabel("Predictive agent proportion")
    ax.set_ylabel("Gini coefficient (cumulative wealth)")
    ax.set_title("Reward Inequality vs Predictive Proportion")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "reward_inequality_by_condition.png", dpi=150)
    plt.close(fig)

    # Figure 3: choice proportions over time (seed 0 per condition)
    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(10, 12), sharex=True)
    if len(CONDITIONS) == 1:
        axes = [axes]
    for ax, condition in zip(axes, CONDITIONS):
        ts = time_series_example[
            time_series_example["condition_id"] == condition.condition_id
        ]
        ax.plot(ts["round"], ts["f_a"], label="Zone A")
        ax.plot(ts["round"], ts["f_b"], label="Zone B")
        ax.plot(ts["round"], ts["f_stay"], label="Stay Out")
        ax.set_ylabel("Proportion")
        ax.set_title(
            f"Condition {condition.condition_id} "
            f"(predictive={condition.pct_predictive:.0%}, seed=0)"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "choice_proportions_over_time.png", dpi=150)
    plt.close(fig)

    # Figure 4: synchronization
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [grouped.get_group(p)["synchronization"].mean() for p in pct_vals]
    stds = [grouped.get_group(p)["synchronization"].std() for p in pct_vals]
    ax.errorbar(pct_vals, means, yerr=stds, marker="o", capsize=3, color="C2")
    ax.set_xlabel("Predictive agent proportion")
    ax.set_ylabel("Mean majority-action proportion")
    ax.set_title("Synchronization vs Predictive Proportion")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "synchronization_by_condition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    example_frames = []

    for condition in CONDITIONS:
        for seed in range(N_SEEDS):
            rng = make_rng(run_seed(condition.condition_id, seed))
            ts, wealth, types = run_one_simulation(condition, seed, T_ROUNDS, rng)
            seed_rows.append(
                compute_seed_metrics(ts, wealth, types, condition, seed, T_ROUNDS)
            )
            if seed == 0:
                example_frames.append(ts)

    seed_summary = pd.DataFrame(seed_rows)
    condition_summary = aggregate_condition_summary(seed_summary)
    time_series_example = pd.concat(example_frames, ignore_index=True)
    export_ts = time_series_example[
        [
            "condition_id",
            "pct_predictive",
            "round",
            "f_a",
            "f_b",
            "f_stay",
            "mean_reward",
            "majority_action_proportion",
        ]
    ].rename(
        columns={
            "f_a": "prop_zone_a",
            "f_b": "prop_zone_b",
            "f_stay": "prop_stay_out",
        }
    )

    seed_summary.to_csv(RESULTS_DIR / "seed_summary.csv", index=False)
    condition_summary.to_csv(RESULTS_DIR / "condition_summary.csv", index=False)
    export_ts.to_csv(RESULTS_DIR / "time_series_example.csv", index=False)
    save_figures(seed_summary, time_series_example)

    print("ABM resource allocation experiment complete")
    print(f"Runs: {len(CONDITIONS)} conditions x {N_SEEDS} seeds = {len(seed_summary)}")
    for _, row in condition_summary.iterrows():
        cid = int(row["condition_id"])
        pct = row["pct_predictive"]
        pop = row.get("mean_population_reward_mean", np.nan)
        gini = row.get("gini_wealth_mean", np.nan)
        sync = row.get("synchronization_mean", np.nan)
        print(
            f"  C{cid} predictive={pct:.0%}: "
            f"mean reward={pop:.4f}, Gini={gini:.4f}, sync={sync:.4f}"
        )
    print(f"Outputs written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
