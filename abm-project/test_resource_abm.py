"""Verification tests for the resource ABM."""

import unittest

import numpy as np

import run_simulation as rs


class TestResourceABM(unittest.TestCase):
    TOLERANCE = 1e-9

    def test_population_and_rewards_valid(self):
        condition = rs.CONDITIONS[0]
        rng = rs.make_rng(42)
        agents, _ = rs.build_population(condition, rng)
        observation = rs.empty_observation()
        wealth = np.zeros(rs.N, dtype=float)

        for t in range(30):
            actions = np.array([a.choose(observation, rng) for a in agents], dtype=int)
            r_a, r_b, _ = rs.draw_zone_pools(rng)
            rewards, n_a, n_b, n_stay = rs.allocate_rewards(actions, r_a, r_b)

            self.assertEqual(n_a + n_b + n_stay, rs.N)
            self.assertGreaterEqual(n_a, 0)
            self.assertGreaterEqual(n_b, 0)
            self.assertGreaterEqual(n_stay, 0)
            self.assertTrue(np.all(np.isfinite(rewards)))

            wealth += rewards
            self.assertTrue(np.all(np.isfinite(wealth)))

            obs = rs.build_observation(t, n_a, n_b, n_stay, r_a, r_b)
            for agent, reward in zip(agents, rewards):
                agent.update(reward, obs)
            observation = obs

    def test_zone_reward_math(self):
        actions = np.array([rs.ZONE_A] * 100 + [rs.ZONE_B] * 50 + [rs.STAY_OUT] * 150)
        r_a, r_b = 95.0, 70.0
        rewards, n_a, n_b, n_stay = rs.allocate_rewards(actions, r_a, r_b)

        self.assertAlmostEqual(rewards[actions == rs.ZONE_A].sum(), r_a, places=9)
        self.assertAlmostEqual(rewards[actions == rs.ZONE_B].sum(), r_b, places=9)
        self.assertTrue(np.all(rewards[actions == rs.STAY_OUT] == 0.0))

        empty_a = np.array([rs.ZONE_B] * rs.N)
        rewards_b, n_a2, _, _ = rs.allocate_rewards(empty_a, r_a, r_b)
        self.assertEqual(n_a2, 0)
        self.assertAlmostEqual(rewards_b.sum(), r_b, places=9)

    def test_reproducibility(self):
        condition = rs.CONDITIONS[2]
        seed = 123
        ts1, w1, _ = rs.run_one_simulation(
            condition, seed, 50, rs.make_rng(rs.run_seed(condition.condition_id, seed))
        )
        ts2, w2, _ = rs.run_one_simulation(
            condition, seed, 50, rs.make_rng(rs.run_seed(condition.condition_id, seed))
        )
        self.assertTrue(np.allclose(ts1["f_a"].values, ts2["f_a"].values))
        self.assertTrue(np.allclose(w1, w2))

    def test_dominant_resource_all_memory(self):
        def fixed_pools(rng):
            return 100.0, 0.0, 0.0

        condition = rs.Condition(99, 0, rs.N, 0)
        ts, _, _ = rs.run_one_simulation(
            condition, 99, 120, rs.make_rng(99), zone_pool_fn=fixed_pools
        )
        late = ts.iloc[50:]
        mean_f_a = late["f_a"].mean()
        mean_f_b = late["f_b"].mean()
        self.assertGreater(mean_f_a, 0.5)
        self.assertGreater(mean_f_a, mean_f_b + 0.15)

    def test_memory_beats_random_in_c1(self):
        """Memory cohort earns more total wealth than the random cohort in C1."""
        condition = rs.CONDITIONS[0]
        mem_totals = []
        rand_totals = []
        for seed in range(5):
            ts, wealth, types = rs.run_one_simulation(
                condition,
                seed,
                100,
                rs.make_rng(rs.run_seed(condition.condition_id, seed)),
            )
            metrics = rs.compute_seed_metrics(
                ts, wealth, types, condition, seed, 100
            )
            mem_totals.append(metrics["total_wealth_memory"])
            rand_totals.append(metrics["total_wealth_random"])
        self.assertGreater(np.mean(mem_totals), np.mean(rand_totals))


if __name__ == "__main__":
    unittest.main()
