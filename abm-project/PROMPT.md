I want to build a Python agent-based model (ABM) of competitive resource allocation. Please read this entire brief before producing an implementation plan. Do not write code yet. 

# Scientific goal
Demonstrate how different decision-making strategies affect collective bahavior in a competitive environment. 

Specifically: 
- Agents repeatedly choose between two resource zones or staying out. 
- Rewards depend both on resource availability and how many agents make the same choice. 
- Some agents react only to past rewards, while other attempt to anticipate future crowding. 

The main scientific question is: How does the proportion of predictive agents affect collective system stability and average agent reward?

The model should make this relationship easy to study systematically. 

# Model Overview
The environment contains three possible actions each round: 
1. Zone A
2. Zone B
3. Stay out
Zones contain limited reward pools that are shared equally among agents selecting that zone. 

Default parameters: 
Parameter
Zone A reward pool ~ Uniform(90, 100)
Zone B reward pool ~ Uniform (60, 80) 
Stay Out reward = 0
Population size = 300
Memory = 3 rounds

If a zone has (n) participants, 
reward_per_agent = zone_reward / n
Stay Out always gives reward 0.

# Agent Types
The population contains three types of agents. 

## RandomeAgent
chooses Zone A, Zone B, or Stay Out uniformly at random. 

## MemoryAgent
Tracks average reward from Zones A and B over a rolling memory window of three rounds. 
Memory agent tend to choose the option with the highest recent average payoff.

## Predictive Agent
Uses recent population statistic to anticipate future crowding and factor this into the recent average payoff of two zones. 

Rather than simply following recent rewards, preditive agents estimate how other agents are likely to respond to observed popuilation behavior and adjust their decisions accordingly. 

Predictive agents should only use observed population information and should not directly access other agents' internal states. 

IMPORTANT: Neither MemoryAgents nor PredictiveAgents should behave deterministically, their deicisions should remain probabilistic, so there is still change (even though small) to choose other actions. 

# Main experiment
The main independent variable is the proportion of PredictiveAgents in the population.

conditions:
Condition 1: 10% Random, 90% Memory, 0% Predictive
Condition 2: 10% Random, 60% Memory, 30% Predictive
Condition 3: 10% Random, 30% Memory, 60% Predictive
Condition 4: 10% Random, 0% Memory, 90% Predictive
The RandomAgent proportion should remain fixed across conditions.

For each proportion, the simulation should be repeated across multiple random seeds.

Keep the project intentionally small and interpretable.

The project should contain only:

resource_abm/
    run_simulation.py
    test_resource_abm.py
    results/

All model classes should be defined inside run_simulation.py.

The test_resource_abm.py file should contain all verification checks.

All generated outputs should be saved inside the results/ folder.

Necessary outputs only

The analysis should focus on two main questions:

Are rewards balanced across agents, or concentrated among only some agents?
Do agents herd into the same option over time, causing fluctuations or oscillations in Zone A, Zone B, and Stay Out choices?
Required tables

The simulation should save CSV tables for the important summaries:

condition_summary.csv
One row per experimental condition, including:
predictive-agent proportion
mean population reward
mean reward by agent type
total wealth by agent type
reward inequality, such as Gini coefficient or standard deviation of cumulative wealth
average proportion choosing Zone A
average proportion choosing Zone B
average proportion choosing Stay Out
synchronization, measured as the average majority-action proportion
seed_summary.csv
One row per condition and random seed, so the results are reproducible and not based on only one run.
time_series_example.csv
Time-series output for one representative seed per condition, including:
round
proportion choosing Zone A
proportion choosing Zone B
proportion choosing Stay Out
mean reward
majority-action proportion
Required figures only

Generate only the necessary figures:

average_reward_by_condition.png
Shows how average reward changes as the proportion of PredictiveAgents increases.
reward_inequality_by_condition.png
Shows whether rewards are balanced across agents or concentrated among a small group.
choice_proportions_over_time.png
Shows the proportion of agents choosing Zone A, Zone B, and Stay Out across time for representative runs.
synchronization_by_condition.png
Shows whether the population tends to herd into the same action.

Do not generate extra figures unless they directly answer one of the two main questions.

Verification checks

The test file should include only necessary validity checks:

Total population always equals N.
Zone counts are never negative.
No NaN or infinite rewards or wealth values.
Same seed and same parameters produce identical results.
Reward distribution is mathematically correct:
total reward distributed in Zone A equals the Zone A reward pool if Zone A has at least one agent;
total reward distributed in Zone B equals the Zone B reward pool if Zone B has at least one agent;
Stay Out gives zero reward.
In the single-dominant-resource edge case, where Zone B and Stay Out give zero reward, agents should mostly favor Zone A after learning.
Run behavior

The project should run with:

python run_simulation.py

Running this script should:

Create the results/ folder if it does not exist.
Run all experimental conditions across multiple seeds.
Save the summary tables.
Save only the required figures.
Print a short summary to the terminal.