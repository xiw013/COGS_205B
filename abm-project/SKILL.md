Project Goal

Build a simple ABM of a competitive resource environment where agents choose between Zone A, Zone B, or Stay Out. The main hypothesis is that higher-order predictive agents may perform well when rare, but lose their advantage when they become common because their anti-crowding strategies synchronize.
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

Modeling Rules

- Keep the model simple and interpretable.
- Do not turn this into a realistic financial market simulation.
- Agent choices should be probabilistic, not fully deterministic.
- Macro-level patterns should emerge from local agent rules, not from hard-coded global behavior.
- Use three agent types: RandomAgent, MemoryAgent, and PredictiveAgent.
- Main manipulation: vary the proportion of PredictiveAgents while keeping RandomAgents fixed.

Technical Constraints

Use Python, numpy, pandas, and matplotlib.

Do NOT use unnecessary frameworks.

The project should run with:

```bash
python run_simulation.py