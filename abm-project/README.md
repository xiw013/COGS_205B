# Final Project Conclusions and Reflections
The model is an agent-based simulation of competitive resource allocation.

## Model Specification

The model contains (N=300) agents. At each round (t), each agent (i) chooses one action:

$a_i(t) \in {A, B, S},$

where (A) is Zone A, (B) is Zone B, and (S) means Stay Out.

The reward pools for the two zones are stochastic:

$R_A(t) \sim \text{Uniform}(90,100),$

$R_B(t) \sim \text{Uniform}(60,80).$

Let $n_A(t)$, $n_B(t)$, and $n_S(t)$ be the number of agents choosing each option. The population constraint is:

$n_A(t)+n_B(t)+n_S(t)=N.$

Rewards are divided equally among agents choosing the same zone:

$$
r_i(t)=
\begin{cases}
\frac{R_A(t)}{n_A(t)}, & \text{if } a_i(t)=A \text{ and } n_A(t)>0, \\
\frac{R_B(t)}{n_B(t)}, & \text{if } a_i(t)=B \text{ and } n_B(t)>0, \\
0, & \text{if } a_i(t)=S.
\end{cases}
$$

Each agent’s cumulative wealth is updated as:

$W_i(t+1)=W_i(t)+r_i(t).$

The model includes three agent types.

### Random agents

Random agents choose uniformly among all three actions:

$P(a_i(t)=A)=P(a_i(t)=B)=P(a_i(t)=S)=\frac{1}{3}.$

### Memory agents

Memory agents track recent zone payoffs over a memory window of (m=3) rounds. Their estimated utilities are:

$U_A(t)=\frac{1}{m}\sum_{k=1}^{m} p_A(t-k),$

$U_B(t)=\frac{1}{m}\sum_{k=1}^{m} p_B(t-k),$

$U_S(t)=0,$

where $p_A(t)=\frac{R_A(t)}{n_A(t)}$ and $p_B(t)=\frac{R_B(t)}{n_B(t)}$. 

Choices are probabilistic using a softmax rule:
$P(a_i(t)=j)=\frac{\exp(U_j(t)/\tau_M)}{\sum_{\ell \in {A,B,S}}\exp(U_\ell(t)/\tau_M)}.$

### Predictive agents

Predictive agents use the same recent payoff information as MemoryAgents, but additionally assume that other agents are likely to respond to the same information. A zone that has recently attracted many agents or has recently become increasingly popular is expected to become even more crowded in the near future.

To estimate this, predictive agents track the recent proportion of the population choosing each zone:

$f_A(t)=\frac{n_A(t)}{N}, \qquad f_B(t)=\frac{n_B(t)}{N}.$

and compute the average occupancy over the previous three rounds:

$\bar f_A(t)=\frac{1}{m}\sum_{k=1}^{m} f_A(t-k),$

as well as the recent change in occupancy:

$\Delta f_A(t)=f_A(t-1)-f_A(t-2),$

and similarly for Zone B.

The average occupancy represents the baseline popularity of a zone. A zone that has consistently attracted many agents is expected to remain relatively crowded. Predictive agents also assume that high recent payoffs attract additional participants. 

They also compute a payoff advantage term:

$E_A(t)=\max(0,\bar p_A(t)-\bar p_B(t)),$

$E_B(t)=\max(0,\bar p_B(t)-\bar p_A(t)).$

The predicted flocking pressure for each zone is the combination of average occupancy and average payoffs: 

$F_A(t)=\bar f_A(t)+\lambda E_A(t),$

$F_B(t)=\bar f_B(t)+\lambda E_B(t),$

where $\lambda$ controls how strongly payoff advantage is expected to attract other agents.

However, the flocking pressure alone cannot distinguish between a stable crowd and a rapidly growing one. The trend term captures whether participation is currently increasing. A positive value of $\Delta f_A(t)$ indicates that additional agents are moving toward Zone A, suggesting that future crowd may be even greater than its recent average would imply.

Predictive agents then discount expected payoffs by predicted crowding:

$U_A(t)=\frac{\bar p_A(t)}{1+\alpha F_A(t)+\beta \max(0,\Delta f_A(t))},$

$U_B(t)=\frac{\bar p_B(t)}{1+\alpha F_B(t)+\beta \max(0,\Delta f_B(t))},$

$U_S(t)=0.$

Choices are again probabilistic using softmax:

$P(a_i(t)=j)=\frac{\exp(U_j(t)/\tau_P)}{\sum_{\ell \in {A,B,S}}\exp(U_\ell(t)/\tau_P)}.$

### Metrics

The simulation records both round-level dynamics and overall outcomes to evaluate the effects of different population compositions.

At each round (t), the average population reward is computed as:

$\bar r(t)=\frac{1}{N}\sum_{i=1}^{N} r_i(t),$

where (r_i(t)) is the reward received by agent (i) during round (t). The average reward over the entire simulation is obtained by averaging (\bar r(t)) across all rounds.

Each agent accumulates wealth over the course of the simulation,

$W_i=\sum_{t=1}^{T}r_i(t),$

where (T) is the total number of rounds. For each agent type (g) (Random, Memory, and Predictive), the model records both the total cumulative wealth:

$W_g=\sum_{i=g}W_i,$

and the average reward earned per agent per round:

$\bar r_g=\frac{1}{N_gT}\sum_{i=g}W_i,$

The proportion of the population selecting each action is tracked throughout the simulation:

$
f_A(t)=\frac{n_A(t)}{N}, \qquad
f_B(t)=\frac{n_B(t)}{N}, \qquad
f_S(t)=\frac{n_S(t)}{N},$

where (n_A(t)), (n_B(t)), and (n_S(t)) denote the numbers of agents choosing Zone A, Zone B, and Stay Out, respectively.

To quantify collective behavior, the model computes a synchronization measure:

$H(t)=\max{(f_A(t),f_B(t),f_S(t))}.$

which represents the proportion of agents choosing the most popular action at round (t). Higher values of $H(t)$ indicate stronger herding behavior, whereas lower values indicate a more balanced distribution of choices across the available actions.

To evaluate whether rewards are distributed evenly across the population or concentrated among a subset of agents, cumulative wealth inequality is measured using the Gini coefficient,

$G=
\frac{2\sum_{i=1}^{N} iW_{(i)}}{N\sum_{i=1}^{N}W_i}
-\frac{N+1}{N},$

where $W_{(i)}$ denotes cumulative wealth sorted from smallest to largest. A larger Gini coefficient indicates greater inequality in accumulated rewards.

The main experimental manipulation is the proportion of predictive agents in the population. The model tests whether increasing predictive agents reduces herding, stabilizes zone occupancy, improves average reward, and changes how evenly rewards are distributed across agents.


## Results
There are 7 conditions in the experiment:
the proportion of Predictive agent varied from 0%, to 5%, 30%, 45%, 60%, 85%, 90%, and the memeory agent proportion decrease to made up total of 90% of the population, there are always 10% of the random agent. 

### Low Proportion of Predictive Agents (Condition 1 and 2)
When the population contains few or no predictive agents, system shows strong herding. As shown in the choice_proportion_over_time plot, agent repeatedly switch almost entirely between Zone A and Zone B, producing large oscillations in resource use. Because memory agents simply follow recently successful zones, they tend to overcrowd whichever zone performed well in the previous rounds, causing its payoff to collapse and making the alternative zone attrative in the next round. This result in high synchronization as shown in the Synchronization_by_condition plot, with over 90% of the population often selecting the same action. The reward structure is also highly unequal, as reflected by large Gini coefficient in the reward_inequality_by_condition plot. In contrast, in average_reward_by_condition plot, random agent perform well since the behavior of random agent is unpredictable. They occasionally enter the zone that is less crowded, which allows them to obtain disproportionately large rewards. Also, when there is only 5% of predictive agent, they have the highest average reward per round since they are making decision by accounting for the population flow. This is result is consist with my hypothesis and predictions of expected qualitative behaviors. 

### Moderate Proportion of Predictive Agents (Condition 3, 4, and 5)
As the porportion of predictive agents increase to moderate levels (approximately 30% - 60%). The choice_proportion_over_time plot shows that the large oscillation are greatly reduced, with proportions of agents selecting Zone A and Zone B fluctuation around relatively stable values. Predictive agents account for recent crowding when making decisions, preventing the entire population from simultaneously switching toward the currently profitable zone. This stabilization is reflected in the synchronization meausre and reward inequality measure. The opportunities become more evenlly distributed across the population. Although predictive agent intially achieve relative high reward when their proportion is very low ~5%, their advantages quickly decreases as their strategy becomes more widespread. At the same time, MemoryAgents benefit from the more stable environment and their average reward steadily increase.

This still partially aligns with my prediction, but I didn't expect the advantages of predictive strategy disapeared such rapidly. Even with 30% of the predictive agent, the system is already stabled with reward evenly distributed among all agents. 

### High Proportions of Predictive Agents (Condition 6 and 7)
When predictive agents dominate the population, the system approaches a stable mixed equilibrium. The choice_proportion_over_time plot show that approximately equal proportion of agents consistenly select Zone A and B. While only a small fraction chooses to stay out. Unlike the low-predictive conditions, there are no large-sclae oscillations or persistent overcrowding. The synchronization meausre reachs its lowest values and the reward inequality remains very low. Both results indivate that the population is well balanced and rewards are distributed relatively evenly among agents. The average_reward_by_condition plot shows that the individual advantage of predictive agents disappeared, and instead memory agents obtain a sligtly higher average reward per round. Overall, increasing the proprotion of predictive agents leads to greater system stability, reduced herding, and a more equitable distribution of rewards. 

One unexpected result is that increasing the proportion of PredictiveAgents stabilizes the system rather than destabilizing it, and the higher overall proportion in zone A causing by the difference in the total rewards of two zone also diminishs as the proportion of predictive agents increases. I think this largly depends on how I define the behavior of the predictive agents, because they are accounting for recent payoffs, recent occupancy, and recent occupancy trends to discount the attractiveness of crowded zones, the probability of choosing two zones moves closer and closer, so the probability of choosing each zone becomes 50%, so the predictive agents and memory agents eventually just make decision randomly between 2 zones. 

## Reflection
Working on this project felt very uncontrolled at several points. One major challenge was figuring out how detailed my prompt should be. If I specified every detail, then it felt like I might as well write the code myself. However, if the prompt was too vague, the AI-generated behavior became much more unpredictable, and it became harder for me to evaluate whether the code was valid.

My initial prompt was not very successful. I asked the AI to generate multiple files, with each file containing a separate class. This structure would normally be neat and interpretable if I were writing the code myself. However, when the files were generated by AI, it became very difficult for me to navigate between them and understand how the whole simulation worked (maybe this is just a problem with me). When I first looked at the results, they also seemed wrong: the inequality measure, rewards, and other outcomes remained almost constant across conditions by change only ~0.03 thorughout the time. Because of this, I revised the prompt. I asked the AI to put all the main model classes inside `run_simulation.py` and to simplify the metrics so that only the most necessary outputs were collected. I then created a new folder and ran the project again.

The second version produced results that were much more interpretable, but I still found a problem. A very high proportion of the population was choosing Stay Out, which did not make sense because Stay Out always gives reward 0. I then looked into the code and found that the PredictiveAgent’s expected rewards could become negative. Since Stay Out had utility 0, this gave Stay Out too much probability weight. I changed the predictive decision rule so that crowding discounted expected rewards instead of subtracting from them directly. I also added more experimental conditions so that the transition from low to high proportions of PredictiveAgents would be easier to interpret. Since it took really time to run the code each time, I only included 7 different conditions. 

To check the accuracy of the codebase, I used several strategies. First, I reviewed the plan to identify high-level inconsistencies before generating code. After the AI generated the implementation and produced results, I checked whether the outputs made sense, especially in edge cases such as conditions with no PredictiveAgents or no MemoryAgents. Finally, I inspected the actual code. I only focused  on the agent classes, the equations determining how each agent type makes decisions, and the code for computing metrics.

Because this was my first time working on an ABM project, I had very little domain knowledge. I was not fully confident about parts of the simulation structure, such as the overall organization of the code and how the full simulation loop should be designed. For those parts, I relied heavily on the AI. This made me realize that using AI for coding projects still requires the user to have enough expertise to check whether the generated code makes sense. Otherwise, it is easy to accept code that runs but does not actually implement the intended model.

I also had trouble using Cursor at first because it was my first time working with it. Each step felt somewhat outside of my control when it asked me to approve and proceed, and it was intimidating to watch the AI create multiple files at once on my computer. I also ran out of free trials quickly, so I changed codes by myself in the end. Overall, though, it was an interesting experience. It showed me both the usefulness and the limitations of AI-assisted programming. AI can help generate code quickly, but the user still needs to guide the project carefully, simplify the structure when necessary, and repeatedly check whether the model behavior matches the intended scientific question.

I still think that this style of programming is likely to become an important part of the future software development, so it is something we will eventually need to become comfortable with. But, even after finishing the project, I have mixed feelings about how well I understand it. On one hand, I understand the scientific question, the overall structure of the model, and the main decision rules for the different types of agents. On the other hand, there are still parts of the implementation and the simulation framework that I was not fully clear and would not have been able to write from scratch without AI assistance. This leaves me feeling that I understand the project conceptually, but not completely at the implementation level.