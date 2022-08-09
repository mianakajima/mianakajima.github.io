---
layout: page
title: A Centralised Soft Actor Critic Deep Reinforcement Learning Approach to District Demand Side Management Through Citylearn
author: Kathirgamanathan et al.
---

Authors: Anjukan Kathirgamanathan, Kacper Twardowski, Eleni Mangina, Donal Finn

Link to Paper: [https://arxiv.org/abs/2009.10562](https://arxiv.org/abs/2009.10562)

## 1 Minute Summary:

The authors used the soft actor critic deep reinforcement learning approach to reduce and smooth (decrease change from 1 hour to the next) aggregated electric demand of a district with residential and commercial buildings.

The soft actor critic is a reinforcement learning method - it is a model-free off-policy reinforcement learning method that tries to maximize reward by acting as randomly as possible. Off-policy refers to the fact that new policies are not generated solely off of data produced by the prior policy.

They used state space variables like:

- Month
- Day
- Hour
- Outdoor temperature
- Direct solar radiation
- Non-shiftable electricity load
- Solar generation
- State of charge of cooling storage - I think this refers to whether they are able to be discharged
- State of charge of DHW storage - I think this refers to whether they are able to be discharged

They used a reward function that included penalizing peak consumption and a manual function that rewarded charging at night and discharging during the day.

The authors compared this to a manually tuned rule-based controller (RBC) which charged cooling (and DHW if available) during the night and discharged during the day based on the hour of the day. The authors saw about a ~10% improvement over this baseline using their approach when evaluated over a multi-objective cost function over the peak electricity demand, average daily electricity peak demand, ramping, load factor, and net electricity consumption of the district.

It seems like RL for DSM is an up and coming area and is thought to be promising since buildings have many complex interactive elements that would be hard to model using just a physics-based approach. Also, buildings are usually pretty different from each other and so RL is seen to be able to perhaps better adapt to different groups of buildings than a manually tuned approach like RBC.

## Interesting Takeaways or Questions:

1. What does a 10% improvement help? Who benefits from this?
    1. From EIA: Demand-side management programs aim to lower electricity demand, **which in turn avoids the cost of building new generators and transmission lines, saves customers money, and lowers pollution from electric generators.** Utilities often implement these programs to comply with state government policies. [https://www.eia.gov/todayinenergy/detail.php?id=38872#:~:text=Demand-side management programs aim,comply with state government policies](https://www.eia.gov/todayinenergy/detail.php?id=38872#:~:text=Demand%2Dside%20management%20programs%20aim,comply%20with%20state%20government%20policies).
2. The authors mentioned that they would like to find a non-manual reward function so this approach would be more generalizable - I wonder what that would take to make? 🤔

### Mentioned Papers Interested in Reading:

José R. Vázquez-Canteli and Zoltán Nagy. 2019. Reinforcement learning for demand response: A review of algorithms and modeling techniques. Applied Energy 235, November 2018 (2019), 1072–1089. [https://doi.org/10.1016/j.apenergy.2018.11.002](https://doi.org/10.1016/j.apenergy.2018.11.002)
