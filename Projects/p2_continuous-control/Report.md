[image1]: ./readme_imgs/1_agent_plot.png "1Agent_Plot"
[image2]: ./readme_imgs/20_agents_plot.png "20Agents_Plot"
[image3]: ./readme_imgs/surrogate_function.png
[image4]: ./readme_imgs/importance_sampling.png
[image5]: ./readme_imgs/REINFORCE.png
[image6]: ./readme_imgs/re-weighting_factor.png
[image7]: ./readme_imgs/PPO_summary.png

# Report of P2: Continuous Control

## Learning Algorithm

###REINFORCE
![][image5]

###[PPO](https://arxiv.org/pdf/1707.06347.pdf)(Proximal Policy Optimization Algorithms)

The __importance sampling__ below tells us we can use old trajectories for computing averages for new policy, as long as we add this extra re-weighting factor, that takes into account how under or over–represented each trajectory is under the new policy compared to the old one.
![][image4]

Expanding the __re-weighting factor__:
![][image6]

The approximate form of the gradient, we can think of it as the gradient of a new object, called the __surrogate function__
![][image3]
So using this new gradient, we can perform gradient ascent to update our policy -- which can be thought as directly maximize the surrogate function.

To summary the PPO algorithm:
![][image7]

###[A3C](https://arxiv.org/pdf/1602.01783.pdf)(Asynchronous Advantage Actor-critic)

###[D4PG](https://openreview.net/pdf?id=SyZipzbCb)(Distributed Distributional Deterministic Policy)

[Reference](https://github.com/ShangtongZhang/DeepRL)

## Plot of Rewards

- __version 1__ the agent receives an average reward (over 100 episodes) of at least +30, or
![1Agent_Plot][image1]

- __version 2__ the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
![20Agents_Plot][image2]

## Ideas of Future Work


