# DM_ReinforcementLearning
This repository contains code to train multi-agent environments for decision making via reinforcement learning. 

# Common algorithms
The following algorithms are well-known in the reinforcement-learning community and serves as a baseline for benchmarking our research algorithms:

| Algorithms | Multi-agent? | Past simple env? | Viewer? | Multi-env? | Obs norm | Rew norm | LSTM/RNN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | ☑ | ☑ | ☑ | ☑ | ☑ | ☐ | ☑ |
| DDPG | ☑ | ☑<sup>1</sup> | ☑ | ☐ | ☐ | ☐ | ☐ |
| TD3 | ☑ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| SAC | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

<sup>1</sup> Gaussian noise diverges but Ornstein Uhlenbeck converges.

# Research algorithms
The following algorithms are specifically tailored to our tasks:

'''TBD'''

# Common algorithms details

| Algorithm | Clipping | KL-divergence |
| --- | --- | --- |
| PPO | ☑ | ☐ |

| Algorithm | Gaussian noise | Ornstein Uhlenbeck noise | Adaptive noise | Pop-art |
| --- | --- | --- | --- | --- |
| DDPG | ☑ | ☑ | ☐ | ☐ |

# Core dependencies
1. [Tensorflow](https://www.tensorflow.org/)
2. [Openai-baseline](https://github.com/openai/baselines)
