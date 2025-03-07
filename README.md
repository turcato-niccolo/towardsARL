# <a href=https://turcato-niccolo.github.io/papers/towardsARL/index.html>Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models </a>

Recent advancements in Large Language Models (LLMs) and Visual Language
Models (VLMs) have significantly impacted robotics, enabling high-level
semantic motion planning applications. Reinforcement Learning (RL), a
complementary paradigm, enables agents to autonomously optimize complex
behaviors through interaction and reward signals. However, designing
effective reward functions for RL remains challenging, especially in
real-world tasks where sparse rewards are insufficient and dense
rewards require elaborate design.
In this work, we propose Autonomous Reinforcement learning for Complex
Human-Informed Environments (ARCHIE), an unsupervised pipeline
leveraging GPT-4, a pre-trained LLM, to generate reward functions
directly from natural language task descriptions. 
The rewards are used to train RL agents in simulated environments,
where we formalize the reward generation process to enhance
feasibility. Additionally, GPT-4 automates the coding of task success
criteria, creating a fully automated, one-shot procedure for
translating human-readable text into deployable robot skills. Our
approach is validated through extensive simulated experiments on
single-arm and bi-manual manipulation tasks using an ABB YuMi
collaborative robot, highlighting its practicality and effectiveness.
Tasks are demonstrated on the real robot setup.

#  Using the code
## Numerical example of rewards:
```
bash run_env2d.sh
```
## 2D Environments benchmarks, training with ARCHIE's generated reward: 
```
bash run_2d_envs_training.sh
```
## Robotics tasks, training with ARCHIE's generated rewards:
```
python learn_task_yumi.py --num_cuda_devices {} --yumi_mode {} --seeds {} --env {} --num_independent_runs {}
```
where 
- `num_cuda_devices` is the number of cuda devices available to distribute seeds (assumes first available is cuda:0)
- `yumi_mode` is `left` for single arm tasks and `bi_manual` for dual arm tasks.
- `seeds` is the number of parallel environments (each with different random seed), in the paper it is the parameter N
- `env` is one of the envs in `environments_experiments.py`
- `num_independent_runs` number of independent runs to execute in parallel, 
  - For example you can run N=2 parallel environments for each different run, with the same parent process.
  - `-seeds 2 -num_independent_runs 3` will run 6 seeds, seeds 0 and 1 will be used by SAC agent 0, 2 and 3 by agent 1, 4 and 5 by agent 2
  - default is `-seeds 4 -num_independent_runs 3`

## Run the reward generation
You need to configure the API keys, get one and insert it into this environment variable:
```
export OPENAI_API_KEY=key
```
Then set the deployment name and endpoint:
```
export OPENAI_DEPLOYMENT_NAME=name
export OPENAI_ENDPOINT=endpoint
```


# Requirements
- Tested with Python 3.10.8
- Packages:
```
argparse
copy
importlib
random
yaml
setproctitle
numpy
torch
os
pickle
redis
tqdm
bisect
pathlib
math
openai
re
pyyaml
pybullet
pybullet_data
time
```