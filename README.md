# Code Template for AAMAS 2026 Competition: Evaluating Adaptive Decision Agents under Non-Stationarity 

This repository contains instructions, evaluation code, and other boilerplate code to get started using NS-Gym for the AAMAS 2026 Competition: Evaluating Adaptive Decision Agents under Non-Stationarity. Please refer to the [competition website](https://nsgym.io/aamas2026_competition.html) for more details about the competition. This competition is entirly based on the [NS-Gym](https://nsgym.io) framework for evaluating adaptive decision agents under non-stationarity.

## Set Up

Fork this repository to your own GitHub account and clone it to your local machine. You can either setpup a virtual environment or use Docker to manage dependencies.

### Virtual Environment

We use `uv` to manage our virtual environment. Visit [this link to install uv](https://docs.astral.sh/uv/getting-started/installation). To set up the environment, run the following commands:

1. **Fork** the repository on GitHub and if you like clone it as a private repository. 

2. Clone the repository to your local machine (be sure to update your GitHub username), i.e

```bash
git clone https://github.com/{your-username}/ns-gym-comp-template.git
```

3. Navigate to the project directory, make a new virtual environment, and activate it. **We are using Python 3.13** for this competition.

```bash
cd ns-gym-comp-template
uv venv --python 3.13
uv .venv/bin/activate
```
4. Install this competition template as a package in editable mode

```bash
uv pip install -e .
```

Installing the competition package will pip install NS-Gym, Stable-Baselines3, PyTorch, Numpy, Pandas, Mujoco, Tensorboard and other dependencies.

To view all installed packages, run:

```bash
uv pip list
```



### Docker
Alternatively, you can use Docker to set up the environment. Make sure you have Docker installed on your machine. Then, build the Docker image using the provided Dockerfile:

```bash
docker build -t ns-gym-comp .
```

**We will be evaluating all agent using this Docker image so please ensure your agent runs correctly within this environment.**



## Accessing Competition Template Code

The competition code is organized as a Python package named `AAMAS_Comp`. You can import this package in your Python scripts or Jupyter notebooks to access the competition environments and evaluation code. For example:

```python
import AAMAS_Comp
``` 


## Developing Your Agent

Implement your agent within the [src/AAMAS_Comp/agent.py](src/AAMAS_Comp/agent.py) and impoort all and all its dependencies into this file. Your agent will be a contained within a unfied `ModelBasedAgent` or `ModelFreeAgent` subclass. Be sure to add any additional dependencies to pyproject.toml so they are included in the Docker image. You can do this using the following command:

```bash
uv add <package-name>
```

Please see the [examples](examples) directory for example agents to get started. We provide example agents using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) RL implemenations, NS-Gym baseline implementations as well as custom implementations specifict to this competition.

## Competition Environments

We provide three pre-configured competition environments under the [src/AAMAS_Comp/example_envs](src/AAMAS_Comp/example_envs) directory. These environments are ready to be used for training and evaluating your agent. You can also create your own custom environments by following the NS-Gym documentation. In addition to the pre-configured environments, competition organizers will provide additional hidden environments for the final evaluation.

### Pre-configured Environments

Pre-configured competition environments can be built by using the Gymnasium `make` function with the appropriate environment ID. Below are the environment IDs for the pre-configured competition environments:

#### Non-Stationary FrozenLake

```python
import gymnasium as gym 
import AAMAS_Comp


change_notification = True
delta_change_notification = True

ns_frozenlake_env = gym.make("ExampleNSFrozenLake-v0", change_notification=change_notification, delta_change_notification=delta_change_notification)
```


#### Non-Stationary CartPole



```python
import gymnasium as gym 
import AAMAS_Comp


change_notification = True
delta_change_notification = True

ns_cartpole_env = gym.make("ExampleNSCartPole-v0", change_notification=change_notification, delta_change_notification=delta_change_notification)
```

#### Non-Stationary MujoCo Ant


```python
import gymnasium as gym 
import AAMAS_Comp


change_notification = True
delta_change_notification = True

ns_ant_env = gym.make("ExampleNSAnt-v0", change_notification=change_notification, delta_change_notification=delta_change_notification)
```

## Evaluating Agents

The primary entry point for evaluating agents is the [evaluation/evaluate_agent.py](evaluation/evaluate_agent.py) script. You can run this script to evaluate your agent on the competition environments. Competition organizars will use the same script to evaluate all submitted agents. 






