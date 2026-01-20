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


### Developing Your Agent

Implement your agent within the [src/comp/agent.py](src/comp/agent.py) and impoort all and all its dependencies into this file. Your agent will be a contained within a unfied `ModelBasedAgent` or `ModelFreeAgent` subclass. Be sure to add any additional dependencies to pyproject.toml so they are included in the Docker image. You can do this using the following command:

```bash
uv add <package-name>
```

Please see the [examples](examples) directory for example agents to get started. We provide example agents using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) RL implemenations, NS-Gym baseline implementations as well as custom implementations specifict to this competition.

### Competition Environments


## Evaluating Agents

The primary entry point for evaluating agents is the [evaluation/evaluate_agent.py](evaluation/evaluate_agent.py) script. You can run this script to evaluate your agent on the competition environments. Competition organizars will use the same script to evaluate all submitted agents. 






