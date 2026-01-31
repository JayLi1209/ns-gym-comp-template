# AAMAS 2026 Competition: Evaluating Adaptive Decision Agents under Non-Stationarity

This repository contains evaluation code, example agents, and boilerplate to get started with the [NS-Gym](https://nsgym.io) framework for the [AAMAS 2026 Competition](https://nsgym.io/aamas2026_competition.html).

## Set Up

1. Click **"Use this template"** at the top of this page to create your own repository (private or public) called `ns-gym-comp-submission`.

2. Clone your new repository and add this template as an upstream remote so you can pull future updates (new environments, examples, evaluation changes, etc.):

```bash
git clone https://github.com/{your-username}/ns-gym-comp-submission.git
cd ns-gym-comp-submission
git remote add upstream https://github.com/scope-lab-vu/ns-gym-comp-template.git
```

3. Add the competition organizers -- [nkepling](https://github.com/nkepling) and [ayanmukhopadhyay](https://github.com/ayanmukhopadhyay) -- as **collaborators** on your repository so we can clone, run, and evaluate your submission.

4. When ready to submit, [open an issue](https://github.com/scope-lab-vu/ns-gym-comp-template/issues/new) on this template repository with a link to your submission repo.

To pull template updates at any time:

```bash
git fetch upstream
git merge upstream/main --allow-unrelated-histories
```


### Virtual Environment

After 

Install [uv](https://docs.astral.sh/uv/getting-started/installation), then:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

This installs NS-Gym, Stable-Baselines3, PyTorch, Gymnasium, MuJoCo, and other dependencies. Verify with `uv pip list`.

### Docker

Build the evaluation container:

```bash
docker build -f docker/eval.Dockerfile -t ns-gym-comp .
```

All submitted agents will be evaluated using this Docker image. Ensure your agent runs correctly within it.

## Project Structure

```
evaluator.py                            # Runs submitted agent against all environments
submission.py                           # YOUR entrypoint -- wire up your agent here
src/AAMAS_Comp/
    __init__.py                         # Environment registration
    base_agent.py                       # ModelBasedAgent, ModelFreeAgent, SB3Agent
    agent.py                            # YOUR agent implementation goes here
    evaluation/
        evaluate_agent.py               # Episode runner and evaluation harness
        utils.py                        # Metric utilities
    examples/
        agents/                         # Baseline agent wrappers
            mcts_example.py             # MCTS (model-based)
            ppo_example.py              # PPO (model-free, SB3)
            sac_example.py              # SAC (model-free, SB3)
        environments/                   # Pre-configured NS environments
            nsFrozenlake.py
            nsCartpole.py
            nsAnt.py
examples/                               # Standalone train/eval scripts
    mcts_example.py
    ppo_example.py
    sac_example.py
docker/
    base.Dockerfile                     # Base image with dependencies
    eval.Dockerfile                     # Evaluation image for submissions
docker-compose.yml                      # Dev and test-submission services
```

## Developing Your Agent

Implement your agent in [src/AAMAS_Comp/agent.py](src/AAMAS_Comp/agent.py). Your agent must subclass either `ModelBasedAgent` or `ModelFreeAgent` from `base_agent.py`. Import all dependencies into this file.

**Do not modify `base_agent.py`** -- the competition evaluator uses its own copy.

### Agent Types

**ModelBasedAgent** -- receives a planning environment for lookahead search:

```python
class ModelBasedAgent(Agent):
    def get_action(self, obs: Dict, planning_env: gym.Env):
        # Use planning_env for simulation/search
        ...
```

**ModelFreeAgent** -- receives only the observation:

```python
class ModelFreeAgent(Agent):
    def get_action(self, obs: Dict):
        # Select action from observation alone
        ...
```

Both agent types are timed via `validate_and_get_action()` during evaluation. Actions are validated against the action space.

Add any new dependencies to `pyproject.toml`:

```bash
uv add <package-name>
```

## Pre-configured Environments

Three non-stationary environments are registered and ready to use. Each wraps a standard Gymnasium environment with NS-Gym schedulers and update functions that modify environment parameters over time. Environment source code is in [src/AAMAS_Comp/examples/environments/](src/AAMAS_Comp/examples/environments/).

All environments accept `change_notification` and `delta_change_notification` kwargs to control whether the agent is informed of parameter changes.

```python
import gymnasium as gym
import AAMAS_Comp  # triggers environment registration

env = gym.make("ExampleNSFrozenLake-v0", change_notification=True, delta_change_notification=True)
```

### ExampleNSFrozenLake-v0

Wraps `FrozenLake-v1` (discrete, 16 states, 4 actions). Transition probabilities `P` are decremented by 0.025 each step via `DistributionDecrementUpdate` on a `ContinuousScheduler`, making the surface progressively more slippery.

### ExampleNSCartPole-v0

Wraps `CartPole-v1` (continuous state, 2 discrete actions). Two parameters change simultaneously:
- **masspole**: increases by 0.1 each step (`IncrementUpdate`, `ContinuousScheduler`)
- **gravity**: random walk every 3 steps (`RandomWalk`, `PeriodicScheduler`)

### ExampleNSAnt-v0

Wraps `Ant-v5` (continuous state/action, MuJoCo). The `torso_mass` decays exponentially with rate 0.9 (`ExponentialDecay`, `PeriodicScheduler` with period 500, active from step 100 to 500), making the ant progressively lighter.

### Competition Evaluation Environments

Final evaluation will use the same three base environments (FrozenLake, CartPole, Ant), but the nature of non-stationarity will differ from the pre-configured examples above. The specific schedulers, update functions, tunable parameters, and their configurations will be different and are not disclosed in advance.

Participants are encouraged to construct and experiment with different non-stationarity conditions beyond the provided examples to build agents that generalize across varying forms of environmental change. See the [NS-Gym documentation](https://nsgym.io) for available schedulers, update functions, and tunable parameters.

## Examples

Standalone scripts in the [examples/](examples/) directory demonstrate training and evaluation. Corresponding agent wrappers are in [src/AAMAS_Comp/examples/agents/](src/AAMAS_Comp/examples/agents/).

### MCTS (`examples/mcts_example.py`)

Evaluates the NS-Gym MCTS implementation (with chance nodes) on `ExampleNSFrozenLake-v0`. This is a **model-based** agent that uses the planning environment for tree search. Configured with rollout depth 50, 100 iterations, UCT constant 1.4, and discount 0.99.

```bash
python examples/mcts_example.py
```

### PPO (`examples/ppo_example.py`)

Trains Stable-Baselines3 PPO on stationary `Ant-v5` with RL Zoo3 tuned hyperparameters and `VecNormalize`, then evaluates on `ExampleNSAnt-v0`. This is a **model-free** agent. VecNormalize statistics are saved alongside the model and loaded during evaluation to normalize observations.

```bash
python examples/ppo_example.py
```

### SAC (`examples/sac_example.py`)

Trains Stable-Baselines3 SAC on stationary `Ant-v5` with default hyperparameters, then evaluates on `ExampleNSAnt-v0`. This is a **model-free** agent. SAC does not require `VecNormalize`, making the pipeline simpler than PPO.

```bash
python examples/sac_example.py
```

## Evaluation

The evaluation harness is in [src/AAMAS_Comp/evaluation/evaluate_agent.py](src/AAMAS_Comp/evaluation/evaluate_agent.py). The competition uses the same harness to evaluate all submissions.

```python
from AAMAS_Comp.evaluation import run_complete_evaluation

run_complete_evaluation(
    env=ns_env,
    agent=agent,
    start_seed=42,
    num_episodes=10,
    name_prefix="MyAgent",
)
```

This runs `num_episodes` episodes with deterministic sequential seeding (`start_seed`, `start_seed + 1`, ...) and saves to `results/{name_prefix}/`:
- `{name_prefix}.zip` -- compressed per-step metrics (observations, actions, rewards, decision times, environment changes)
- `metadata.json` -- seed range, episode count, total wall time
- `summary.json` -- per-episode and aggregate statistics (mean reward, decision time, number of transition function changes)

### Docker Evaluation

Edit [submission.py](submission.py) to wire up your agent. The evaluator calls `get_agent(env_id)` once per base environment, passing the environment ID (`"FrozenLake-v1"`, `"CartPole-v1"`, or `"Ant-v5"`). Use this to load environment-specific model weights, hyperparameters, or agent classes:

```python
def get_agent(env_id: str):
    if env_id == "Ant-v5":
        model = PPO.load("models/ppo_ant/ppo_ant.zip")
        return MyModelFreeAgent(model=model)
    elif env_id == "FrozenLake-v1":
        return MyModelBasedAgent(d=50, m=100)
    elif env_id == "CartPole-v1":
        return MyModelFreeAgent()
```

Then test locally:

```bash
docker compose run test-submission
```

This runs [evaluator.py](evaluator.py) inside the container against all three competition environments. The `test-submission` service mounts your `submission.py` and `src/` so changes are picked up without rebuilding.

## Submission

1. Ensure the competition organizers ([nkepling](https://github.com/nkepling), [ayanmukhopadhyay](https://github.com/ayanmukhopadhyay)) are collaborators on your repository.
2. Verify your submission runs successfully with `docker compose run test-submission`.
3. [Open an issue](https://github.com/scope-lab-vu/ns-gym-comp-template/issues/new) on this template repository with a link to your submission repo.


