from ns_gym.wrappers import NSFrozenLakeWrapper
from ns_gym.update_functions import DistributionDecrementUpdate
from ns_gym.schedulers import ContinuousScheduler
import gymnasium as gym


def make_env():
    base_env = gym.make("FrozenLake-v1").unwrapped
    scheduler = ContinuousScheduler()

    k = 0.025
    update_fn  = DistributionDecrementUpdate(scheduler=scheduler, k=k)

    tunable_params = {"P": update_fn}

    ns_env = NSFrozenLakeWrapper(base_env,
                                tunable_params=tunable_params,
                                change_notification=True,
                                delta_change_notification=True,
                                initial_prob_dist=[1,0,0])
    
    return ns_env
    

if __name__ == "__main__":
    env = make_env()

    