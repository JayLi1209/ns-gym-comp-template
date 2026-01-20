import gymnasium as gym
from ns_gym.wrappers import MujocoWrapper
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import StepWiseUpdate
import numpy as np
import mujoco


def make_env(**kwargs):

    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    env = gym.make("Ant-v5", render_mode="human").unwrapped
    
    # Define a real update function to make the Ant "floatier" over time
    scheduler = ContinuousScheduler(start=10, end=1000)
    
    # The step size will reduce gravity's pull each step
    updateFn = StepWiseUpdate(
        scheduler, [np.array([0, 0, -9.8]), np.array([0, 0, -1000.0])]
    )

    tunable_params = {"gravity": updateFn}


    ns_env = MujocoWrapper(env,
                           tunable_params,
                           change_notification=change_notification,
                           delta_change_notification=delta_change_notification)
    
    return ns_env


if __name__ == "__main__":
    ns_env = make_env()