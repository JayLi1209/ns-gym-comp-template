from gymnasium.envs.registration import register

register(id="ExampleNSFrozenLake-v0",
         entry_point="AAMAS_Comp.example_envs.nsFrozenlake:make_env",
         disable_env_checker=True,
         )


del register