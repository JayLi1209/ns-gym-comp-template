from gymnasium.envs.registration import register

register(id="ExampleNSFrozenLake-v0",
         entry_point="AAMAS_Comp.example_envs.nsFrozenlake:make_env",
         disable_env_checker=True,
         )


register(id="ExampleNSCartPole-v0",
         entry_point="AAMAS_Comp.example_envs.nsCartpole:make_env",
         disable_env_checker=True,
         )


register(id="ExampleNSAnt-v0",
         entry_point="AAMAS_Comp.example_envs.nsAnt:make_env",
         disable_env_checker=True,
         )

del register