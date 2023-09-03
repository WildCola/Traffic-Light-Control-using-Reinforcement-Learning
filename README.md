# Traffic-Light-Control-using-Reinforcement-Learning

This repository contains three reinforcement learning agents made for traffic flow optimization that are compatible with the RESCO benchmark, as well as the modified agent_config and main files required to run these agents.

## Instructions:
- download the RESCO benchmark form: https://github.com/Pi-Star-Lab/RESCO/tree/main/resco_benchmark
- add the agents to the agents folder in the benchmark
- replace the agent_config.py and main.py files from the benchmark with the ones from this repo
- run the agent of your choice (example: python main.py --agent IDDQN --map cologne1)
- if you want to plot your results you can add custom_graph.py to your utils folder

## Additional requirements:
- SUMO: https://eclipse.dev/sumo/
- SUMO-RL: https://github.com/LucasAlegre/sumo-rl/tree/88ede9bafe06333b9837dc9eb996befd4f482085
