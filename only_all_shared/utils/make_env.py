"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multi_both.env import Community
    from pathlib import Path

    # create multiagent environment
    if benchmark:
        climate_zone = 7
        data_path = Path("multi_both/data/Climate_Zone_"+str(climate_zone))
        params = {'data_path':data_path, 
                'building_attributes':'building_attributes.json', 
                'weather_file':'weather_data.csv', 
                'building_ids':["Building_"+str(i) for i in [1,2,3]],
                'buildings_states_actions':'multi_both/buildings_state_action_space.json', 
                'simulation_period': (0, int(8760)), 
                'deterministic': False, 
                'save_memory': False }
        env = Community(**params)
    else:
        climate_zone = 7
        data_path = Path("multi_both/data/Climate_Zone_"+str(climate_zone))
        params = {'data_path':data_path, 
                'building_attributes':'building_attributes.json', 
                'weather_file':'weather_data.csv', 
                'building_ids':["Building_"+str(i) for i in [1,2,3]],
                'buildings_states_actions':'multi_both/buildings_state_action_space.json', 
                'simulation_period': (0, int(8760)), 
                'deterministic': False, 
                'save_memory': False }
        '''
        climate_zone = 7
        data_path = Path("multiagent/data/Climate_Zone_"+str(climate_zone))
        params = {'data_path':data_path, 
                'building_attributes':'building_attributes.json', 
                'weather_file':'weather_data.csv', 
                'building_ids':["Building_"+str(i) for i in [1,2,3]],
                'buildings_states_actions':'multiagent/buildings_state_action_space.json', 
                'simulation_period': (0, int(49)), 
                'deterministic': False, 
                'save_memory': False }
        '''
        env = Community(**params)
    return env
