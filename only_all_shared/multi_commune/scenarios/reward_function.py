"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np
from pathlib import Path

# Reward used in the CityLearn Challenge. Reward function for the multi-agent (decentralized) agents.
class reward_function_ma:
    def __init__(self, n_agents, building_info, deterministic):
        self.n_agents = n_agents
        self.building_info = building_info
        self.timestep = 0
        self.max_peak = (sum(building['charging_rate'] for uid, building in building_info.items())/ n_agents) *n_agents
    
    def normalize(self, value, min_val=0.09, max_val=0.21):
         return (value - min_val) / (max_val - min_val)
    
    def get_rewards(self, hour, diction, timestep, peak,  deterministic ):
        #print("*"*15)
        #print("DICTION", diction)
        # Initialize building rewards
        building_rewards = {building: 0 for building in diction}

        #Reward 2 
        for uid, building_info  in diction.items():
            price = building_info['price']
            b_soc = building_info['only_soc']
            present = building_info['present']
            charge = building_info['only_charge']
            in_temp = building_info['in_temp']
            hvac_energy = building_info['only_hvac']
            demand = charge + hvac_energy
            max_charge = self.building_info[uid]['charging_rate']
            max_hvac = self.building_info[uid]['nominal_power']
            n_price = self.normalize(price, min_val=0.09, max_val=0.21)

            comfort_reward = 1 if in_temp > 75 or in_temp < 67 else 0.0

            overcharge_penalty = 2*(-1 * max(0, b_soc - 95) / 5)
            no_charge_reward = 0.1 if charge == 0 and present == 0 else 0
            no_present_penal = -1 if charge > 0 and present == 0 else 0
            soc_reward =  (charge/max_charge)*present + overcharge_penalty*present + no_present_penal
            contribution = demand / peak if peak > 0 else 0

            if charge > 0:
                soc_reward = soc_reward 
                individual_reward = ((1- peak/self.max_peak) * contribution) * (1-n_price)
            else:
                soc_reward =   no_charge_reward
                individual_reward = 0

            w_price = 1

            building_rewards[uid] -= comfort_reward
            building_rewards[uid] += soc_reward  
            #building_rewards[uid] += -w_price* (n_price* ((charge + hvac_energy)/(max_charge+max_hvac)))
            building_rewards[uid] += -w_price* (n_price* (charge/max_charge + hvac_energy/max_hvac))
            building_rewards[uid] += individual_reward

        reward = np.array(list(building_rewards.values()))
        self.time_step = timestep
 
        return reward
    