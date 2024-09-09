import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
from gym import spaces
from multi_commune.scenarios.reward_function import reward_function_ma
from pathlib import Path
from multi_commune.building import Buildings
import math
from multi_commune.preprocessing import *
from gym.spaces import Box


gym.logger.set_level(48)


class Community(gym.Env):
    def __init__(self, data_path, building_attributes, weather_file, building_ids,  buildings_states_actions = None, simulation_period = (0,8759), deterministic =False, cost_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption', 'net_electricity_cost','net_comfort_penalty'], save_memory = True, verbose = 0):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        # set required vectorized gym env property
        self.n = len(building_ids)
        self.data_path = data_path
        self.building_attributes = building_attributes
        self.building_ids = building_ids
        self.weather_file = weather_file
        self.verbose = verbose
        self.simulation_period = simulation_period
        self.deterministic = deterministic
        self.terminal = []

        params_loader = {'data_path':data_path,
                         'building_attributes':self.data_path / self.building_attributes,
                         'weather_file':self.data_path / self.weather_file,
                         'building_ids':building_ids,
                         'buildings_states_actions':self.buildings_states_actions,
                         'simulation_period' : simulation_period,
                         'deterministic' : deterministic,
                         'save_memory':save_memory}

        self.buildings, self.observation_space, self.action_space = Buildings.building_loader(**params_loader)
        print("OBSSPACE", self.observation_space)
        self._create_encoder()
        self.update_observation_space()
        self.reset(eval)

    def get_state_action_spaces(self):
        return self.observation_space, self.action_space
    
    def select_random_48_hours(self, evaluation_mode=False):
        if evaluation_mode:
            # Define the first 48-hour period starting from the first occurrence of 12 PM
            print("Evaluating")
            twelve_pm_hours = np.where(np.array(self.buildings[list(self.buildings.keys())[0]].sim_results['hour']) == 12)[0]
            if len(twelve_pm_hours) == 0:
                raise ValueError("No 12 PM hours found in simulation data.")
            start_hour = twelve_pm_hours[0]
            # Ensure the 48-hour window does not exceed the year bounds
            if start_hour + 48*5 > len(self.buildings[list(self.buildings.keys())[0]].sim_results['hour']):
                raise ValueError("The 48-hour period exceeds the available simulation data.")
            self.selected_hours = np.array(range(start_hour, start_hour + 48*5 + 1))
        else:
            # Original random selection logic
            twelve_pm_hours = np.where(np.array(self.buildings[list(self.buildings.keys())[0]].sim_results['hour']) == 12)[0]
            start_hour = random.choice(twelve_pm_hours)
            if start_hour + 48 > len(self.buildings[list(self.buildings.keys())[0]].sim_results['hour']):
                start_hour = len(self.buildings[list(self.buildings.keys())[0]].sim_results['hour']) - 48 - 1
            self.selected_hours = np.array(range(start_hour, start_hour + 48 + 1))
        
        return self.selected_hours
        
    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings.values():
            building.time_step = self.time_step


    def get_building_information(self):
        np.seterr(divide='ignore', invalid='ignore')
        # Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand
        building_info = {}
        for uid, building in self.buildings.items():
            building_info[uid] = {}
            building_info[uid]['building_type'] = building.building_type
            building_info[uid]['nominal_power'] = building.nominal_power 
            building_info[uid]['capacity']  =  building.capacity 
            building_info[uid]['charging_rate'] = building.charging_rate 
        return building_info
    
    def get_building_data(self):
        np.seterr(divide='ignore', invalid='ignore')
        # Houlry Weather and Building info 
        building_data = {}
        period = self.simulation_period[0], self.simulation_period[1]
        for uid, building in self.buildings.items():
            building_data[uid] = {}
            building_data[uid]['hour'] = building.sim_results['hour'][period[0]:period[1]]
            building_data[uid]['tout'] = building.sim_results['t_out'][period[0]:period[1]]
            building_data[uid]['price'] = building.sim_results['price'][period[0]:period[1]]
        return building_data

    def step(self, actions):
        self.building_rewards = {}
        self.peak_hour = 0

        for uid, building in self.buildings.items():
            self.building_rewards[uid] = {}
            price = building.sim_results['price'][self.time_step]
            if self.buildings_states_actions[uid]['actions']['thermostat']:
                building_hvac_energy = building.set_building_temperature(actions[0])
                self.peak_hour += building_hvac_energy
                self.building_rewards[uid].update({
                'only_hvac': building_hvac_energy,
                'thermostat': building.set_temperature,
                'in_temp': building.hvac.indoor_temp})
                #EXCEL Outputs
                self.results_hvac[uid].append(building_hvac_energy)
                self.results_temp[uid].append(building.hvac.indoor_temp)
                self.results_thermost[uid].append(building.set_temperature)

                actions = actions[1:]
            if self.buildings_states_actions[uid]['actions']['charger']:
                building_charger_energy = building.set_charging_power(actions[0])
                self.peak_hour += building_charger_energy
                self.building_rewards[uid].update({
                'only_charge': building_charger_energy,
                'only_soc': building.charging_station.soc,
                'present': building.charging_station.present})

                #EXCEL OUTPUTS
                self.results_action[uid].append(building.norm_action)
                self.results_soc[uid].append(building.charging_station.soc) 
                self.results_charge[uid].append(building_charger_energy)
                self.results_present[uid].append(building.charging_station.present)

                actions = actions[1:]
            else:
                building_hvac_energy = building.set_building_temperature(0)
                building_charger_energy = building.set_charging_power(0)

            self.building_rewards[uid].update({
                'price': price})
            
        for uid, building in self.buildings.items():
            self.results_peak[uid].append(sum(self.building_rewards[uid]['only_hvac'] + self.building_rewards[uid]['only_charge'] for uid in self.buildings))
        
        self.next_hour()
        self.state = []
        peak = (sum(self.building_rewards[uid]['only_hvac'] + self.building_rewards[uid]['only_charge'] for uid in self.buildings))
        s = []
        state_names = []
        for uid, building in self.buildings.items():
            for state_name, value in self.buildings_states_actions[uid]['states'].items():
                if value == True:
                    '''HVAC'''
                    if state_name == "t_in":
                        s.append(building.hvac.indoor_temp)
                        state_names.append(state_name)
                    elif state_name == "hvac_energy":
                        s.append(building_hvac_energy)
                        state_names.append(state_name)
                        '''Charging Station'''
                    elif state_name == "soc":
                        s.append(building.charging_station.soc)
                    elif state_name == "charger_power":
                        s.append(building_charger_energy) 
                    elif state_name == "peak":
                        s.append(peak)
                        ''' Other States '''
                    elif (state_name != "t_in") and (state_name != "hvac_energy") and (state_name != "soc") and(state_name != "charger_power") and(state_name != "penalty") and(state_name != "arrival")and(state_name != "peak"):
                        s.append(building.sim_results[state_name][self.time_step])
                    state_names.append(state_name)

        self.state = np.array(s, dtype=np.float32)

        '''             REWARD          '''
        rewards = self.reward_function.get_rewards(building.sim_results['hour'][self.time_step],self.building_rewards, self.time_step, peak, self.deterministic)
        
        for (uid, building), reward in zip(self.buildings.items(), rewards):
            self.results_rew[uid].append(reward)

        terminal = self._terminal(self.deterministic)
        return (self._get_ob(), sum(rewards)/3, terminal, {})

    def reset(self, eval=False):
        #Initialization of variables

        self.selected_hours = self.select_random_48_hours(eval)
        self.hour = iter(self.selected_hours)
        self.next_hour()

        self.results_action = {uid: [] for uid in self.buildings}
        self.results_rew = {uid: [] for uid in self.buildings}
        self.results_comf = {uid: [] for uid in self.buildings}
        self.results_thermost = {uid: [] for uid in self.buildings}
        self.results_charge = {uid: [] for uid in self.buildings}
        self.results_hvac = {uid: [] for uid in self.buildings}
        self.results_soc = {uid: [] for uid in self.buildings}
        self.results_temp = {uid: [] for uid in self.buildings}
        self.results_present = {uid: [] for uid in self.buildings}
        self.results_peak = {uid: [] for uid in self.buildings}


        self.reward_function = reward_function_ma(len(self.building_ids), self.get_building_information(), self.deterministic)
        self.state = []
        # Calculate initial HVAC and charging power without modifying state
        initial_hvac_energy = {}
        initial_charger_energy = {}

        for uid, building in self.buildings.items():
            initial_hvac_energy[uid] = building.set_building_temperature(0.575)
            initial_charger_energy[uid] = building.set_charging_power(random.uniform(-1, 1)) * building.sim_results['present'][self.time_step]

        peak = sum(initial_hvac_energy[uid] + initial_charger_energy[uid] for uid in self.buildings)
        s = []
        state_names = []
        for uid, building in self.buildings.items():
            building.reset()
            for state_name, value in self.buildings_states_actions[uid]['states'].items():
                if value:
                    '''HVAC'''
                    if state_name == "t_in":
                        s.append(building.hvac.indoor_temp)
                        state_names.append(state_name)
                    elif state_name == "hvac_energy":
                        s.append(initial_hvac_energy[uid])  # Use initial value
                        state_names.append(state_name)
                        '''Charging Station'''
                    elif state_name == "soc":
                        s.append(building.charging_station.soc)
                    elif state_name == "charger_power":
                        s.append(initial_charger_energy[uid])  # Use initial value
                    elif state_name == 'peak': 
                        s.append(peak)                        
                        ''' Other States '''
                    elif (state_name != "t_in") and (state_name != "hvac_energy") and (state_name != "soc") and(state_name != "charger_power")and(state_name != "arrival")and(state_name != "penalty"):
                        s.append(building.sim_results[state_name][self.time_step])
            state_names.append(state_name)
        self.state = np.array(s, dtype=np.float32)
        return self._get_ob()
    

    def _get_ob(self):
        state_ = np.array(self.encoder * self.state)  # Element-wise multiplication
        flattened_state = []
        for item in state_:
            if isinstance(item, np.ndarray):
                flattened_state.extend(item.flatten())
            else:
                flattened_state.append(item)
        flattened_state = np.array(flattened_state)
        return flattened_state

    def _create_encoder(self):
        ''' Encoder ''' 
        # Assuming buildings are indexed the same as observation spaces
        state_n = 0
        self.encoder = []
        obs_space = self.observation_space  # Single observation space
        for uid, building in self.buildings.items():
            for s_name, s in self.buildings_states_actions[uid]['states'].items():
                if not s:
                    continue
                elif s_name in ["month", "hour"]:
                    # Normalize month and hour if the length check is valid
                    self.encoder.append(periodic_normalization(obs_space.high[state_n]))
                    state_n += 1
                elif s_name == "day":
                    # One-hot encoding for days
                    self.encoder.append(onehot_encoding([1,2,3,4,5,6,7,8]))
                    state_n += 1
                else:
                    # Normalizing other states
                    self.encoder.append(normalize(obs_space.low[state_n], obs_space.high[state_n]))
                    state_n += 1  

        # Create np.array of normalized states for the single agent 
        self.encoder = np.array(self.encoder)

    def update_observation_space(self):
        states = self.reset(eval)
        state = states  # Assuming states is a list with a single state for the single agent
        low = np.full(state.shape, -1, dtype=state.dtype)
        high = np.full(state.shape, 1, dtype=state.dtype)
        self.observation_space = Box(low=low, high=high, dtype=state.dtype)
        print("ENV obs space", self.observation_space)
        
    def _terminal(self, deterministic):
        is_terminal = bool(self.time_step >= self.selected_hours[-1])
        if is_terminal:
            for building in self.buildings.values():
                building.terminate()
            #self.reset()
        return is_terminal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def eval(self, logger, eval_num):
        building_data = self.get_building_data()
        for building in self.buildings.values():
            building.terminate()
        # Log the scalars to TensorBoard for each agent and each timestep

        for uid, agent_i in zip(self.buildings.keys(), range(len(self.buildings.keys()))):
            for timestep in range(len(self.results_charge[uid])):
                logger.add_scalars('agent%i/eval%i' % (agent_i, eval_num),
                                        {
                                            'HVACEnergy': (self.results_hvac[uid][timestep]),
                                            'TempOut': (building_data[uid]['tout'][timestep]),
                                            'TempIn': (self.results_temp[uid][timestep]),
                                            #'Comfort': (self.results_comf[uid][timestep]),
                                            #'Thermostat': (self.results_thermost[uid][timestep]),
                                            'Charge': (self.results_charge[uid][timestep]),
                                            #'Price':  building_data[uid]['price'][timestep],
                                            'SOC': (self.results_soc[uid][timestep]),
                                            #'Presence': self.results_present[uid][timestep],
                                            #'Penalty': self.results_penalty[uid][timestep],
                                            'Peak': self.results_peak[uid][timestep],
                                            #'Reward': self.results_rew[uid][timestep],
                                            #'Action': self.results_action[uid][timestep],
                                        },
                                        timestep)
                

    import pandas as pd
    from openpyxl import Workbook

    def get_final_results(self):
        max_length = 48*5  # Ensure all data arrays match this length

        # Initialize the dictionary to hold the data for each building
        results_dict = {
            'UID': [],
            'HVAC_Energy': [],
            'TempIn': [],
            'Thermostat': [],
            'Charge': [],
            'SOC': [],
            'Peak': [],
            'Reward': [],
            'Price': [],
            'Hour': [],
            'Total_Energy': []
        }

        # Collect data from results
        for uid in self.buildings.keys():
            hvac_energy = self.results_hvac[uid][:max_length]
            temp_in = self.results_temp[uid][:max_length]
            thermostat = self.results_thermost[uid][:max_length]
            charge = self.results_charge[uid][:max_length]
            soc = self.results_soc[uid][:max_length]
            peak = self.results_peak[uid][:max_length]
            reward = self.results_rew[uid][:max_length]
            price = [self.buildings[uid].sim_results['price'][i] for i in self.selected_hours[:max_length]]
            hour = [self.buildings[uid].sim_results['hour'][i] for i in self.selected_hours[:max_length]]
            total_energy = [hvac + chg for hvac, chg in zip(hvac_energy, charge)]

            results_dict['UID'].append(uid)
            results_dict['HVAC_Energy'].append(hvac_energy)
            results_dict['TempIn'].append(temp_in)
            results_dict['Thermostat'].append(thermostat)
            results_dict['Charge'].append(charge)
            results_dict['SOC'].append(soc)
            results_dict['Peak'].append(peak)
            results_dict['Reward'].append(reward)
            results_dict['Price'].append(price)
            results_dict['Hour'].append(hour)
            results_dict['Total_Energy'].append(total_energy)

        with pd.ExcelWriter('final_results_com_central_week.xlsx', engine='openpyxl') as writer:
            for uid in self.buildings.keys():
                building_df = pd.DataFrame({
                    'HVAC_Energy': results_dict['HVAC_Energy'][results_dict['UID'].index(uid)],
                    'TempIn': results_dict['TempIn'][results_dict['UID'].index(uid)],
                    'Thermostat': results_dict['Thermostat'][results_dict['UID'].index(uid)],
                    'Charge': results_dict['Charge'][results_dict['UID'].index(uid)],
                    'SOC': results_dict['SOC'][results_dict['UID'].index(uid)],
                    'Peak': results_dict['Peak'][results_dict['UID'].index(uid)],
                    'Reward': results_dict['Reward'][results_dict['UID'].index(uid)],
                    'Price': results_dict['Price'][results_dict['UID'].index(uid)],
                    'Hour': results_dict['Hour'][results_dict['UID'].index(uid)],
                    'Total_Energy': results_dict['Total_Energy'][results_dict['UID'].index(uid)]
                })
                building_df.to_excel(writer, sheet_name=f'Building_{uid}', index=False)

            # Adding a summary sheet with the sum of HVAC_Energy and Charge for each building
            summary_dict = {
                'UID': results_dict['UID'],
                'Total_HVAC_Energy': [sum(hvac) for hvac in results_dict['HVAC_Energy']],
                'Total_Charge': [sum(chg) for chg in results_dict['Charge']],
                'Total_Energy': [sum(energy) for energy in results_dict['Total_Energy']],
                'Total_Reward': [sum(reward) for reward in results_dict['Reward']]
            }
            summary_df = pd.DataFrame(summary_dict)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print("Results have been exported to final_results_com_central_week.xlsx")


