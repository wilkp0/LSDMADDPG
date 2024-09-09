from gym import spaces
import numpy as np
import random 
import json
import pandas as pd


class Buildings:
    def __init__(self, save_memory=True):
        self.buildings = {}
        self.observation_spaces = []
        self.action_spaces = []
        self.save_memory = save_memory

    def building_loader(data_path, building_attributes, weather_file, building_ids, buildings_states_actions, simulation_period, deterministic, save_memory = True):
        n_agents = len(building_ids)
        n_strategies = 1
        with open(building_attributes) as json_file:
            data = json.load(json_file)
        buildings, observation_spaces, action_spaces, strategy_spaces = {},[],[],[]

        for uid, attributes in zip(data, data.values()):
            if uid in building_ids:

                charger = Charging_Station(efficiency = attributes['Charger']['efficiency'],
                                            charging_rate = attributes['Charger']['charging_rate'],
                                            capacity = attributes['Charger']['capacity'],
                                            save_memory = save_memory)        
                thermmodel = ThermalModel(resist = attributes['House']['Req'],
                                            c = attributes['House']['c'],
                                            theater = attributes['House']['THeater'],
                                            tcool = attributes['House']['Tcool'],
                                            mdot = attributes['House']['Mdot'],
                                            m = attributes['House']['M'],
                                            tinic = attributes['House']['TinIC'],
                                            save_memory = save_memory)
                hvac = HVAC(thermmodel=thermmodel,nominal_power = attributes['HVAC']['nominal_power'], 
                    eta_tech = attributes['HVAC']['technical_efficiency'], 
                    t_target_temp = attributes['HVAC']['t_target_temp'], 
                    save_memory = save_memory)
                
                building = Building(buildingId = uid, charging_station = charger, hvac = hvac, save_memory = save_memory)
            
                #Datetime
                data_file = 'Building_1' + '.csv'
                simulation_data = data_path / data_file
                with open(simulation_data) as csv_file:
                    date_data = pd.read_csv(csv_file)
                building.sim_results['month'] = list(date_data['Month'][0:8760])
                building.sim_results['day'] = list(date_data['Day Type'][0:8760])
                building.sim_results['hour'] = list(date_data['Hour'][0:8760])


                ''' Charger ''' 
                # Load presence data based on building type
                presence_file_path = data_path / f"{attributes['Building_Type']}_presence.csv"
                with open(presence_file_path) as csv_file:
                    presence_data = pd.read_csv(csv_file)
                building.sim_results['present'] = list(presence_data['Presence'])
                charger.sim_results['present'] = list(presence_data['Presence'])

                price_file_path = data_path / "yearly_price_profile.csv"
                with open(price_file_path) as csv_file:
                    price_data = pd.read_csv(csv_file)
                building.sim_results['price'] = list(price_data['Price'])
                
                weather_file = data_path / 'weather_data_new.csv'
                with open(weather_file) as csv_file:
                    weather_data = pd.read_csv(csv_file)
                building.sim_results['t_out'] = list(weather_data['Outdoor Drybulb Temperature [C]'][0:8760])
                
                # Reading the building attributes
                building.building_type = attributes['Building_Type']
                building.nominal_power = attributes['HVAC']['nominal_power']
                building.capacity = attributes['Charger']['capacity']
                building.charging_rate = attributes['Charger']['charging_rate']

                #Calculating Min and Max for states for Normalization 
                s_low, s_high = [], []
                st_low, st_high = [], []
                for state_name, value in zip(buildings_states_actions[uid]['states'], buildings_states_actions[uid]['states'].values()):
                    if value == True:
                        '''HVAC'''
                        if state_name == "t_in":
                            s_low.append(60)
                            s_high.append(80)
                        elif state_name == "hvac_energy":
                            s_low.append(0)
                            s_high.append(building.nominal_power)
                            '''Charging Station'''
                        elif state_name == "soc":
                            s_low.append(0)
                            s_high.append(100)   
                        elif state_name == "charger_power": 
                            s_low.append(0)                                                                                                                     
                            s_high.append(building.charging_rate)
                        elif state_name == "peak": 
                            s_low.append(0.0)       
                            total_power_and_charge = sum(building['HVAC']['nominal_power'] + building['Charger']['charging_rate'] for building in data.values())
                            s_high.append(total_power_and_charge)
                        elif state_name == "strategy": 
                            st_low.extend([0] * (n_agents * n_strategies))
                            st_high.extend([1] * (n_agents * n_strategies))      
                            ''' Other States ''' 
                        elif (state_name != "t_in")  and(state_name != "hvac_energy") and(state_name != "soc") and(state_name != "charger_power"):
                            s_low.append(min(building.sim_results[state_name]))
                            s_high.append(max(building.sim_results[state_name]))
                        else:
                            s_low.append(0.0)
                            s_high.append(1.0)

                a_low, a_high, building.action_type = [], [], []  
                for action_name, value in zip(buildings_states_actions[uid]['actions'], buildings_states_actions[uid]['actions'].values()):
                    if value == True:
                        if action_name =='thermostat':
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            building.action_type.append(action_name)
                        elif action_name =='charger':
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            building.action_type.append(action_name)

                building.set_state_space(np.array(s_high), np.array(s_low))
                building.set_action_space(np.array(a_high), np.array(a_low))
                building.set_strat_space(np.array(st_high), np.array(st_low))

                observation_spaces.append(building.observation_space)
                action_spaces.append(building.action_space)
                strategy_spaces.append(building.strategy_space)

                buildings[uid] = building

        for building in buildings.values():
            building.reset()

        print("tradegy space", strategy_spaces)
        return buildings, observation_spaces, action_spaces, strategy_spaces


class Building:  
    def __init__(self, buildingId, charging_station = None, hvac = None,  save_memory = True):


        self.building_type = None
        self.buildingId = buildingId
        self.charging_station = charging_station
        self.hvac = hvac
        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
        
        #if self.charging_station is not None:
        #    self.charging_station.reset()
        #if self.hvac is not None:
        #    self.hvac.reset()

        self.charging_station_energy = 0.0
        self.charging_power = 0 
        self.set_temperature = 0

    def set_state_space(self, high_state, low_state):
        # Setting the state space and the lower and upper bounds of each state-variable
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_strat_space(self, high_state, low_state):
        # Setting the state space and the lower and upper bounds of each state-variable
        self.strategy_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

    def set_action_space(self, max_action, min_action):
        # Setting the action space and the lower and upper bounds of each action-variable
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

    def set_building_temperature(self,action):

        #Building Limits
        building_max_temp = 80
        building_min_temp = 60
        # Normalize action 
        normalized_action = (action + 1) / 2 * (building_max_temp - building_min_temp) + building_min_temp
        self.set_temperature = normalized_action 

        self.hvac_energy = self.hvac.control(self.set_temperature, self.sim_results['t_out'][self.time_step])


        self.hvac.time_step = self.time_step + 1 
        return self.hvac_energy

    def set_charging_power(self,action):
        # or amount of energy delivered to the vehicle (kW)
        charging_rate_min = 0
        charging_rate_max = 1
        # Normalize action 
        normalized_action = (action + 1) / 2 * (charging_rate_max - charging_rate_min) + charging_rate_min
        self.charging_power = normalized_action*self.charging_station.charging_rate
        self.norm_action = self.charging_power
        # Set charging power, ensuring any power below 1.5 is set to 0
        self.charging_power = self.charging_power if self.charging_power >= 1.5 else 0
        self.charging_station_energy = self.charging_station.charger(self.charging_power)

        self.charging_station.time_step = self.time_step + 1
        return self.charging_station_energy
    
    def reset(self):
        if self.charging_station :
            self.charging_station.reset()
        if self.hvac :
            self.hvac.reset()

    def terminate(self):
        if self.charging_station :
            self.charging_station.terminate()
        if self.hvac:
            self.hvac.terminate()

class Charging_Station:
    def __init__(self, capacity = None, efficiency = None, charging_rate= None, save_memory = True):
        self.save_memory  = save_memory
        self.time_step = 0 
        ''' Charger ''' 
        self.charging_rate = charging_rate 
        self.efficiency = efficiency

        ''' Electric Vehivle ''' 
        self.capacity = capacity 
        self.soc_init = np.random.uniform(0.05, 0.5 )*self.capacity
        self.soc = self.soc_init

        self.sim_results = {}
    
    def terminate(self):
        pass

    def charger(self, power = 0):

        self.efficiency = 1 #(-0.40478*power**2 + 6.23059*power + 66.8633)/100
        '''Scaled EV eff'''
        #self.efficiency = (-0.1695*power**2 + 4.0321*power + 66.8633)/100
        #self.efficiency = (-0.40478 * (11 / max_power)**2 * power**2 + 6.23059 * (11 / max_power) * power + 66.8633)
        self.energy_to_ev = power* self.efficiency
        self.updated_ev_soc = self.electric_vehicle(self.energy_to_ev)
        return power

    def reset_ev_soc_if_necessary(self):
        min_soc = 36
        max_soc = 66
        if self.time_step > 0:
            previous_state = self.sim_results['present'][self.time_step - 1]
            current_state = self.sim_results['present'][self.time_step]
            if previous_state == 0 and current_state == 1:
                self.soc = random.randint(min_soc, max_soc)
                self.soc = 36
        else:
            self.soc = 36


    def electric_vehicle(self, energy):
        self.reset_ev_soc_if_necessary()
        self.present = self.sim_results['present'][self.time_step]
        if not self.present:
            self.soc = 0.0
        else:
            self.soc = self.soc + (energy/self.capacity)*100
        self.soc = min(self.soc, 100)

    def reset(self):
        self.soc = self.sim_results['present'][self.time_step]*36
        self.energy_to_ev = 0
        ''' Electric Vehivle ''' 

class HVAC:
    def __init__(self, thermmodel=None, nominal_power=None, eta_tech=None, t_target_temp=None, save_memory=True):
        self.thermmodel = thermmodel
        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        self.t_target_temp = t_target_temp
        self.time_step = 0
        self.save_memory = save_memory
        self.indoor_temp = 0
        self.comfort = 0
        self.thermmodel.max_power = self.nominal_power
        self.thermmodel.eta_tech = self.eta_tech


    def control(self, set_temperature=70, outdoor_temp=20):
        '''Send the temperature to the thermostat and receive: energy, temp, cost'''
        hvac_energy, house_temp = self.thermmodel.connections(set_temperature, outdoor_temp)
        self.indoor_temp = house_temp

        return hvac_energy

    def reset(self):
        self.nominal_power = None
        self.eta_tech = None
        self.t_target_temp = None

    def terminate(self):
        pass


class ThermalModel:
    def __init__(self, resist=None, c=None, theater=None, tcool=None, mdot=None, m=None, tinic=None,max_power=None, eta_tech=0.25, save_memory=True):
        self.frequency = 60
        self.action = 0
        self.Req = resist
        self.c = c
        self.THeater = theater
        self.TCool = tcool
        self.Mdot = mdot
        self.M = m
        self.Troom = tinic
        self.blowercmd = 0
        self.outgain = tinic
        self.eta_tech = eta_tech
        self.max_power = max_power


    def connections(self, action, tout):
        self.action = action
        self.hourlytemp = tout
        self.kWh = 0

        # Update COPs based on current outdoor temperature
        self.update_cops(tout)
        for _ in range(self.frequency):
            self.Terr = self.FahrtoCels(self.action) - self.Troom
            self.blowercmd = self.Thermostat(self.Terr)
            self.heat_flow = self.Heater(self.blowercmd, self.Troom)
            self.kWh += self.energyCalc()
            self.Tout = self.hourlytemp
            self.Troom = self.House(self.heat_flow, self.Tout)
        self.temp_building = self.CelstoFahr(self.Troom)
        return self.kWh, self.temp_building

    def House(self, heat_flow, Tout):
        self.HeatLosses = ((self.Troom - Tout) * 1/self.Req)*60  
        self.outgain = (1 / (self.M * self.c)) * (heat_flow - self.HeatLosses)
        self.Troom += self.outgain
        return self.Troom

    def Heater(self, blowercmd, Troom):
        heat, cool = blowercmd
        self.heat = heat
        self.cool = cool
        # Calculate the heating and cooling gains in watts
        HeatGain = max(0, min(((self.THeater - Troom) * (self.Mdot * self.c))*60, self.max_power * 1000 * 5*60))
        CoolGain = max(0, min(((Troom - self.TCool) * (self.Mdot * self.c))*60, self.max_power * 1000 * 5*60))

        self.heater_out = heat * HeatGain
        self.cool_out = cool * CoolGain

        return self.heater_out - self.cool_out

    def FahrtoCels(self, F):
        return 5 / 9 * (F - 32)

    def CelstoFahr(self, C):
        return 9 / 5 * C + 32

    def Thermostat(self, Terr):
        switch_on_point_heat = 0.6
        switch_on_point_cool = -0.6
        heat = 1 if Terr > switch_on_point_heat else 0
        cool = 1 if Terr < switch_on_point_cool else 0
        return heat, cool

    def energyCalc(self):
        energy = (self.cool_out / self.cop_cooling) + (self.heater_out / self.cop_heating)
        return  energy * (1 / 3.6e6)  #from Watts to kwh
    

    def update_cops(self, outdoor_temp):
        min_temp_diff = 0.1
        self.cop_heating = np.clip(self.eta_tech * (self.THeater + 273.15) / np.clip((self.THeater - outdoor_temp), min_temp_diff, None), 1, 5)
        self.cop_cooling = np.clip(self.eta_tech * (self.TCool + 273.15) / np.clip((outdoor_temp - self.TCool), min_temp_diff, None), 1, 5)
