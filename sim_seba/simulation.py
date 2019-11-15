# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:50:04 2019

@author: Sebastian
"""
#%%
from gurobipy import *
import simpy
import pickle
from numpy.random import RandomState, randint
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from itertools import product
import time
import numpy as np
from functools import partial, wraps
import sys
sys.path.append('C:/Users/sebas/OneDrive/Escritorio/Investigacion/Tesis/thesis/thesis/modules/')
from helper_funcs import load_pickle, get_company_assignment, save_pickle
from Fleet import update_objective, get_results, set_rho, modify_model, update_robust_cnstr

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.

    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)

            # Perform actual operation
            ret = func(*args, **kwargs)

             # Call "post" callback
            if post:
                post(resource)

            return ret
        return wrapper
    
    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))

def monitor(data, resource):
    #'''This is our monitoring callback.'''
    item = (resource._env.now,  # The current simulation time
            resource.count,  # The number of users
            #len(resource.queue),  # The number of queued processes
            )    
    data.append(item)

def update_firestations(original_df, new_values):
    temp = original_df.copy()
    temp.update(new_values)
    for k in P.vehicle_types:
        temp[k] = temp[k].astype(int)
    return temp

#####################################################################
######################### SIMULATION ################################
#####################################################################

class Fire_station(object):
    """
    Description: This class takes care of creating fire stations
    on the study region. 
    """
    firestations_dict = {}

    def __init__(self, env, firestation_id, district):
        self.env = env
        self.firestation_id = firestation_id
        self.vehicles = []
        self.district = district
        Fire_station.firestations_dict[firestation_id] = self

class Vehicle(object):
    """
    Description: This class takes care of creating vehicles resources assigned
    to each fire station. 
    """
    id_generator = 0
    turnOverTimeSeed = 0
    perDistrictAndType = {}
    vehiclesDict = {}

    def __init__(self, env, vtype, firestation_id):
        self.env = env
        self.id = Vehicle.id_generator
        Vehicle.id_generator += 1
        self.activate()
        self.type = vtype
        self.firestation = firestation_id
        self.attended_emergencies = []

    def activate(self):
        self.server = simpy.Resource(self.env, capacity=1)
    
    def attend_emergency(self, emergency, travel_time_to, 
                    travel_time_from, turnover_time,
                    last_response, service_time=100):
        self.attended_emergencies.append(emergency)
        with self.server.request() as request:
            yield request
            if P.verbose:
                district = Fire_station.firestations_dict[self.firestation].district
                print('Vehicle {}-{} from fire station {} of district {} has been selected at {}'.format(self.id, self.type, self.firestation, district, self.env.now))
            yield self.env.timeout(turnover_time)
            if P.verbose: 
                print('Vehicle {}-{} has been sent at {} with a turnover time of {}'.format(self.id, self.type, self.env.now, turnover_time))
            yield self.env.timeout(travel_time_to)
            if P.verbose:
                print('Vehicle {}-{} arrived at {}'.format(self.id, self.type, self.env.now))
            waiting_time = last_response-travel_time_to-turnover_time
            if waiting_time < 0.01:
                waiting_time = 0
            yield self.env.timeout(waiting_time)
            if P.verbose:
                print('Vehicle {}-{} waited {}'.format(self.id, self.type, waiting_time))
            yield self.env.timeout(service_time)
            if P.verbose:
                print('Vehicle {}-{} finish serving at {}'.format(self.id, self.type, self.env.now))
            yield self.env.timeout(travel_time_from)
            if P.verbose:
                print('Vehicle {}-{} came back to fire station {} at {}'.format(self.id, self.type, self.firestation, self.env.now))

    @classmethod
    def Pump_truck(cls, env, firestation_id):
        return cls(env, 'b', firestation_id)
    
    @classmethod
    def Ladder_truck(cls, env,  firestation_id):
        return cls(env, 'q', firestation_id)

    @classmethod
    def Hazmat_truck(cls, env, firestation_id):
        return cls(env, 'h', firestation_id)
    
    @classmethod
    def Forest_truck(cls, env, firestation_id):
        return cls(env, 'f', firestation_id)

    @classmethod
    def Rescue_truck(cls, env, firestation_id):
        return cls(env, 'h', firestation_id)
    
    @classmethod
    def new_vehicle_by_type(cls, env, v_type, firestation_id):
        if v_type == 'b':
            v = cls.Pump_truck(env, firestation_id)
        elif v_type == 'q':
            v = cls.Ladder_truck(env, firestation_id)
        elif v_type == 'h':
            v =  cls.Hazmat_truck(env, firestation_id)
        elif v_type == 'r':
            v = cls.Rescue_truck(env, firestation_id)
        elif v_type == 'f':
            v = cls.Forest_truck(env, firestation_id)
        cls.perDistrictAndType[Fire_station.firestations_dict[firestation_id].district][v_type].append(v)
        cls.vehiclesDict[v.id] = v
        return v

class Emergency_call_center(object):
    """
    Description: This class takes care of managing vehicle dispatch with the 
    purpose of responding to emergencies. 
    """

    def __init__(self, env, arrivals, travel_times_to, travel_times_from, stream):
        self.responseTimes = []
        self.arrival_gen = Arrivals_generator(env, arrivals, stream)
        self.travel_times_to = travel_times_to
        self.travel_times_from = travel_times_from
        self.env = env
        self.env.process(self.attend_emergency())
        self.S = stream
        self.dispatch_registry = {}

    def attend_emergency(self):
        while self.arrival_gen.emergency_count < len(self.arrival_gen.arrivals):
            emergency = yield self.env.process(self.arrival_gen.generateCall())
            self.sendFullResponse(emergency)

    def sendFullResponse(self, emergency):
        dispatch_info = []
        if emergency['type'] == 'otro':
            dispatch_protocol = self.S.generate_vtype_dispatch()
        else:
            dispatch_protocol = P.dispatch_protocol[emergency['district_type']][emergency['type']]
        for vehicle_type, dispatch_number in dispatch_protocol.items() :
            dispatch_dict = self.travel_times_to[str(emergency.name)][vehicle_type]
            dispatched_vehicles = 0
            for firestation_id, travel_time_to in dispatch_dict.items():
                temp_vehicles = Fire_station.firestations_dict[firestation_id].vehicles[vehicle_type]
                for vehicle in temp_vehicles:
                    if vehicle.server.count == 0:
                        turnover_time = self.S.generate_turnover_time()
                        travel_time_from = self.travel_times_from[str(emergency.name)][vehicle_type][firestation_id]
                        dispatch_info.append((vehicle, travel_time_to, travel_time_from, turnover_time))
                        dispatched_vehicles += 1
                    if dispatched_vehicles == dispatch_number:
                        break
                if dispatched_vehicles == dispatch_number:
                        break
        last_response = np.array([t_to+turnover_time for v, t_to, t_from, turnover_time in dispatch_info]).max()
        service_time = self.S.generate_service_time(emergency['type'])
        temp = []
        for vehicle, travel_time_to, travel_time_from, turnover_time in dispatch_info:  
            self.env.process(vehicle.attend_emergency(emergency.name, travel_time_to, travel_time_from, turnover_time, last_response, service_time))
            temp.append(vehicle.id)
        self.dispatch_registry[emergency.name] = {'response_time': last_response, 'dispatched_vehicles': temp, 'district_type': emergency['district_type'], 'type': emergency['type'], 'district': emergency['district']}

class Arrivals_generator(object):
    """
    Description: This class takes care of generating emergency calls arrivals. 
    """

    def __init__(self, env, arrivals, streams):
        self.env = env
        self.arrivals = arrivals
        self.streams = streams
        self.emergency_count = 0

    def generateCall(self):
        idx = self.arrivals.index[self.emergency_count]
        interarrival = self.arrivals.loc[idx,'interarrival']
        yield self.env.timeout(interarrival)
        e_type = self.arrivals.loc[idx]['type']
        if P.verbose:
            print('New emergency {}-{} ocurred at {} on district {}'.format(self.emergency_count, e_type, self.env.now, self.arrivals.loc[idx,'district']))
        self.emergency_count += 1
        return self.arrivals.loc[idx]

class Statistics(object):
    """
    Description: This class takes care of storing and processing statistics 
    produced by the simulation. 
    """

    @staticmethod
    def add_monitor():
        data_srv = {}
        monitor_srv = {}
        #Collect Statistics on Servers
        for vehicle_id, vehicle in Vehicle.vehiclesDict.items():
            data_srv[vehicle_id] = []
            monitor_srv[vehicle_id] = partial(monitor, data_srv[vehicle_id])
            patch_resource(vehicle.server, post=monitor_srv[vehicle_id])
        return data_srv

    @staticmethod
    def compute_utilization(data, sim_time):
        if len(data)>0:
            usage_time = 0
            for i, obs in enumerate(data):
                if (obs[1] == 1) & (i!=len(data)-1):
                    usage_time += data[i+1][0] - obs[0] 
                elif (obs[1] == 1) & (i == len(data)-1):
                    usage_time += sim_time - obs[0]

            utilization = usage_time/sim_time
        else:
            utilization = 0
        return utilization

    @classmethod
    def getUtilizationStatistics(cls, data, sim_time):
        avg_rho, rhos = {}, {}
        for n, k in product(Vehicle.perDistrictAndType.keys(), P.vehicle_types):
            temp = [cls.compute_utilization(data[v.id], sim_time) for v in Vehicle.perDistrictAndType[n][k]]
            avg_rho[n,k] = np.array(temp).mean()
            rhos[n,k] = temp
        avg_rho = {key:0 if np.isnan(item) else item for key, item in avg_rho.items()}
        _ = {'avg_rho':avg_rho, 'sim_rhos':rhos}
        return _

    @staticmethod
    def compute_coverage(data):
        response_results = {'response_time':{}, 'coverage':{}}
        for e_type in P.emergency_types:
            emergencies = {key:item for key, item in data.items() if item['type'] == e_type}
            response_times = np.array([item['response_time'] for _, item in emergencies.items()])
            responses_coverage = np.array([1 if (item['response_time']<= P.coverage_target[item['district_type']]) else 0 for _, item in emergencies.items()])
            response_results['response_time'][e_type] = {'avg': response_times.mean(), 'std': response_times.std(), 'data': response_times}
            response_results['coverage'][e_type] = responses_coverage.sum()/len(responses_coverage)
        new_data = {key: item for key, item in data.items() if item['type'] != 'otro'}
        response_times = np.array([item['response_time'] for _, item in new_data.items()])
        responses_coverage = np.array([1 if (item['response_time'] <= P.coverage_target[item['district_type']]) else 0 for _, item in data.items()])       
        response_results['response_time']['total'] = {'avg': response_times.mean(), 'std': response_times.std(), 'data': response_times}
        response_results['coverage']['total'] = responses_coverage.sum()/len(responses_coverage)
        return response_results
    
    @classmethod
    def getResponseStatistics(cls, data):
        results = {}
        for district in P.districts_dict.keys():
            emergencies = {key:item for key, item in data.items() if item['district'] == district}
            results[district] = cls.compute_coverage(emergencies)
        results['total'] = cls.compute_coverage(data)        
        return results

    @staticmethod
    def getCoverageAndResponse(data):
        results, results_ = {key:{} for key in data[0].keys()}, {key:{} for key in data[0].keys()}
        for key in data[0].keys():
            for key_ in data[0][key]['coverage'].keys():
                obs = np.array([replica[key]['coverage'][key_] for replica in data])
                results[key][key_] = {'avg': obs.mean(), 'std': obs.std()}
                obs = np.array([replica[key]['response_time'][key_]['avg'] for replica in data])
                results_[key][key_] = {'avg': obs.mean(), 'std': obs.std()}
        return results, results_

class P:
    """
    Description: This class takes care of declaring simulation parameters. 
    """
    mean_serv_time = {'fuego':3, 'rescate':1.5, 'hazmat':4, 
    'forestal':4, 'otro':1.5}
    emergency_types = {'fuego', 'rescate', 'hazmat', 'forestal', 'otro'}
    coverage_target = {'urban':10*60, 'mesourban':12*60, 'suburban':20*60}
    districts = {}
    default_rho = 0.01
    turn_over = (2, 3, 6)
    verbose = False
    vehicle_types = ['b', 'q', 'r', 'h', 'f']
    dispatch_protocol = {
        'urban': {
            'fuego': {'b':2, 'q':1},
            'rescate':{'r':1, 'b':1},
            'forestal': {'b':1, 'f':1},
            'hazmat':{'h':1, 'b':1}},
        'mesourban': {
            'fuego': {'b':1, 'q':1},
            'rescate':{'r':1, 'b':1},
            'forestal': {'b':1, 'f':1},
            'hazmat':{'h':1, 'b':1}},
        'suburban': {
            'fuego': {'b':2},
            'rescate':{'r':1, 'b':1},
            'forestal': {'f':1},
            'hazmat':{'b':1}}}

class Streams(object):
    """
    Description: This class takes care of the creation of all distributions
    inside the simulation. 
    """
    def __init__(self, service_time_seed, turnover_seed, v_type_seed):
        """
        This is the constructor of the class.
        
        Parameters:
            demand_seed: A seed to implement CRN on the demand
        """
        self.service_time_rand = RandomState()
        self.service_time_rand.seed(service_time_seed)
        self.turnover_time_rand = RandomState()
        self.turnover_time_rand.seed(turnover_seed)
        self.otro_rand = RandomState()
        self.otro_rand.seed(v_type_seed)

    def generate_service_time(self, e_type):
        """
        Implements random.exponential() with lambda(1/10)
        """
        mean = P.mean_serv_time[e_type]
        return self.service_time_rand.exponential(mean*3600)
    
    def generate_turnover_time(self):
        a, b, c = P.turn_over
        return self.service_time_rand.triangular(a*60, b*60, c*60)      
    
    def generate_vtype_dispatch(self):
        dispatch = [{'b':1}, {'r':1}, {'q':1}, {'h':1}, {'f':1}]
        prob = np.full(len(dispatch), 1/len(dispatch))
        return self.otro_rand.choice(dispatch, 1, p=prob)[0]

class Simulation_model(object):
    """
    This is the canvas equivalent found in all proprietary software. It consolidate
    all objects and information needed to run a single scenario with just one replicate

    Methods:
        __init__()
        run()
    """
    def __init__(self, seeds, firestations, arrivals, travel_times_to, travel_times_from, simulation_time):
        """
        This is the constructor of the Model object:
            
        Parameters:
            demand_seed: A seed to control randomness in arrivals
            vehicles: information with vehicles types and locations
            arrivals: dataframe with arrivals
            simulation_time: the length of the simulation
        """
        self.service_time_seed = seeds[0]
        self.turnover_seed = seeds[1]
        self.vtype_dispatch_seed = seeds[2]
        self.arrivals = arrivals
        self.firestations = firestations
        self.simulation_time = simulation_time
        self.travel_times_to = travel_times_to
        self.travel_times_from = travel_times_from
        
    def run(self):
        """
        This is the run method that integrates environment, streams and the
        simulation model
        
        Return:
            All the information collected by the observe method implemented in
            the Warehouse object
        """
        Vehicle.id_generator = 0
        env = simpy.Environment()
        S = Streams(self.service_time_seed, self.turnover_seed, self.vtype_dispatch_seed)
        Vehicle.perDistrictAndType = {district: {v_type: [] for v_type in P.vehicle_types} for district in self.firestations.district.unique()}
        
        #Create fire stations
        for firestation_id in self.firestations.index:
            temp_vehicles = {}
            Fire_station(env, firestation_id, self.firestations.loc[firestation_id]['district'])
            for vehicle_type in P.vehicle_types:
                temp_list = []
                for v in range(self.firestations.loc[firestation_id][vehicle_type]):
                    #Create Servers
                    temp_list.append(Vehicle.new_vehicle_by_type(env, vehicle_type, firestation_id))
                temp_vehicles[vehicle_type] = temp_list
            Fire_station.firestations_dict[firestation_id].vehicles = temp_vehicles
        #Create Emergency call center 
        ecc = Emergency_call_center(env, self.arrivals, self.travel_times_to, self.travel_times_from, S)
        #Add monitors to collect statistics
        data_srv = Statistics.add_monitor()
        env.run(until=self.simulation_time)
        response_stats = Statistics.getResponseStatistics(ecc.dispatch_registry)
        utilization_stats = Statistics.getUtilizationStatistics(data_srv, self.simulation_time)
        #ecc.dispatch_registry
        return response_stats, utilization_stats

class Location_model:

    def __init__(self, demands_dict, facilities_dict, districts_dict, load_dir):
        self.model = read(load_dir)
        self.demands_dict = demands_dict
        self.districts_dict = districts_dict
        self.facilities_dict = facilities_dict
    
    def update_model_params(self, p_new, p_reloc, q):
        self.districts_dict = {key: [i[0], i[1], p_reloc, p_new, i[4], q] \
            for key, i in self.districts_dict.items()}
        modify_model(self.model, self.districts_dict, self.facilities_dict)

    def get_demand_vars(self):
        variables = self.model.getVars()
        solution_variables = {var.VarName: {'value':var.X,'coef':var.Obj} for var in variables}
        sol_y = dict((tuple(key[2:-1].split(',')), value) for (key, value) in solution_variables.items() if key[0] == "y" )
        return sol_y

    def optimize_model(self):
        self.model.optimize()
        results = get_results(self.model, self.demands_dict)
        firestations = []
        vehicles = {l: [] for l in P.vehicle_types}
        for j, z_var in results['z'].items():
            firestations.append(j)
            s = [key[0] for key, item in results['s'].items() if (key[1]==j) and (item['value']>0.5)]
            if len(s)>0:
                j_ = s[0]
                for k, item in vehicles.items():
                    x_value = int(round(results['x'][j, k]['value'], 0))
                    if x_value == P.facilities_dict[j_][0][k]:
                        item.append(x_value)
                    else:
                        item.append(P.facilities_dict[j_][0][k])
            else:
                for k, item in vehicles.items():
                    item.append(int(round(results['x'][j, k]['value'], 0)))
        vehicles['node_id'] = firestations
        return pd.DataFrame(data=vehicles, index=vehicles['node_id']), self.get_demand_vars()

class Fleet_exc(Location_model):
    
    def __init__(self, demands_dict, facilities_dict, districts_dict, load_dir):
        Location_model.__init__(self, demands_dict, facilities_dict, districts_dict, load_dir)

    def update_obj(self, new_avg_rho):
        new_avg_rho = set_rho(P.default_rho, new_avg_rho)
        update_objective(self.model, self.demands_dict, self.districts_dict, new_avg_rho)

class Fleet(Location_model):
    
    def __init__(self, demands_dict, facilities_dict, districts_dict, load_dir):
        Location_model.__init__(self, demands_dict, facilities_dict, districts_dict, load_dir)

class Robust(Location_model):

    def __init__(self, demands_dict, facilities_dict, districts_dict, load_dir):
        Location_model.__init__(self, demands_dict, facilities_dict, districts_dict, load_dir)

    def update_cnstr(self, new_avg_rho):
        for key in new_avg_rho.keys():
            new_avg_rho[key] = set_rho(P.default_rho, new_avg_rho[key])
        update_robust_cnstr(self.model, self.demands_dict, self.districts_dict, new_avg_rho)

class All_new(Location_model):

    def __init__(self, demands_dict, facilities_dict, districts_dict, load_dir):
        Location_model.__init__(self, demands_dict, facilities_dict, districts_dict, load_dir)

    def optimize_model(self):
        self.model.optimize()
        results = get_results(self.model, self.demands_dict)
        firestations = []
        vehicles = {l: [] for l in P.vehicle_types}
        for j, z_var in results['z'].items():
            firestations.append(j)
            for k, item in vehicles.items():
                item.append(int(round(results['x'][j, k]['value'], 0)))
        vehicles['node_id'] = firestations
        return pd.DataFrame(data=vehicles, index=vehicles['node_id']), self.get_demand_vars()


class Experiment(object):
    """
    This class implements a parallelized process to perform experimental design 
    evaluation. It has two method implemented
        
        Methods:
            __init__()
            run()
    """
    seeds = [(i, i, i) for i in range(36)]

    def __init__(self, num_new, num_reloc, q, location_model, 
        initial_firestations, runtime=365*24*3600, epsilon=0.005):
        """
        This is the construtor of the class.
        
        Parameters:
            num_replics: An int value specifying the number of replicates to 
                        perform
            scenarios: A list of tuples, where each tuple contains the basic
                        information to instantiate a Model object
        """
        self.location_model = location_model
        self.location_model.update_model_params(num_new, num_reloc, q)
        #if P.verbose == False:
            #self.location_model.model.setParam('OutputFlag', 0)
        self.firestations = initial_firestations
        self.runtime = runtime
        self.epsilon = epsilon
        self.model_params = (num_new, num_reloc, q)

    def update_params(self, results):
        if self.location_model.__class__ == Robust:
            self.location_model.update_cnstr(results['rhos'])
        elif self.location_model.__class__ == Fleet_exc:
            self.location_model.update_obj(results['avg_rho'])
    
    def eval_stopping_criterion(self):
        if self.location_model.__class__ == Robust:
            if self.solutions[-1].equals(self.solutions[-2]):
                print("Stopping criterion met. Solution doesn't change.")
                criterion = 1
                return True, criterion
        elif self.location_model.__class__ == Fleet_exc:
            max_rho = np.array([abs(rho - exp.results[-2]['avg_rho'][key]) for key, rho in exp.results[-1]['avg_rho'].items()]).max()
            if max_rho < self.epsilon:
                print('Stopping criterion met. Max rho {}'.format(max_rho))
                criterion = 1
                return True, criterion
        if len(self.solutions)>2:
            for pos in range(len(self.solutions)-2):
                if self.solutions[-1].equals(self.solutions[pos]):
                    print('Stopped due to a loop. Solution already found.')
                    criterion = 2
                    return True, criterion
        print('Stopping criterion not met, continue with next iteration...')
        return False, 0
    
    def iterative_procedure(self, initial_params, initial_sols=[], iterations=3, update=False, **kwargs):
        tm = time.time()
        print('############### Experiment {} ###############'.format(self.model_params))
        if len(initial_sols) == 0:
            self.solutions = [self.firestations.copy()]
        else:
            self.solutions = initial_sols
        self.results = [{'avg_rho': initial_params['avg_rho'].copy()}]
        self.y_vars = [{}]
        if update:
            print('Updating model...')
            self.update_params(initial_params)
        criterion = 0
        proc_time = time.time()
        for iter_ in range(iterations):
            final_iter = iter_
            print('--------- Iteration {} ---------'.format(iter_))
            print('Optimizing model ...')
            sol, y_var = self.location_model.optimize_model()
            sol = update_firestations(self.firestations, sol)
            self.solutions.append(sol.copy())
            self.y_vars.append(y_var)
            print('Elapsed time: {}'.format(time.time()-tm))
            print('Simulating solution ...')
            sim_results = Experiment.compute_params(self.runtime, sol, **kwargs)
            print('Elapsed time: {}'.format(time.time()-tm))
            print('Updating model with computed parameters...')
            self.update_params(sim_results)
            self.results.append(sim_results)
            _, criterion = self.eval_stopping_criterion()
            print('Elapsed time: {}'.format(time.time()-tm))
            if _:
                break
            if iter_ == iterations-1:
                print('Stopped by number of iterations')
        total_runtime = time.time() - proc_time
        final_iter = iter_+1
        return self.solutions, self.results, total_runtime, criterion, final_iter, self.y_vars

    @staticmethod
    def compute_params(runtime, firestations, years=6, traffic=6):
        utilization_results, response_results = [], []
        temp = firestations[['node_id','b','q','r','f','h']]
        temp = temp.loc[~(temp[['b','q','r','f','h']]==0).all(axis=1)]   
        for k, (i, j) in enumerate(product(range(years), range(traffic))):
            print('--- Replica {}'.format(k))
            travel_times = \
                {'out': load_pickle('C:/Users/sebas/OneDrive/Escritorio/Investigacion/Tesis/Distances/distances_out_events_{}_{}.pickle'.format(i, j)),
                'in': load_pickle('C:/Users/sebas/OneDrive/Escritorio/Investigacion/Tesis/Distances/distances_in_events_{}_{}.pickle'.format(i, j))}
            response_times = get_company_assignment(travel_times, P.EVENTS[i], temp, P.REQUIREMENTS)
            sim = Simulation_model(Experiment.seeds[k], firestations, P.EVENTS[i], response_times['out'], response_times['in'] , runtime)
            response_stats, utilization_stats = sim.run()
            utilization_results.append(utilization_stats['avg_rho'])
            response_results.append(response_stats)
        print('Calculating simulation parameters...')            
        avg_utilizations = {key: np.array([r[key] for r in utilization_results]).mean() for key in utilization_results[0].keys()}
        std_utilizations = {key: np.array([r[key] for r in utilization_results]).std() for key in utilization_results[0].keys()}
        rhos = {key:r for key, r in enumerate(utilization_results)}
        coverage, response = Statistics.getCoverageAndResponse(response_results)
        results = {'avg_rho': avg_utilizations, 'std_rho': std_utilizations, \
            'rhos': rhos, 'coverage': coverage, 'response_time':response}
        return results


if __name__ == "__main__":
    FIRESTATIONS = pd.read_csv('csv/locations.csv')
    FIRESTATIONS['node_id'] = FIRESTATIONS.node_id.astype(str)
    FIRESTATIONS.set_index(['node_id'], drop=False, inplace=True)
    P.EVENTS = {}
    for i in range(6):
        P.EVENTS[i] = pd.read_csv('csv/events_{}.csv'.format(i))
        P.EVENTS[i].set_index(['Unnamed: 0'], drop=True, inplace=True)
        P.EVENTS[i]['node_id'] = P.EVENTS[i].node_id.astype(str)
    P.REQUIREMENTS = {'fuego':['b','q'], 'rescate':['b','r'], 'forestal':['b','f'],
                'hazmat':['b','h'], 'otro':['b','q','r','h','f']}
    P.demands_dict = load_pickle('pkl/demands_dict.pkl')
    P.districts_dict = load_pickle('pkl/districts_dict.pkl')
    P.facilities_dict = load_pickle('pkl/facilities_dict.pkl')

    '''
    experiments =  [(1,0,1), (1,0,2), (2,0,1), (2,0,2),
                   (0,1,0), (0,2,0), (1,1,1), (1,1,2), (1,2,1), (1,2,2),
                   (2,1,1), (2,1,2), (2,2,1), (2,2,2)]
    '''
    '''
    experiments = [(0,0,0)]

    #Run discrete experiments
    for p_new, p_reloc, q in experiments:
        dis = Fleet(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/discrete_fleet.lp')
        exp = Experiment(p_new, p_reloc, q, dis, FIRESTATIONS)
        tm_ = time.time()
        locs, y_vars = exp.location_model.optimize_model()
        locs = update_firestations(FIRESTATIONS, locs)
        runtime = time.time() - tm_
        results = {'locs':locs, 'runtime':runtime, 'y_vars':y_vars}
        save_pickle(results, 'pkl/results/discrete_fleet_{}_{}_{}.pickle'.format(p_new, p_reloc, q))

    # Run all new discrete
    all_new = All_new(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/discrete_all_new_fleet.lp')
    tm_ = time.time()
    locs, y_vars = all_new.optimize_model()
    locs = update_firestations(FIRESTATIONS, locs)
    runtime = time.time() - tm_
    results = {'locs':locs, 'runtime':runtime, 'y_vars':y_vars}
    save_pickle(results, 'pkl/results/discrete_all_new_fleet.pickle')
    
    #Run fleet
    initial_params = load_pickle('pkl/results/initial_params.pickle')
    for p_new, p_reloc, q in experiments:
        fleet = Fleet_exc(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/fleet.lp')
        exp = Experiment(p_new, p_reloc, q, fleet, FIRESTATIONS)
        locs, results, runtime, stopping_criterion, num_iter, y_vars = exp.iterative_procedure(initial_params, 10, years=6, traffic=5)
        results = {'locs':locs, 'results':results, 'runtime':runtime, 'stop':stopping_criterion, 'num_iter':num_iter, 'y_vars':y_vars}
        save_pickle(results, 'pkl/results/fleet_{}_{}_{}.pickle'.format(p_new, p_reloc, q))
    '''
    
    experiments =  [(1,0,1)]

    #Run robust
    initial_params = load_pickle('pkl/results/initial_params.pickle')
    for p_new, p_reloc, q in experiments:
        rob = Robust(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/robust_model.lp')
        exp = Experiment(p_new, p_reloc, q, rob, FIRESTATIONS)
        locs, results, runtime, stopping_criterion, num_iter, y_vars = exp.iterative_procedure(initial_params, iterations=10, years=6, traffic=5)
        results = {'locs':locs, 'results':results, 'runtime':runtime, 'stop':stopping_criterion, 'num_iter':num_iter, 'y_vars':y_vars}
        save_pickle(results, 'pkl/results/robust_{}_{}_{}_fixed.pickle'.format(p_new, p_reloc, q))
    
        
    '''
    #Continue to run
    #Run fleet
    experiments =  [(2,1,2)]
    for p_new, p_reloc, q in experiments:
        previous_results = load_pickle('pkl/results/fleet_{}_{}_{}.pickle'.format(p_new, p_reloc, q))
        initial_params = previous_results['results'][-1]
        initial_sols = previous_results['locs']
        fleet = Fleet_exc(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/fleet.lp')
        exp = Experiment(p_new, p_reloc, q, fleet, FIRESTATIONS)
        locs, results, runtime, stopping_criterion, num_iter, y_vars = exp.iterative_procedure(initial_params, initial_sols, 10, years=6, traffic=5)
        results = {'locs':locs, 'results':results, 'runtime':runtime, 'stop':stopping_criterion, 'num_iter':num_iter, 'y_vars':y_vars}
        save_pickle(results, 'pkl/results/fleet_{}_{}_{}_part_2.pickle'.format(p_new, p_reloc, q))

    experiments =  [(1,1,2), (2,1,1,), (2,1,2), (2,2,1)]
    for p_new, p_reloc, q in experiments:
        previous_results = load_pickle('pkl/results/robust_{}_{}_{}.pickle'.format(p_new, p_reloc, q))
        initial_params = previous_results['results'][-1]
        initial_sols = previous_results['locs']
        rob = Robust(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/robust_model.lp')
        exp = Experiment(p_new, p_reloc, q, rob, FIRESTATIONS)
        locs, results, runtime, stopping_criterion, num_iter, y_vars = exp.iterative_procedure(initial_params, initial_sols, 10, years=6, traffic=5)
        results = {'locs':locs, 'results':results, 'runtime':runtime, 'stop':stopping_criterion, 'num_iter':num_iter, 'y_vars':y_vars}
        save_pickle(results, 'pkl/results/robust_{}_{}_{}_part_2.pickle'.format(p_new, p_reloc, q))
    '''
    #%%
    #Run robust
    initial_params = load_pickle('pkl/results/initial_params.pickle')
    rob = Robust(P.demands_dict, P.facilities_dict, P.districts_dict, 'models/robust_model.lp')
    rhos = {key: {_: rho if rho>0 else 0.1 for _,rho in item.items()} for key, item in initial_params['rhos'].items() if int(key)<30}

    rob.update_model_params(1, 0, 1)
    rob.update_cnstr(rhos)
    rob.model.write('models/robust_cnstr_update.lp')

#%%
