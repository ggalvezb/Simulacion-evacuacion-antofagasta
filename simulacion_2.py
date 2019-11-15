import itertools
import pandas as pd
from collections import OrderedDict
import numpy as np
import geopandas as gpd
from numpy.random import RandomState
import simpy
import sys
import time   #Para probar los tiempos de ejecucion

class Family(object):
    ID=0
    families=[]
    def __init__(self, env, members, housing, start_scape, velocity, route):
        self.ID=Family.ID
        Family.ID+=1                    
        self.members = members          
        self.housing = housing           
        self.start_scape = start_scape  
        self.velocity = velocity                
        self.route = route              
        self.env=env

        #The family go out
        self.env.process(self.evacuate())


    @staticmethod
    def get_members(element):
        age_list=list(synthetic_population.loc[synthetic_population['House ID']==element].Age)
        sex_list=list(synthetic_population.loc[synthetic_population['House ID']==element].Sex)
        adult=len([l for l in age_list if 18<=l<60])
        young=len([l for l in age_list if 12<=l<18])
        kid=len([l for l in age_list if 0<=l<12])
        old=len([l for l in age_list if 60<=l<150])
        men=len([l for l in sex_list if l==1])
        woman=len([l for l in sex_list if l==2])
        members={'adults':adult,'youngs':young,'kids':kid,'olds':old,'males':men,'women':woman}    
        return members
    
    @staticmethod
    def get_route(element,type_road):
        object_id=str(int(list(people_to_evacuate.loc[people_to_evacuate['House ID']==element]['OBJECTID'])[0]))
        # route=type_road[str(object_id)][0] #ESTO LO DEBO ACTIVAR CUANDO CARGUE LAS NUEVAS RUTAS
        route=type_road[str(object_id)][0]
        return(route)

    @staticmethod
    def get_velocity(members):
        kids=members['kids']
        adults=members['adults']+members['youngs']+members['olds']
        total_person=kids+adults
        velocity=((kids*1.5)+(adults*1.3))/total_person
        return(velocity)

    @classmethod
    def builder_families(cls,env,type_road,S):
        house_id=list(OrderedDict.fromkeys(people_to_evacuate['House ID'])) #list of house_id
        for element in house_id[:1]:
            members=Family.get_members(element)
            housing=element
            route_home_mt=Family.get_route(element,type_road)
            velocity=Family.get_velocity(members)
            start_scape=S.generate_startscape_rand(members)
            Family.families.append(Family(env,members,housing,start_scape,velocity,route_home_mt))


    def evacuate(self):
        ################
        # Salen de sus casas
        ################
        print("Tiempo de simulacion: "+str(self.env.now))
        print('Familia '+str(self.ID)+' inicia escape')
        yield self.env.timeout(self.start_scape)  
        
        while True:

            ################
            # Inician una calle
            ################
            print("Tiempo de simulacion: "+str(self.env.now))
            id_to_search=self.route.pop(0)
            street_find = next(filter(lambda x: x.ID == id_to_search, Street.streets))
            print('Familia '+str(self.ID)+' llega a calle '+str(street_find.ID)+' en tiempo '+str(self.env.now))
            if len(self.route)>0:
                street_find.flow+=1
                yield self.env.timeout(street_find.velocity)
                street_find.flow-=1

            else:
                print("FIN")
                break      



class Street(object):
    streets=[]

    def __init__(self,ID,height):
        self.ID=ID
        self.flow=0
        self.velocity=3.85 #Fast running speed
        self.height=height

    @staticmethod
    def get_height(ID):
        try:
            u=streets.loc[streets['id']==str(ID)]['u'].item()  
            v=streets.loc[streets['id']==str(ID)]['v'].item()
            u_height=int(nodes.loc[nodes['id']==str(u)]['Dem_2'].item())
            v_height=int(nodes.loc[nodes['id']==str(v)]['Dem_2'].item())
            return(abs(u_height-v_height)) 
        except:
            print("Que he fallao:"+str(ID))      

    @classmethod
    def builder_streets(cls):
        street_id=list(streets['id'])
        contador=0
        control=1000
        for element in street_id:
            ID=element
            height=Street.get_height(ID)
            Street.streets.append(Street(ID,height))
            contador+=1
            if contador==control:
                print("Faltan "+str(len(street_id)-contador)+' para que empieze la simulacion')
                control+=1000


'''
class Building(object):
    buildings=[]
    ID=0

    def __init__(self,height):
        self.ID=Building.ID
        Building.ID+=1
        self.height=height
        self.capacity=height*10
        self.num_family=0 
        self.state='open'
    
    @classmethod
    def builder_building(cls):
'''

class Streams(object):
    def __init__(self,startscape_seed):
        self.startscape_rand=RandomState()
        self.startscape_rand.seed(startscape_seed)

    def generate_startscape_rand(self,members):
        if members['kids']==0 and members['olds']==0:            
            stratscape_vals=np.arange(2,10)
            startscape_prob= (0.2,0.3,0.3,0.15,0.05,0.0,0.0,0.0)
        elif members['kids']>0 and members['olds']==0:            
            stratscape_vals=np.arange(2,10)
            startscape_prob= (0.0,0.1,0.15,0.30,0.3,0.15,0.0,0.0)  
        elif members['kids']==0 and members['olds']>0:            
            stratscape_vals=np.arange(2,10)
            startscape_prob= (0.0,0.0,0.0,0.1,0.3,0.3,0.15,0.15)  
        else:            
            stratscape_vals=np.arange(2,10)
            startscape_prob= (0.0,0.0,0.0,0.0,0.2,0.3,0.3,0.2)                  
        return(self.startscape_rand.choice(stratscape_vals,p=startscape_prob))   


class Model(object):
    def __init__(self, seeds,simulation_time):
        self.startscape_seed=seeds[0]
        self.simulation_time=simulation_time

    def run(self):
        env=simpy.Environment()
        S = Streams(self.startscape_seed)
        Family.builder_families(env,home_to_mt_load,S)
        Street.builder_streets()
        env.run(until=self.simulation_time)



if __name__ == '__main__':
    #Cargo datos
    persons_data = pd.read_csv("data/personas_antofagasta.csv")
    synthetic_population=pd.read_csv('data/synthetic_population.csv')
    houses_to_evacuate=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Individual_Houses/House_to_evacuate/Houses_to_evacuate.shp')
    houses_to_evacuate.OBJECTID=houses_to_evacuate.OBJECTID.astype(int)
    #ID mayor a 2219 en nodos es un edificio!!!!!
    people_to_evacuate=synthetic_population.merge(houses_to_evacuate,how='left',left_on='ObjectID',right_on='OBJECTID')
    people_to_evacuate=people_to_evacuate.dropna(subset=['OBJECTID'])
    streets=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Corrected_Road_Network/Antofa_nodes_cut_edges/Antofa_edges.shp')
    nodes=gpd.read_file('data/nodos_con_altura/Antofa_nodes_altura.shp')
    #ID mayor a 4439 en streets es una calle de edificio!!!!!
    home_to_mt_load = np.load('data/caminos/home_to_mt.npy').item()
    home_to_bd_load = np.load('data/caminos/home_to_bd.npy').item()
    bd_to_mt_load = np.load('data/caminos/bd_to_mt.npy').item()

     
    seeds = list(zip(*3*[iter([i for i in range(1*3)])]))
    simulation_time=100
    Model(seeds,simulation_time).run()


#TESTEO

# def car(env):
#     while True:
#         print('Start parking at %d' % env.now)
#         parking_duration = 5
#         yield env.timeout(parking_duration)

#         print('Start driving at %d' % env.now)
#         trip_duration = 2
#         yield env.timeout(trip_duration)
#         if env.now == 14:
#             print("hola")
#             yield env.timeout(time_simulation-env.now)

# env = simpy.Environment()
# env.process(car(env))
# time_simulation=20  
# env.run(until=time_simulation)      


#Por agregar cuando al simulacion ya funcione una vez
'''
class Replicator(object):
    def __init__(self, seeds):
        self.seeds=seeds

    def run(self,params):
        return [Model(seeds,*params).run() for seeds in self.seeds], params   

class Experiment(object):
    def __init__(self,num_replics,scenarios=4):
        self.seeds = list(zip(*3*[iter([i for i in range(num_replics*3)])]))
        self.scenarios = scenarios
    
    def run(self):
        cpu = mp.cpu_count()
        self.results = Parallel(n_jobs=cpu, verbose=5)(delayed(Replicator(self.seeds).run)(scenario) for scenario in self.scenarios)    

'''


