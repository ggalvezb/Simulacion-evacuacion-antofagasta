import itertools
import pandas as pd
from collections import OrderedDict
from collections import Counter
import numpy as np
import geopandas as gpd
from numpy.random import RandomState
import simpy
import sys
import time   #Para probar los tiempos de ejecucion
import os

#Para el modelo de optimizacion
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError

#Para paralelizar
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing as mp

#Para crear grafos y obtener camino minimo
import igraph

inicio=time.time()
class Family(object):
    ID=0
    families=[]

    family_statistics=[]
    def __init__(self, members, housing, velocity, route,meating_point,scenario,route_lenght,geometry,people_for_stats):
        self.ID=Family.ID
        Family.ID+=1                    
        self.members = members          
        self.housing = housing           
        self.start_scape = None  
        self.velocity = velocity                
        self.route = route   
        self.route_lenght=route_lenght           
        self.env=None
        self.meating_point=meating_point
        self.scenario=scenario
        self.geometry=geometry

        #Stats
        self.family_stats={}
        self.path=[]
        self.delays=0
        self.people=people_for_stats

    @staticmethod
    def get_members(element):
        age_list=list(synthetic_population.loc[synthetic_population['House ID']==element].Age)
        sex_list=list(synthetic_population.loc[synthetic_population['House ID']==element].Sex)
        people_for_stats=[{'Age':x,'Sex':y}for (x,y) in zip(age_list,sex_list)]
        adult=len([l for l in age_list if 18<=l<60])
        young=len([l for l in age_list if 12<=l<18])
        kid=len([l for l in age_list if 0<=l<12])
        old=len([l for l in age_list if 60<=l<150])
        men=len([l for l in sex_list if l==1])
        woman=len([l for l in sex_list if l==2])
        members={'adults':adult,'youngs':young,'kids':kid,'olds':old,'males':men,'women':woman}    
        return members,people_for_stats

    @staticmethod
    def get_route_length(route):
        route_length=0
        for street in route:
            street_find = next(filter(lambda x: x.ID == street, Street.streets))
            route_length+=street_find.lenght
        return(route_length)    

    @staticmethod
    def get_route(element,type_road,scenario,house_df):
        if scenario=='scenario 1':
            object_id=str(int(list(house_df['OBJECTID'])[0]))
            route=type_road[str(object_id)][0].copy()
            length_route=Family.get_route_length(route)
            meating_point=(int(type_road[str(object_id)][1]),'MP')

        elif scenario=='scenario 2':
            object_id=str(int(list(house_df['OBJECTID'])[0]))
            route_to_mt=home_to_mt_load[str(object_id)][0]
            length_route_to_mt=Family.get_route_length(route_to_mt)
            meating_point=int(home_to_mt_load[str(object_id)][1]) 
            route_to_bd=home_to_bd_load[str(object_id)][0]
            length_route_to_bd=Family.get_route_length(route_to_bd)
            building=int(home_to_bd_load[str(object_id)][1])
            prob_go_bd=length_route_to_mt/(length_route_to_mt+length_route_to_bd)
            prob_go_mt=length_route_to_bd/(length_route_to_mt+length_route_to_bd)
            if prob_go_bd>=0.85:
                route=route_to_bd
                meating_point=(building,'BD')
                length_route=length_route_to_bd
            elif prob_go_mt>=0.85:
                route=route_to_mt
                meating_point=(meating_point,'MP')
                length_route=length_route_to_mt
            else: 
                route=np.random.choice([route_to_mt,route_to_bd],p=[prob_go_mt,prob_go_bd])
                if route==route_to_mt:
                    print("A mp")
                    meating_point=(meating_point,'MP')
                    length_route=length_route_to_mt 
                elif route==route_to_bd:
                    print("A edificio")
                    meating_point=(building,'BD')
                    length_route=length_route_to_bd

        elif scenario=='scenario 3':
            object_id=str(int(list(house_df['OBJECTID'])[0]))
            if int(object_id) in optimal_scape.keys():
                route=optimal_scape[int(object_id)][0]
                length_route=Family.get_route_length(route)
                building=int(optimal_scape[int(object_id)][1])
                if int(optimal_scape[int(object_id)][1])<150:
                    meating_point=(building,'BD')
                else:
                    meating_point=(building,'MP')
            else:
                route=home_to_mt_load[str(object_id)][0]
                length_route=Family.get_route_length(route)
                building=int(home_to_mt_load[str(object_id)][1])
                meating_point=(building,'MP')
        return(route,meating_point,length_route)
  
    @staticmethod
    def get_velocity(members):
        kids=members['kids']
        adults=members['adults']+members['youngs']
        olds=members['olds']
        total_person=kids+adults+olds
        velocity=((kids*1.3)+(adults*1.5)+(olds*0.948))/total_person
        return(velocity)

    def streets_statistics(self,id_to_search,velocity):
        street_dict={'ID':id_to_search,'Velocity':velocity}
        self.path.append(street_dict)

    @classmethod
    def builder_families(cls,type_road,scenario):
        house_id=list(OrderedDict.fromkeys(people_to_evacuate['House ID'])) #list of house_id
        start=time.time()
        for element in house_id[0:1]:
            members,people_for_stats=Family.get_members(element)
            house_df=people_to_evacuate.loc[people_to_evacuate['House ID']==element]
            housing=list(house_df['ObjectID'])[0]
            geometry=list(house_df['geometry'])[0]
            route,meating_point,length_route=Family.get_route(element,type_road,scenario,house_df)
            velocity=Family.get_velocity(members)
            Family.families.append(Family(members,housing,velocity,route,meating_point,scenario,length_route,geometry,people_for_stats))
        print("fin construir familias ", (time.time())-start)

    def evacuate(self):
        route_copy=self.route.copy()
        ################
        # Salen de sus casas
        ################
        self.delays=self.start_scape
        yield self.env.timeout(self.start_scape)  

        while True:
            ################
            # Inician una calle
            ################
            if len(route_copy)!=0:
                id_to_search=route_copy.pop(0)
                street_find = next(filter(lambda x: x.ID == id_to_search, Street.streets))
                street_find.flow+=1
                if street_find.flow>street_find.capacity: street_find.velocity=0.751 
                velocity=min(street_find.velocity,self.velocity)
                # print("Velocidad en m/s: ",velocity)
                # print("Largo de calle: ",street_find.lenght)
                # print("Tiempo de viaje en la calle: ",(street_find.lenght/velocity))
                Family.streets_statistics(self,id_to_search,street_find.lenght/velocity)
                yield self.env.timeout(street_find.lenght/velocity)
                street_find.flow-=1
            if len(route_copy)==0: #Final de ruta
                if self.meating_point[1]=='MP': #Llega a punto de encuentro
                    print('FAMILIA  '+str(self.ID)+' TERMINA EVACUACIÓN Y LLEGAN A PUNTO DE ENCUENTRO '+str(self.meating_point)+'EN TIEMPO '+str(self.env.now))
                    id_to_search=self.meating_point[0]    
                    meatingpoint_find = next(filter(lambda x: x.ID == id_to_search, MeatingPoint.meating_points))
                    new_members=dict(Counter(meatingpoint_find.members)+Counter(self.members))
                    meatingpoint_find.members=new_members
                    meatingpoint_find.persons+=self.members['males']+self.members['women']
                    break

                elif self.meating_point[1]=='BD': #Llega a edificio
                    id_to_search=self.meating_point[0]
                    building_search=next(filter(lambda x: x.ID == id_to_search, Building.buildings))
                    print('FAMILIA '+str(self.ID)+' LLEGAN A EDIFICO '+str(building_search.ID)+' Y ESTE SE ENCUENTRA '+str(building_search.state)+' EN TIEMPO '+str(self.env.now))
                    if building_search.state == 'open':
                        building_search.num_family+=1
                        building_search.capacity-=self.members['males']+self.members['women']
                        new_members=dict(Counter(building_search.members)+Counter(self.members))
                        building_search.members=new_members
                        if building_search.capacity<=0: building_search.state='close'
                    else:
                        ##########
                        # Si el edificio esta cerrado se van a un punto de encuentro
                        ##########
                        route_copy=bd_to_mt_load[str(self.housing)][0]
                        self.meating_point=bd_to_mt_load[str(self.housing)][1]
                        while True:
                            ##########
                            # Vuelven a calle
                            ##########
                            id_to_search=route_copy.pop(0)
                            street_find = next(filter(lambda x: x.ID == id_to_search, Street.streets))
                            street_find.flow+=1
                            if street_find.flow>street_find.capacity: street_find.velocity=0.751 
                            velocity=min(street_find.velocity,self.velocity)
                            Family.streets_statistics(self,id_to_search,street_find.lenght/velocity)
                            yield self.env.timeout(street_find.lenght/velocity)
                            street_find.flow-=1
                            if len(route_copy)==0:
                                ###########
                                # Llegan a un punto de encuentro
                                ###########
                                print('esto no deberia pasar FAMILIA  '+str(self.ID)+' TERMINA EVACUACIÓN Y LLEGAN A PUNTO DE ENCUENTRO '+str(self.meating_point)+' EN TIEMPO '+str(self.env.now))
                                id_to_search=self.meating_point    
                                meatingpoint_find = next(filter(lambda x: x.ID == id_to_search, MeatingPoint.meating_points))
                                new_members=dict(Counter(meatingpoint_find.members)+Counter(self.members))
                                meatingpoint_find.members=new_members
                                meatingpoint_find.persons+=self.members['males']+self.members['women']
                                break
                    break


class Street(object):
    streets=[]

    def __init__(self,ID,height,type_street,lenght,capacity,velocity):
        self.ID=ID
        self.flow=0
        self.velocity=velocity
        self.height=height
        self.type=type_street
        self.lenght=lenght
        self.capacity=int(capacity)  #Si se supera este valor se considera atochado y la calle baja su velocidad a 0.751 m/s

    @staticmethod
    def get_capacity(type_street,lenght):
        if type_street=='residential': width=4 
        elif type_street=='primary': width=8
        elif type_street=='tertiary': width=2
        else: width=4
        area=width*lenght
        return(area*1.55)  #Se considera que con 1.55 personas por m2 se puede transitar libremente

    @staticmethod
    def get_velocity(height):
        if height<5.6: velocity=3.85
        elif 5.6<height<=8: velocity=0.91
        elif 8<height<=11.2: velocity=0.76
        elif 11.2<height<=14: velocity=0.60
        elif 14<height<=30: velocity=0.31
        elif 30<=height: velocity=0.02
        return(velocity)

    @classmethod
    def builder_streets(cls):
        street_id=list(streets['id'])
        contador=0
        control=1000
        for i in range(len(streets)):
            ID=streets.loc[i]['id']
            height=streets.loc[i]['height']
            type_street=streets.loc[i]['highway']
            lenght=streets.loc[i]['length']
            capacity=Street.get_capacity(type_street,lenght)
            velocity=Street.get_velocity(height)
            Street.streets.append(Street(ID,height,type_street,lenght,capacity,velocity))
            contador+=1
            if contador==control:
                print("Faltan "+str(len(street_id)-contador)+' para que empieze la simulacion')
                control+=1000

class Building(object):
    buildings=[]

    def __init__(self,ID,height,geometry):
        self.ID=ID
        self.height=height
        self.capacity=(height/3)*5
        self.num_family=0 
        self.state='open'
        self.geometry=geometry
        self.members={'adults':0,'youngs':0,'kids':0,'olds':0,'males':0,'women':0}

    
    @classmethod
    def builder_building(cls):
        for element in buildings['fid']:
            ID=int(element)
            building=buildings.loc[buildings['fid']==element]
            height=int(building['Base'].item())
            geometry=building['geometry'].item()
            Building.buildings.append(Building(ID,height,geometry))
     
class MeatingPoint(object):
    meating_points=[]

    def __init__(self,ID):
        self.ID=ID 
        self.members={'adults':0,'youngs':0,'kids':0,'olds':0,'males':0,'women':0}
        self.persons=0

    @classmethod
    def builder_Meatinpoint(cls):
        for i in range(len(meating_points)):
            ID=meating_points.loc[i].OBJECTID
            MeatingPoint.meating_points.append(MeatingPoint(ID))

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
    def __init__(self, seeds,scenario,simulation_time):
        self.startscape_seed=seeds
        print("seed 2: ",self.startscape_seed)
        self.simulation_time=simulation_time
        self.scenario=scenario

    def run(self):
        S = Streams(self.startscape_seed)
        env=simpy.Environment()
        for family in Family.families:
            family.start_scape=S.generate_startscape_rand(family.members)
            family.env=env
            family.env.process(family.evacuate())

        for building in Building.buildings:
            building.capacity=(building.height/3)*5
            building.num_family=0 
        env.run()

class Replicator(object):
    def __init__(self, seeds):
        self.seeds=seeds

    def run(self,params):
        scenario=params[0]
        if scenario=='scenario 1': route_scenario=home_to_mt_load
        elif scenario=='scenario 2': route_scenario=home_to_bd_load
        else: route_scenario=optimal_scape
        Street.builder_streets()
        Building.builder_building()
        MeatingPoint.builder_Meatinpoint()
        print("EMPIEZA CONSTRUCCION DE FAMILIA")
        Family.builder_families(route_scenario,scenario)
        print("LARGO DE FAMILIAS {} DE CALLES {} EDIFICIOS {} Y MP {}".format(len(Family.families),len(Street.streets),len(Building.buildings),len(MeatingPoint.meating_points)))

        # return [Model(seeds,*params).run() for seeds in self.seeds], params
        return [Model(seeds,*params).run() for seeds in self.seeds]

class Experiment(object):
    def __init__(self,num_replics,scenarios):
        self.seeds = list(zip(*3*[iter([i for i in range(num_replics*3)])]))
        self.scenarios = scenarios
    
    def run(self):
        cpu = mp.cpu_count()
        # self.results = Parallel(n_jobs=cpu, verbose=5)(delayed(Replicator(self.seeds).run)(scenario) for scenario in self.scenarios)
        for scenario in self.scenarios:
            Replicator(self.seeds).run(scenario)
            Family.families=[]
            Street.streets=[]
            Building.buildings=[]
            MeatingPoint.meating_points=[]
            

if __name__ == '__main__':
    #Cargo datos
    directory=os.getcwd()
    persons_data = pd.read_csv("data/personas_antofagasta.csv")
    synthetic_population=pd.read_csv('data/synthetic_population.csv')
    houses_to_evacuate=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Individual_Houses/House_to_evacuate/Houses_to_evacuate.shp')
    houses_to_evacuate.OBJECTID=houses_to_evacuate.OBJECTID.astype(int)
    #ID mayor a 2219 en nodos es un edificio!!!!!
    people_to_evacuate=synthetic_population.merge(houses_to_evacuate,how='left',left_on='ObjectID',right_on='OBJECTID')
    people_to_evacuate=people_to_evacuate.dropna(subset=['OBJECTID'])
    streets=gpd.read_file('data/calles_con_delta_altura/calles_delta_altura.shp')
    nodes=gpd.read_file('data/nodos_con_altura/Antofa_nodes_altura.shp')
    #ID mayor a 4439 en streets es una calle de edificio!!!!!
    home_to_mt_load = np.load('data/caminos/home_to_mt.npy').item()
    home_to_bd_load = np.load('data/caminos/home_to_bd.npy').item()
    bd_to_mt_load = np.load('data/caminos/bd_to_mt.npy').item()
    optimal_scape=np.load('data/scape_route_optimal.npy').item()
    buildings=gpd.read_file('data/edificios/Edificios_zona_inundacion.shp')
    meating_points=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Tsunami/Puntos_Encuentro/Puntos_Encuentro_Antofagasta/puntos_de_encuentro.shp')
    nodes_without_buildings=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Corrected_Road_Network/Antofa_nodes_cut_edges/sin_edificios/Antofa_nodes.shp')

    time_sim=500
    scenarios=[('scenario 2',time_sim)]
    # scenarios = [('scenario 1',time),('scenario 2',time)]
    exp = Experiment(4,scenarios)
    exp.run()

print("TERMINO")
