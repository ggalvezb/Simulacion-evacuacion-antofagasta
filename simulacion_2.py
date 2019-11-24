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

from sklearn.externals.joblib import Parallel, delayed
import multiprocessing as mp

class Family(object):
    ID=0
    families=[]
    def __init__(self, env, members, housing, start_scape, velocity, route,meating_point,scenario):
        self.ID=Family.ID
        Family.ID+=1                    
        self.members = members          
        self.housing = housing           
        self.start_scape = start_scape  
        self.velocity = velocity                
        self.route = route              
        self.env=env
        self.meating_point=meating_point
        self.building=0
        self.scenario=scenario

        #Create the env for the family
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
    def get_route_length(route):
        route_length=0
        for street in route:
            street_find = next(filter(lambda x: x.ID == street, Street.streets))
            route_length+=street_find.lenght
        return(route_length)    


    @staticmethod
    def get_route(element,type_road,scenario):
        if scenario=='scenario 1':
            object_id=str(int(list(people_to_evacuate.loc[people_to_evacuate['House ID']==element]['OBJECTID'])[0]))
            route=type_road[str(object_id)][0].copy()
            meating_point=(int(type_road[str(object_id)][1]),'MP')
        elif scenario=='scenario 2':
            object_id=str(int(list(people_to_evacuate.loc[people_to_evacuate['House ID']==element]['OBJECTID'])[0]))
            route_to_mt=home_to_mt_load[str(object_id)][0]
            length_route_to_mt=Family.get_route_length(route_to_mt)
            meating_point=int(home_to_mt_load[str(object_id)][1]) 
            route_to_bd=home_to_bd_load[str(object_id)][0]
            length_route_to_bd=Family.get_route_length(route_to_bd)
            building=int(home_to_bd_load[str(object_id)][1])
            prob_go_bd=length_route_to_mt/(length_route_to_mt+length_route_to_bd)
            prob_go_mt=length_route_to_bd/(length_route_to_mt+length_route_to_bd)
            # print("MT: "+str(prob_go_mt)+"  "+str(length_route_to_mt)+"    BD: "+str(prob_go_bd)+"  "+str(length_route_to_bd))
            route=np.random.choice([route_to_mt,route_to_bd],p=[prob_go_mt,prob_go_bd])
            if route==route_to_mt: meating_point=(meating_point,'MP') 
            elif route==route_to_bd:meating_point=(building,'BD')
        return(route,meating_point)

    @staticmethod
    def get_velocity(members):
        kids=members['kids']
        adults=members['adults']+members['youngs']
        olds=members['olds']
        total_person=kids+adults
        velocity=((kids*1.3)+(adults*1.5)+(olds*0.948))/total_person
        return(velocity)


    @classmethod
    def builder_families(cls,env,type_road,S,scenario):
        house_id=list(OrderedDict.fromkeys(people_to_evacuate['House ID'])) #list of house_id
        for element in house_id[:5]:
            members=Family.get_members(element)
            housing=list(people_to_evacuate.loc[people_to_evacuate['House ID']==element]['ObjectID'])[0]
            route,meating_point=Family.get_route(element,type_road,scenario)
            velocity=Family.get_velocity(members)
            start_scape=S.generate_startscape_rand(members)
            Family.families.append(Family(env,members,housing,start_scape,velocity,route,meating_point,scenario))
    
    @classmethod
    def reset_class(cls):
        cls.ID=0
        cls.families=[]

    def evacuate(self):
        ################
        # Salen de sus casas
        ################
        yield self.env.timeout(self.start_scape)  

        while True:
            ################
            # Inician una calle
            ################
            id_to_search=self.route.pop(0)
            street_find = next(filter(lambda x: x.ID == id_to_search, Street.streets))
            street_find.flow+=1
            if street_find.flow>street_find.capacity: street_find.velocity=0.751 
            velocity=min(street_find.velocity,self.velocity)
            yield self.env.timeout(velocity*street_find.lenght)
            street_find.flow-=1
            if len(self.route)==0: #Final de ruta
                if self.meating_point[1]=='MP': #Llega a punto de encuentro
                    print('FAMILIA  '+str(self.ID)+' TERMINA EVACUACIÓN Y LLEGAN A PUNTO DE ENCUENTRO '+str(self.meating_point  ))
                    id_to_search=self.meating_point[0]    
                    meatingpoint_find = next(filter(lambda x: x.ID == id_to_search, MeatingPoint.meating_points))
                    new_members=dict(Counter(meatingpoint_find.members)+Counter(self.members))
                    meatingpoint_find.members=new_members
                    meatingpoint_find.persons+=self.members['males']+self.members['women']
                    break

                elif self.meating_point[1]=='BD': #Llega a edificio
                    id_to_search=self.meating_point
                    building_search=next(filter(lambda x: x.ID == id_to_search, Building.buildings))
                    print('FAMILIA '+str(self.ID)+' LLEGAN A EDIFICO '+str(building_search.ID)+' Y ESTE SE ENCUENTRA '+str(building_search.state)+' EN TIEMPO '+str(self.env.now))
                    if building_search.state == 'open':
                        building_search.num_family+=1
                        building_search.capacity-=self.members['males']+self.members['women']
                        if building_search.capacity<=0: building_search.state='close'
                    else:
                        ##########
                        # Si el edificio esta cerrado se van a un punto de encuentro
                        ##########
                        self.route=bd_to_mt_load[str(self.housing)][0]
                        self.meating_point=bd_to_mt_load[str(self.housing)][1]
                        while True:
                            ##########
                            # Vuelven a calle
                            ##########
                            id_to_search=self.route.pop(0)
                            street_find = next(filter(lambda x: x.ID == id_to_search, Street.streets))
                            street_find.flow+=1
                            if street_find.flow>street_find.capacity: street_find.velocity=0.751 
                            velocity=min(street_find.velocity,self.velocity)
                            yield self.env.timeout(velocity*street_find.lenght)
                            street_find.flow-=1
                            if len(self.route)==0:
                                ###########
                                # Llegan a un punto de encuentro
                                ###########
                                print('FAMILIA  '+str(self.ID)+' TERMINA EVACUACIÓN Y LLEGAN A PUNTO DE ENCUENTRO '+str(self.meating_point)+' EN TIEMPO '+str(self.env.now))
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

    @classmethod
    def reset_class(cls):
        cls.streets=[]   
               

class Building(object):
    buildings=[]

    def __init__(self,ID,height):
        self.ID=ID
        self.height=height
        self.capacity=height*10
        self.num_family=0 
        self.state='open'
    
    @classmethod
    def builder_building(cls):
        for element in buildings['fid']:
            ID=element
            height=int(buildings.loc[buildings['fid']==element]['Base'].item())
            Building.buildings.append(Building(ID,height))

    @classmethod
    def reset_class(cls):
        cls.buildings=[]        

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

    @classmethod
    def reset_class(cls):
        cls.meating_points=[]        

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
        if self.scenario=='scenario 1': route_scenario=home_to_mt_load
        elif self.scenario=='scenario 2': route_scenario=home_to_bd_load
        env=simpy.Environment()
        S = Streams(self.startscape_seed)
        Street.builder_streets()
        Family.builder_families(env,route_scenario,S,self.scenario)
        # sys.exit()
        Building.builder_building()
        MeatingPoint.builder_Meatinpoint()
        
        env.run()
        #Termino la replica y reinicio las clases
        # Family.reset_class()
        # Street.reset_class()
        # Building.reset_class()
        # MeatingPoint.reset_class()

class Replicator(object):
    def __init__(self, seeds):
        self.seeds=seeds

    def run(self,params):
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
    buildings=gpd.read_file('data/edificios/Edificios_zona_inundacion.shp')
    meating_points=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Tsunami/Puntos_Encuentro/Puntos_Encuentro_Antofagasta/puntos_de_encuentro.shp')



    time=500
    scenarios=[('scenario 2',time)]
    # scenarios = [('scenario 1',time),('scenario 2',time)]
    exp = Experiment(1,scenarios)
    exp.run()






