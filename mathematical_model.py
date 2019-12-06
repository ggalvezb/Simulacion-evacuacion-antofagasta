from simulacion_2 import Family
import simpy
import pandas as pd
import geopandas as gpd
import numpy as np
from collections import OrderedDict
from collections import Counter
import time
import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import igraph
import sys

#Cargo datos
persons_data = pd.read_csv("data/personas_antofagasta.csv")
synthetic_population=pd.read_csv('data/synthetic_population.csv')
houses_to_evacuate=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Individual_Houses/House_to_evacuate/Houses_to_evacuate.shp')
houses_to_evacuate.OBJECTID=houses_to_evacuate.OBJECTID.astype(int)
#ID mayor a 2219 en nodos es un edificio!!!!!
people_to_evacuate=synthetic_population.merge(houses_to_evacuate,how='left',left_on='ObjectID',right_on='OBJECTID')
people_to_evacuate=people_to_evacuate.dropna(subset=['OBJECTID'])
# streets=gpd.read_file('data/calles_con_delta_altura/calles_delta_altura.shp')
streets=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Corrected_Road_Network/Antofa_nodes_cut_edges/Antofa_edges.shp')
nodes=gpd.read_file('data/nodos_con_altura/Antofa_nodes_altura.shp')
#ID mayor a 4439 en streets es una calle de edificio!!!!!
home_to_mt_load = np.load('data/caminos/home_to_mt.npy').item()
home_to_bd_load = np.load('data/caminos/home_to_bd.npy').item()
bd_to_mt_load = np.load('data/caminos/bd_to_mt.npy').item()
buildings=gpd.read_file('data/edificios/Edificios_zona_inundacion.shp')
meating_points=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Tsunami/Puntos_Encuentro/Puntos_Encuentro_Antofagasta/puntos_de_encuentro.shp')
nodes_without_buildings=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Corrected_Road_Network/Antofa_nodes_cut_edges/sin_edificios/Antofa_nodes.shp')

class Family(object):
    ID=0
    families=[]
    def __init__(self, members, housing, route,meating_point,route_lenght,geometry):
        self.ID=Family.ID
        Family.ID+=1                    
        self.members = members          
        self.housing = housing           
        self.route = route   
        self.route_lenght=route_lenght           
        self.meating_point=meating_point
        self.geometry=geometry

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
    def get_route(element,house_df):
        object_id=str(int(list(house_df['OBJECTID'])[0]))
        route=home_to_bd_load[str(object_id)][0]
        length_route=Family.get_route_length(route)
        building=int(home_to_bd_load[str(object_id)][1])
        meating_point=(building,'BD')
        return(route,meating_point,length_route)

    @classmethod
    def builder_families(cls):
        house_id=list(OrderedDict.fromkeys(people_to_evacuate['House ID'])) #list of house_id
        start=time.time()
        for element in house_id:
            members=Family.get_members(element)
            house_df=people_to_evacuate.loc[people_to_evacuate['House ID']==element]
            housing=list(house_df['ObjectID'])[0]
            geometry=list(house_df['geometry'])[0]
            route,meating_point,length_route=Family.get_route(element,house_df)
            Family.families.append(Family(members,housing,route,meating_point,length_route,geometry))
        print("Termina familias con tiempo ", (time.time())-start)

    @classmethod
    def reset_class(cls):
        cls.ID=0
        cls.families=[]            

class Street(object):
    streets=[]

    def __init__(self,ID,lenght):
        self.ID=ID
        self.lenght=lenght

    @classmethod
    def builder_streets(cls):
        street_id=list(streets['id'])
        contador=0
        control=1000
        for i in range(len(streets)):
            ID=streets.loc[i]['id']
            lenght=streets.loc[i]['length']
            Street.streets.append(Street(ID,lenght))
            contador+=1
            if contador==control:
                print("Faltan "+str(len(street_id)-contador)+' para que empieze la simulacion')
                control+=1000

class Building(object):
    buildings=[]

    def __init__(self,ID,height,geometry,building_type):
        self.ID=ID
        self.height=height
        self.capacity=(height/3)*5
        self.geometry=geometry
        self.building_type=building_type
    
    @classmethod
    def builder_building(cls):
        for element in buildings['fid']:
            ID=int(element)
            building=buildings.loc[buildings['fid']==element]
            height=int(building['Base'].item())
            geometry=building['geometry'].item()
            building_type='Building'
            Building.buildings.append(Building(ID,height,geometry,building_type))
        for element in meating_points['OBJECTID']:
            ID=int(element)
            meating_point=meating_points[meating_points['OBJECTID']==element]
            height=9999
            geometry=meating_point.geometry
            building_type="Meating point"
            Building.buildings.append(Building(ID,height,geometry,building_type))

    @classmethod
    def reset_class(cls):
        cls.buildings=[]        


Street.builder_streets()
print("Termina calles")
Building.builder_building()
Family.builder_families()


#Creacion de grafo
g = igraph.Graph(directed = True)
g.add_vertices(list(nodes.id))
g.add_edges(list(zip(streets.u, streets.v)))
g.es['id']=list(streets['id'])
g.es['length']=list(streets['length'])

#Min distancia
def min_dist(point, gpd2):
    gpd2['Dist'] = gpd2.apply(lambda row:  point.distance(row.geometry),axis=1)
    geoseries = gpd2.iloc[gpd2['Dist'].idxmin()]
    return geoseries


def get_route_length(route):
    route_length=0
    for street in route:
        street_find = next(filter(lambda x: x.ID == street, Street.streets))
        route_length+=street_find.lenght
    if route_length ==0:
        route_length=9999999
    return(route_length)

def get_route(family_point,building_point):
    inicio_id=min_dist(family_point, nodes_without_buildings)['id']
    inicio_vertex=g.vs.find(name=str(inicio_id)).index
    fin_id_bd=min_dist(building_point, nodes)['id']
    fin_vertex_bd=g.vs.find(name=str(fin_id_bd)).index
    # shortest_path=g.get_shortest_paths(inicio_vertex, to=fin_vertex_bd, weights=g.es['length'], mode=igraph.OUT, output="epath")[0]
    # path_id=[]
    # for j in range(len(shortest_path)):
    #     path_id.append(g.es[shortest_path[j]]['id'])
    return(inicio_vertex,fin_vertex_bd)

def get_vertex(point):
    inicio_id=min_dist(point, nodes_without_buildings)['id']
    inicio_vertex=g.vs.find(name=str(inicio_id)).index
    return(inicio_vertex)



    
print("EMPIEZA MODELO OPTI")
#################
# Modelo de opti
# ############### 

#Parametros de las familias
start=time.time()
T_exec=30
olds_fam=[]
kids_fam=[]
id_fams=[]
building_distance=[]
num_members=[]
family_vertex=[]

i=0
for element in Family.families:
    if element.route_lenght<=500 and (element.members['olds']>0 or element.members['kids']>0):
        olds_fam.append(element.members['olds'])
        kids_fam.append(element.members['kids'])
        id_fams.append(element.housing)
        num_members.append(element.members['males']+element.members['women'])
        family_vertex.append(get_vertex(element.geometry))
        # for building in Building.buildings:
        #     building_distance.append(get_route_length(get_route(element.geometry,building.geometry)))
num_families=len(olds_fam)

#Parametros de los edificios/puntos de encuentro
id_buildings=[]
cap_bd=[]
building_vertex=[]
buildings[buildings['Base']==90]
# building_distance=pd.DataFrame()
for element in Building.buildings:
    print("Edificio id: ",element.ID)
    cap_bd.append(int(element.capacity))
    id_buildings.append(element.ID)
    building_vertex_temp=get_vertex(element.geometry)
    building_vertex.append(building_vertex_temp)
    # building_distance[element.ID]=g.shortest_paths_dijkstra(source=family_vertex,target=building_vertex_temp,weights=g.es['length'],mode=igraph.ALL)
num_buildings=len(cap_bd)

# #Arreglo de dataframe
# building_distance_2=pd.DataFrame()
# for columns in building_distance.columns:
#     distance=[element[0] for element in building_distance[columns]]
#     building_distance_2[columns]=distance


#Guardar distancias a edificio
# building_distance_2.to_csv('Distancias.csv')

#Cargar distancia a edificio
distances=pd.read_csv('Distancias.csv')
distances=distances.drop(['Unnamed: 0'],axis=1)
distances=distances.values



# building_distance=[get_route_length(get_route(family.geometry,building.geometry))for family in Family.families[:2] for building in Building.buildings[:10]]
# building_distance=np.resize(building_distance,(2,10))

len(Family.families)
print("Termina carga de datos del modelo y se demoro ",(time.time())-start)


####### Variables de decision ##########
Model=cplex.Cplex()
print("Empieza la creacion de variables ")
start=time.time()

x_vars = np.array([["x("+str(id_fams[i])+","+str(id_buildings[j])+")"  for j in range(0,num_buildings)] for i in range(0,num_families)])
x_varnames = x_vars.flatten()
x_vartypes = 'B'*len(x_varnames)
x_varlb = [0.0]*len(x_varnames)
x_varub = [1.0]*len(x_varnames)
x_varobj=[(distances[i,j])/(0.7*olds_fam[i]+0.3*kids_fam[i]) for j in range(num_buildings) for i in range(num_families)]

Model.variables.add(obj = x_varobj, lb = x_varlb, ub = x_varub, types = x_vartypes, names = x_varnames)
Model.objective.set_sense(Model.objective.sense.minimize)


####### Restricciones #######

for j in range(num_buildings):
    ind=[x_vars[i,j] for i in range(num_families)]
    val=[num_members[i] for i in range(num_families)]
    Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ind, val = val)], 
                                senses = ['L'], 
                                rhs = [cap_bd[j]])

for i in range(num_families):
    ind=[x_vars[i,j] for j in range(num_buildings)]
    val=[1.0 for j in range(num_buildings)]
    Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ind, val =val )], 
                                senses = ['E'], 
                                rhs = [1.0])


# for i in range(num_families):
#      for j in range(num_buildings):
#          Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=[x_vars[i,j]],val=[distances[i,j]])], 
#                                         senses =['L'], 
#                                         rhs = [500])   

Model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=[x_vars[i,j]],val=[distances[i,j]]) for i in range(num_families) for j in range(num_buildings)], 
                                        senses =['L'for i in range(num_families) for j in range(num_buildings)], 
                                        rhs = [895 for i in range(num_families) for j in range(num_buildings)])   



end=time.time()
print("Termina creacion de modelo con tiempo de ",(time.time())-start)

Model.parameters.timelimit.set(float(T_exec))
Model.parameters.workmem.set(9000.0)
print("EMPIEZA SOLVE")
Model.solve()

print("\nObjective Function Value = {}".format(Model.solution.get_objective_value()))


print("INICIA CREADOR DE RUTAS")
start=time.time()
#Creacion de rutas de escape
path={}
for i in range(0,num_families):
    for j in range(0,num_buildings):
        if(Model.solution.get_values("x("+str(id_fams[i])+","+str(id_buildings[j])+")")!=0.0):
            print("x("+str(id_fams[i])+","+str(id_buildings[j])+")"+" = "+str(Model.solution.get_values("x("+str(id_fams[i])+","+str(id_buildings[j])+")")))
            family_find = next(filter(lambda x: x.housing == id_fams[i], Family.families))
            building_find=next(filter(lambda x: x.ID==id_buildings[j],Building.buildings))
            inicio_id=min_dist(family_find.geometry, nodes_without_buildings)['id']
            inicio_vertex=g.vs.find(name=str(inicio_id)).index
            try:
                fin_id_bd=min_dist(building_find.geometry.item(), nodes)['id']
            except:
                fin_id_bd=min_dist(building_find.geometry, nodes)['id'] 
            fin_vertex_bd=g.vs.find(name=str(fin_id_bd)).index
            shortest_path=g.get_shortest_paths(inicio_vertex, to=fin_vertex_bd, weights=g.es['length'], mode=igraph.ALL, output="epath")[0]
            path_id=[]
            for z in range(len(shortest_path)):
                path_id.append(g.es[shortest_path[z]]['id'])
            path[id_fams[i]]=[path_id,id_buildings[j]]


#Guardar diccionario
np.save('scape_route_optimal.npy', path)

#cargar diccionario
optimal_scape=np.load('data/scape_route_optimal.npy').item()
optimal_scape[27350]

print("termina el creador de rutas en tiempo ",(time.time())-start)

