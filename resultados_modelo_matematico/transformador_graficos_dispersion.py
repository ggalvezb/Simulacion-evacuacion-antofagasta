import pandas as pd 
import geopandas as gpd 
import numpy as np

#Niños
optimal_scape=np.load('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\scape_route_optimal_ninos_primero.npy').item()
houses_to_evacuate=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Individual_Houses/House_to_evacuate/Houses_to_evacuate.shp')

evacuacion=[] #En esta lista acumulare valores binarios por cada familia. 1 si es evacuada a edificio, 0 en otro caso
for i in range(len(houses_to_evacuate)):
    object_id=str(int(list(houses_to_evacuate['OBJECTID'])[i]))
    if int(object_id) in optimal_scape.keys():
        if int(optimal_scape[int(object_id)][1])<150:
            evacuacion.append(1)
        else:
            evacuacion.append(0)
    else:
        evacuacion.append(0)
houses_to_evacuate['Evacuacion_binaria']=evacuacion

houses_to_evacuate.to_file("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\shape_dist_ninos\\dist_ninos_edificios.shp")


#Abuelos
optimal_scape=np.load('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\scape_route_optimal_abuelos_primero.npy').item()
houses_to_evacuate=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Individual_Houses/House_to_evacuate/Houses_to_evacuate.shp')

evacuacion=[] #En esta lista acumulare valores binarios por cada familia. 1 si es evacuada a edificio, 0 en otro caso
for i in range(len(houses_to_evacuate)):
    object_id=str(int(list(houses_to_evacuate['OBJECTID'])[i]))
    if int(object_id) in optimal_scape.keys():
        if int(optimal_scape[int(object_id)][1])<150:
            evacuacion.append(1)
        else:
            evacuacion.append(0)
    else:
        evacuacion.append(0)
houses_to_evacuate['Evacuacion_binaria']=evacuacion

houses_to_evacuate.to_file("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\shape_dist_abuelos\\dist_abuelos_edificios.shp")





