import pandas as pd 
import geopandas as gpd 

data=gpd.read_file("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\shape_dist_ninos\\dist_ninos_edificios.shp")
data.to_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados_modelo_matematico\\shape_dist_ninos\\dist_ninos_edificios.csv")