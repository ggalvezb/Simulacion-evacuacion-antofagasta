import pandas as pd 
import geopandas as gpd
from collections import Counter
from colour import Color


carpeta_fuente="C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\calles"

tiempo=10
tamaños=[]
for i in range(19):
    shape=gpd.read_file(carpeta_fuente+"\\calles de escenario scenario 3 replica 1 tiempo {}.shp".format(tiempo))
    contador_max=max(list(Counter(shape['Flow']).keys()))
    tamaños.append((contador_max,tiempo))
    contador=Counter(shape['Flow']).keys()
    tiempo+=200

shape=gpd.read_file(carpeta_fuente+"\\calles de escenario scenario 3 replica 1 tiempo {}.shp".format(1010))
contador_max=max(list(Counter(shape['Flow']).keys()))
tamaños.append((contador_max,tiempo))
contador=Counter(shape['Flow'])
contador_keys=Counter(shape['Flow']).keys()


#Seteo el rango de colores
white = Color("white")
colors = list(white.range_to(Color("red"),100))
shape.plot()






import matplotlib.pyplot as plt
shape.plot(column='Flow', cmap='OrRd',scheme='quantiles',legend='True')
plt.savefig("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\plot cobertura calles\\hola")


import os