import geopandas as gpd
import matplotlib.pyplot as plt

streets=gpd.read_file('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\data\\calles_con_delta_altura\\calles_delta_altura.shp')

pendientes=[]
for i in range(len(streets)):
    height=streets.loc[i]['height'].item()
    length=streets.loc[i]['length'].item()
    try:
        pendientes.append((height/length)*100)
    except:
        pendientes.append(0)
streets['pending']=pendientes

plt.hist(pendientes, bins = 30)

streets.to_file("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\transformacion_datos_altura\\resultados\\calles_antofa.shp",geometry='geometry',driver='ESRI Shapefile')