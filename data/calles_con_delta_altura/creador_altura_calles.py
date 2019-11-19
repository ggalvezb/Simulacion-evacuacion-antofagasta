import numpy as np
import geopandas as gpd
import pandas as pd

streets=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Corrected_Road_Network/Antofa_nodes_cut_edges/Antofa_edges.shp')
nodes=gpd.read_file('data/nodos_con_altura/Antofa_nodes_altura.shp')

def get_height(ID):
    u=streets.loc[streets['id']==str(ID)]['u'].item()  
    v=streets.loc[streets['id']==str(ID)]['v'].item()
    u_height=int(nodes.loc[nodes['id']==str(u)]['Dem_2'].item())
    v_height=int(nodes.loc[nodes['id']==str(v)]['Dem_2'].item())
    return(abs(u_height-v_height))

street_id=list(streets['id'])
contador=0
control=1000
altura=[]
for element in street_id:
    ID=element
    height=get_height(ID)
    altura.append(height)
    contador+=1
    if contador==control:
        print("Faltan "+str(len(street_id)-contador)+' para que empieze la simulacion')
        control+=1000

streets.to_file(driver='ESRI Shapefile',filename=r'C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/Simulacion-evacuacion-antofagasta/data/calles_con_delta_altura/calles_delta_altura.shp')
