import pandas as pd
import os
import geopandas as gpd
import ast

###-------------Data---------------###
streets=gpd.read_file('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\data\\calles_con_delta_altura\\calles_delta_altura.shp')
meating_points=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Tsunami/Puntos_Encuentro/Puntos_Encuentro_Antofagasta/puntos_de_encuentro.shp')
meating_points_reproyectado=gpd.read_file('C:\\Users\\ggalv\\Desktop\\MP_reproyectados\\puntos_de_encuentro_reproyectados_mundial.shp')
edificios_reproyectado=gpd.read_file('C:\\Users\\ggalv\\Desktop\\MP_reproyectados\\edificios_reproyectados_mundial.shp')


directorio= os.getcwd()

delay=[]
start_scape_time=[]
end_scape_time=[]
evacuation_time=[]
length_scape_route=[]

for i in range(1,31):
    data_familia=pd.read_csv(directorio+'\\scenario 1 replica '+str(i)+' Family.csv',sep=",")
    delay.append(data_familia['Delays'].mean())
    start_scape_time.append(data_familia['Start scape time'].mean())
    end_scape_time.append(data_familia['End scape time'].mean())
    evacuation_time.append(data_familia['Evacuation time'].mean())
    length_scape_route.append(data_familia['Length scape route'].mean())

for i in range(1,2):
    data_familia=pd.read_csv(directorio+'\\scenario 1 replica '+str(i)+' Family.csv',sep=",")
    data_MP=pd.read_csv(directorio+'\\scenario 1 replica '+str(i)+' MP.csv',sep=",")
    data_BD=pd.read_csv(directorio+'\\scenario 1 replica '+str(i)+' BD.csv',sep=",")

#Saco datos de MP de un escenario 
data_MP=pd.read_csv(directorio+'\\scenario 1 replica 1 MP.csv',sep=",")
MP_ID,X,Y,Adult,Young,Kids,Elders,Males,Women=[],[],[],[],[],[],[],[],[]
for j in range(len(data_MP)):
    meating_point_find=meating_points_reproyectado.loc[meating_points_reproyectado['OBJECTID']==data_MP.loc[j,'ID']]
    MP_ID.append(data_MP.loc[j,'ID'])
    members_transform=ast.literal_eval(data_MP.loc[j,'Members'])
    Adult.append(members_transform['adults'])
    try:
        Young.append(members_transform['youngs'])
    except:
        Young.append(0)
    try:
        Elders.append(members_transform['olds'])
    except:
        Elders.append(0)
    Kids.append(members_transform['kids'])
    Males.append(members_transform['males'])
    Women.append(members_transform['women'])
    X.append(meating_point_find['geometry'].x.item())
    Y.append(meating_point_find['geometry'].y.item())
diccionario={"MP_ID":MP_ID,"X":X,"Y":Y,"Adult":Adult,"Young":Young,"Kids":Kids,"Elders":Elders,"Males":Males,"Women":Women}
df=pd.DataFrame(diccionario)
df.to_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario1\\prueba MD\\datos_MP.csv")

