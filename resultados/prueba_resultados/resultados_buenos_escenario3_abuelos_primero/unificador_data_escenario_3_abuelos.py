import pandas as pd 
import ast
import statistics as st
import geopandas as gpd

house_to_evacuate=gpd.read_file("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\tsunami\\Shapefiles\\Individual_Houses\\House_to_evacuate\\Houses_to_evacuate.shp")
streets=gpd.read_file('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\data\\calles_con_delta_altura\\calles_delta_altura.shp')
meating_points=gpd.read_file('C:/Users/ggalv/Google Drive/Respaldo/TESIS MAGISTER/tsunami/Shapefiles/Tsunami/Puntos_Encuentro/Puntos_Encuentro_Antofagasta/puntos_de_encuentro.shp')
meating_points_reproyectado=gpd.read_file('C:\\Users\\ggalv\\Desktop\\MP_reproyectados\\puntos_de_encuentro_reproyectados_mundial.shp')
edificios_reproyectado=gpd.read_file('C:\\Users\\ggalv\\Desktop\\MP_reproyectados\\edificios_reproyectados_mundial.shp')


data_familia=pd.read_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\scenario 3 replica 1 Family.csv")
data_MP=pd.read_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\scenario 3 replica 1 MP.csv")
data_BD=pd.read_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\scenario 3 replica 1 BD.csv")

#Saco datos de Familia de un escenario
Lugar_escape,Adult,Young,Kids,Elders,Males,Women,Latitud,Longitud=[],[],[],[],[],[],[],[],[]
for i in range(len(data_familia)):
    family_find=house_to_evacuate.loc[house_to_evacuate['OBJECTID']==data_familia.loc[i,'Housing']]
    data_familia_transformado=ast.literal_eval(data_familia.loc[i,'Safe point'])
    try:
        if data_familia_transformado[1]=='BD':
            Lugar_escape.append("Edificio")
        else:
            Lugar_escape.append("Encuentro")
    except:
        Lugar_escape.append("Encuentro")
    members_transform=ast.literal_eval(data_familia.loc[i,'Members'])
    Adult.append(members_transform['adults'])
    try:
        Young.append(members_transform['youngs'])
    except:
        Young.append(0)
    try:
        Elders.append(members_transform['olds'])
    except:
        Elders.append(0)
    try:
        Kids.append(members_transform['kids'])
    except:
        Kids.append(0)
    try:
        Males.append(members_transform['males'])
    except:
        Males.append(0)
    try:
        Women.append(members_transform['women'])
    except:
        Women.append(0)
    inicial=family_find['LATITUD'].item().maketrans 
    final = family_find['LATITUD'].item().translate(inicial(', .', '., '))
    Latitud.append(final)
    inicial=family_find['LONGITUD'].item().maketrans 
    final = family_find['LONGITUD'].item().translate(inicial(', .', '., '))
    Longitud.append(final)
data_familia['Tipo']=Lugar_escape
data_familia["Adult"]=Adult
data_familia["Young"]=Young
data_familia["Kids"]=Kids
data_familia["Elders"]=Elders
data_familia["Males"]=Males
data_familia["Women"]=Women
data_familia['Latitud']=Latitud
data_familia['Longitud']=Longitud

data_familia.to_excel("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\prueba FM\\datos_FM.xlsx")

#Saco datos de MP de un escenario 
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
df.to_excel("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\prueba MD\\datos_MP.xlsx")

#Saco datos de BD de un escenario 
BD_ID,X,Y,Adult,Young,Kids,Elders,Males,Women,Num_family=[],[],[],[],[],[],[],[],[],[]
for j in range(len(data_BD)):
    BD_ID.append(data_BD.loc[j,'ID'])
    Num_family.append(data_BD.loc[j,'Num Family'])
    members_transform=ast.literal_eval(data_BD.loc[j,'Members'])
    Adult.append(members_transform['adults'])
    try:
        Young.append(members_transform['youngs'])
    except:
        Young.append(0)
    try:
        Elders.append(members_transform['olds'])
    except:
        Elders.append(0)
    try:
        Kids.append(members_transform['kids'])
    except:
        Kids.append(0)
    try:
        Males.append(members_transform['males'])
    except:
        Males.append(0)
    try:
        Women.append(members_transform['women'])
    except:
        Women.append(0)
    X.append(edificios_reproyectado.loc[(data_BD.loc[j,'ID']-1),'geometry'].x)
    Y.append(edificios_reproyectado.loc[(data_BD.loc[j,'ID']-1),'geometry'].y)
diccionario={"BD_ID":BD_ID,"X":X,"Y":Y,"Adult":Adult,"Young":Young,"Kids":Kids,"Elders":Elders,"Males":Males,"Women":Women,"Num_family":Num_family}
df=pd.DataFrame(diccionario)
df.to_excel("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario3_abuelos_primero\\prueba BD\\datos_BD.xlsx")

