import pandas as pd
import os
import statistics as st

directorio= os.getcwd()

###############################
### ----- Escenario 1 ----- ###
###############################
delay_1=[]
evacuation_time_1=[]
length_scape_route_1=[]
for i in range(1,31):
    data_familia=pd.read_csv('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario1\\scenario 1 replica '+str(i)+' Family.csv',sep=",")
    delay_1.append(data_familia['Delays'].mean())
    evacuation_time_1.append(data_familia['Evacuation time'].mean())
    length_scape_route_1.append(data_familia['Length scape route'].mean())

###############################
### ----- Escenario 2 ----- ###
###############################

delay_2=[]
evacuation_time_2=[]
length_scape_route_2=[]
for i in range(1,31):
    data_familia=pd.read_csv('C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\resultados_buenos_escenario2\\scenario 2 replica '+str(i)+' Family.csv',sep=",")
    delay_2.append(data_familia['Delays'].mean())
    evacuation_time_2.append(data_familia['Evacuation time'].mean())
    length_scape_route_2.append(data_familia['Length scape route'].mean())

###############################
### ----- Dataframe  ----- ###
###############################
df=pd.DataFrame(columns=['Avarage Delay','Avarage Evacuation Time','Avarage Length Scape'])
df.loc["Escenario 1"]=[st.mean(delay_1), st.mean(evacuation_time_1), st.mean(length_scape_route_1)]
df.loc["Escenario 2"]=[st.mean(delay_2), st.mean(evacuation_time_2), st.mean(length_scape_route_2)]
