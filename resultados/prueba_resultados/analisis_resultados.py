import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import fiona 

data_family=pd.read_csv("C:\\Users\\ggalv\\Google Drive\\Respaldo\\TESIS MAGISTER\\Simulacion-evacuacion-antofagasta\\resultados\\prueba_resultados\\scenario 1 replica 1 Family.csv")
geometry=[Point(xy) for xy in zip(data_family.x,data_family.y)]
crs={'init':'epsg:5361'}
gdf_family=gpd.GeoDataFrame(data_family,crs=crs,geometry=geometry)
# print(gdf_family.head())
gdf_family.to_file("Family.shp",driver="ESRI Shapefile")