# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:34:52 2018

@author: rodel
"""

import matplotlib.pyplot as plt
import fiona
import geopandas as gp
from mpl_toolkits.basemap import Basemap
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import osmnx as ox
from scipy.spatial import ConvexHull
from shapely.geometry import Point, LineString
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#print(plt.style.available)
plt.style.use('seaborn-ticks')
from matplotlib import rc
from matplotlib import rcParams
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','serif':['Computer Modern Sans serif']})
rc('text', usetex=True)
rcParams['text.latex.unicode']= True

#Data travel ties
trip_times = [5, 10, 15, 20, 25] #in minutes
travel_speed = 4.5 #walking speed in km/hour

#Open streets map
G = nx.read_gpickle("C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/multi_graph.gpickle")

gdf_nodes = ox.graph_to_gdfs(G, edges=False)
x, y = gdf_nodes['geometry'].unary_union.centroid.xy
center_node = ox.get_nearest_node(G, (y[0], x[0]))

meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, k, data in G.edges(data=True, keys=True):
    print(data)
    break
    #data['time'] = data['length'] / meters_per_minute


def roads_plot(title, description, fig_size, graph, trip_times, iso_colors, day_time,
               save_dir, center_nodes, fcl, fc, mc, rc, lw, facs, bounds=()):
    '''
    title,description,save_dir = str
    figsize= (h,w)
    graph = networkX.graph
    trip_times = [a,b,c,d] in minutes
    iso_colors = [color1,color2,color3,color4]
    day_time = str in ['t_08', 't_11','t_13','t_15','t_18','t_20','t_23']
    center_nodes = [(x0,y0),(x1,y1),...,(xk,yk)]
    '''
    
    #title = title
    #description = description
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, facecolor=fc, frame_on=True)
   # fig.suptitle(title, fontsize=20, y=.94, color=fontcolor)
    node_Xs = [float(x) for _, x in graph.nodes(data='x')]
    node_Ys = [float(y) for _, y in graph.nodes(data='y')]
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    if len(bounds)==0:
        west, south, east, north = edges.total_bounds
    else:
        west, south, east, north = bounds[0],bounds[1],bounds[2],bounds[3]
    cx, cy = (west + east)/2, (south + north)/2
    m = Basemap(llcrnrlon=west, llcrnrlat=south, urcrnrlon=east, urcrnrlat=north,
            lat_0=cx, lon_0=cy, resolution='c', projection='mill')
    m.drawmapscale(
        west + 0.032, south + 0.018,
        west, south, 6, units='km', barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555', fontsize = 35,
        zorder=3,yoffset=0.02*(m.ymax-m.ymin))
    routes_list = []
    node_colors = {}
    for center_node in center_nodes:
        routes = []
        for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
            subgraph = nx.ego_graph(graph, center_node, radius=trip_time, distance='travel_t_4')
            for node in subgraph.nodes():
                node_colors[node] = color
            lines = []
            for u, v, data in subgraph.edges(keys=False, data=True):
                if 'geometry' in data:
                    xs, ys = data['geometry'].xy
                    xs, ys = m(xs, ys)
                    lines.append(list(zip(xs, ys)))
            routes.append(lines)
        routes_list.append(routes)
    lines = []
    for u, v, data in graph.edges(keys=False, data=True):
        if 'geometry' in data:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry'].xy
            xs, ys = m(xs, ys)
            lines.append(list(zip(xs, ys)))
    lc = LineCollection(lines, colors =rc ,linewidths=lw[0], alpha=1, zorder=2)
    #lc = LineCollection(lines, colors ='#000000' ,linewidths=0.1, alpha=0.5, zorder=2)
    #lc = LineCollection(lines, colors ='#ec2d01' ,linewidths=0.15, alpha=0.5, zorder=2)
    ax.add_collection(lc)
    for route in routes_list:
        for i,lines in enumerate(route):
            lc = LineCollection(lines, colors = iso_colors[i] ,linewidths=lw[1], alpha=1, zorder=4+i)
            ax.add_collection(lc)
            # scatter plot the nodes
            #node_Xs = [node_Xs[i] for i in np.where(np.array(ns)!=0)[0]]
            #node_Ys = [node_Ys[i] for i in np.where(np.array(ns)!=0)[0]]
            #ns = [ns[i] for i in np.where(np.array(ns)!=0)[0]]
            #nc = [nc[i] for i in np.where(np.array(nc)!='none')[0]]
            #node_Xs,node_Ys = m(node_Xs,node_Ys)
            #ax.scatter(node_Xs, node_Ys, s=ns, c=nc, alpha=0.5, zorder=2)
    facility_lat = [item['geometry']['coordinates'][1] for item in facs]
    facility_lon = [item['geometry']['coordinates'][0] for item in facs] 
    lon_,lat_=m(facility_lon,facility_lat)
    ax.scatter(lon_,lat_,marker="^",c='#F5F5F5',s=150,alpha = 1, zorder = 15,edgecolors='black', linewidths=0.35)         
    m.drawmapboundary(fill_color=fc, linewidth=1,color='k')
    m.readshapefile('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/comuna_mapa','comunas',  color='#444444', linewidth=.4, zorder=1)
    for info, shape in zip(m.comunas_info, m.comunas):
        color = mc
        patches = [Polygon(np.array(shape), True,alpha=0.1)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
    
    n_bin = 100
    colors_cbar = [i for i in iso_colors[::-1]]
    print(colors_cbar)
    colors_cbar.append(rc)
    print(colors_cbar)
    cmap = mpl.colors.ListedColormap(name='my_list',colors=colors_cbar) #Colores discretos
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_list',colors_cbar,N=n_bin) #Colores continuos
    norm = mpl.colors.Normalize(vmin=trip_times[0], vmax=trip_times[-1]+2)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    vmin, vmax = trip_times[0],trip_times[-1]
    delta = (vmax-vmin)/((len(trip_times)))
    #tick_positions = [trip_times[0]+delta*x+delta/2 for x in range(len(trip_times))]
    tick_positions = trip_times + [trip_times[-1]+2]
    fig.tight_layout()
    '''
    'upper right':1,'upper left':2,'lower left':3,'lower right':4,'right':5,'center left':6,
    'center right':7,'lower center':8,'upper center':9,'center':10
    '''
    #cax = inset_axes(ax, width="1%", height="22%", loc=2) # A bbox_to_anchor can be specified instead of loc
    #lon, lat = m(-73.20,-36.78)
    #cax = inset_axes(ax, width="1%", height="22%", bbox_to_anchor = (lon,lat))
    cax = fig.add_axes([0.3, -0.04, 0.4, 0.02], zorder=3)

    #cbar = m.colorbar(sm, ax=cax,location='right',ticks=tick_positions,size='2%')
    cb = ColorbarBase(cax,cmap=cmap,norm=norm, orientation='horizontal',ticks=tick_positions)
    cbar_ticks = [str(x+2) for x in trip_times] + ['$>$ 10'] #in minutes
    #cb.ax.set_yticklabels(cbar_ticks)
    #cb.ax.tick_params(labelsize=14)
    #cb.ax.set_ylabel('Tiempo de respuesta (minutos)', rotation=90,fontsize=12.5,labelpad=13)
    cb.ax.set_xticklabels(cbar_ticks)
    cb.ax.tick_params(labelsize=40)
    cb.ax.set_xlabel('Response Time (minutes)', fontsize=45,labelpad=13)
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=.0,dpi=300, facecolor = 'w')
    
#################### Plot Parameters ####################
title = ''
description = ''
fontcolor='#666666'
facecolor = 'w'
mapcolor = '#989DA4'
mapcolor = '#AAABA9'
roadcolor = 'w'
linewidth = [1.3, 1.4]
figsize = (28, 20)
trip_times = [2,4,6,8]
save_dir ='C:/Users/Sebastian/Desktop/Ivan/img/cobertura_calles_.png'

#### Open roads
G = nx.read_gpickle("C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/multi_graph.gpickle")

#### Roads colors
#iso_colors = ox.get_colors(n=len(trip_times), cmap='Reds', start=0.65, return_hex=True)
iso_colors = ox.get_colors(n=len(trip_times), cmap='Blues', start=0.55, return_hex=True)

#### Facilities to plot
#fac = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/FINAL/Resultados_shp/all_new_fac_1_facs.shp')
#facilities = [node for node in fac if (int(node['properties']['abierto'])>0)]
facilities = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Companias de Bomberos/Companias_con_recursos.shp')
center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]
#center_node = center_node[21:23]

#### Bounds
comunas = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/comuna_mapa.shp')
bounds = comunas.bounds
bounds = (-73.222,-36.89,-72.928,-36.71)

####Option 1
save_dir ='C:/Users/Sebastian/Desktop/Ivan/img/cobertura_calles_1.png'
facecolor = 'w'
mapcolor = '#dddddd'
roadcolor = 'w'

####Option 2
#Blue
save_dir ='C:/Users/Sebastian/Desktop/Ivan/img/cobertura_calles_2.1.png'
facecolor = '#4d4946'
mapcolor = '#dddddd'
roadcolor = '#ffffff'
#Red
save_dir ='C:/Users/Sebastian/Desktop/Ivan/img/cobertura_calles_2.2.png'
iso_colors = ox.get_colors(n=len(trip_times), cmap='Reds', start=0.55, return_hex=True)

####Option 1
save_dir ='C:/Users/Sebastian/Desktop/Ivan/img/cobertura_calles_1.png'
facecolor = 'w'
mapcolor = '#dddddd'
roadcolor = '#a1a1a1'


roads_plot(title=title,description=description,fig_size=figsize,graph=G,
           trip_times=trip_times,iso_colors=iso_colors,day_time='t_18',
           save_dir=save_dir,center_nodes=center_node,bounds=bounds,
           fc=facecolor, mc=mapcolor, fcl=fontcolor,rc=roadcolor, lw=linewidth,
           facs = facilities)

def new_roads_plot(title,description,fig_size,graph,trip_times,iso_colors,new_iso_colors,
                   day_time,save_dir,center_nodes,new_center_nodes,bounds=()):
    '''
    title,description,save_dir = str
    figsize= (h,w)
    graph = networkX.graph
    trip_times = [a,b,c,d] in minutes
    iso_colors = [color1,color2,color3,color4]
    day_time = str in ['t_08', 't_11','t_13','t_15','t_18','t_20','t_23']
    center_nodes = [(x0,y0),(x1,y1),...,(xk,yk)]
    '''
    #title = title
    #descripton = description
    fontcolor='#666666'
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, facecolor='#f2f2f2', frame_on=False)
    #fig.suptitle(title, fontsize=20, y=.94, color=fontcolor)
    node_Xs = [float(x) for _, x in graph.nodes(data='x')]
    node_Ys = [float(y) for _, y in graph.nodes(data='y')]
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    if len(bounds)==0:
        west, south, east, north = edges.total_bounds
    else:
        west, south, east, north = bounds[0],bounds[1],bounds[2],bounds[3]
    cx, cy = (west + east)/2, (south + north)/2
    m = Basemap(llcrnrlon=west, llcrnrlat=south, urcrnrlon=east, urcrnrlat=north,
            lat_0=cx, lon_0=cy, resolution='c', projection='mill')
    m.drawmapscale(
        west + 0.032, south + 0.018,
        west, south, 6, units='km', barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555', fontsize = 40,
        zorder=3,yoffset=0.02*(m.ymax-m.ymin))
    routes_list = []
    node_colors = {}
    for center_node in center_nodes:
        routes = []
        for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
            subgraph = nx.ego_graph(graph, center_node, radius=trip_time, distance='travel_t_4')
            for node in subgraph.nodes():
                node_colors[node] = color
            lines = []
            for u, v, data in subgraph.edges(keys=False, data=True):
                if 'geometry' in data:
                    xs, ys = data['geometry'].xy
                    xs, ys = m(xs, ys)
                    lines.append(list(zip(xs, ys)))
            routes.append(lines)
        routes_list.append(routes)
    new_routes_list = []
    new_node_colors = {}
    for center_node in new_center_nodes:
        new_routes = []
        for trip_time, color in zip(sorted(trip_times, reverse=True), new_iso_colors):
            subgraph = nx.ego_graph(graph, center_node, radius=trip_time, distance='travel_t_4')
            for node in subgraph.nodes():
                new_node_colors[node] = color
            lines = []
            for u, v, data in subgraph.edges(keys=False, data=True):
                if 'geometry' in data:
                    xs, ys = data['geometry'].xy
                    xs, ys = m(xs, ys)
                    lines.append(list(zip(xs, ys)))
            new_routes.append(lines)
        new_routes_list.append(new_routes)
    lines = []
    for u, v, data in graph.edges(keys=False, data=True):
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            xs, ys = m(xs, ys)
            lines.append(list(zip(xs, ys)))
    lc = LineCollection(lines, colors ='gray' ,linewidths=0.5, alpha=0.6, zorder=2)
    ax.add_collection(lc)
    for route in routes_list:
        for i,lines in enumerate(route):
            lc = LineCollection(lines, colors = iso_colors[i] ,linewidths=0.5, alpha=0.65, zorder=10+i)
            ax.add_collection(lc)
    for route in new_routes_list:
        for i,lines in enumerate(route):
            lc = LineCollection(lines, colors = new_iso_colors[i] ,linewidths=0.55, alpha=0.65, zorder=4+i)
            ax.add_collection(lc)
    m.drawmapboundary(fill_color='#f2f2f2', linewidth=1.6,color='k')
    m.readshapefile('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/comuna_mapa','comunas',  color='#444444', linewidth=.4, zorder=1)
    for info, shape in zip(m.comunas_info, m.comunas):
        color = 'w'
        patches = [Polygon(np.array(shape), True,alpha=0.1)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
    n_bin = 100
    colors_cbar = [i for i in iso_colors[::-1]]
    cmap = mpl.colors.ListedColormap(name='my_list',colors=colors_cbar) #Colores discretos
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_list',colors_cbar,N=n_bin) #Colores continuos
    norm = mpl.colors.Normalize(vmin=trip_times[0], vmax=trip_times[-1])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    vmin, vmax = trip_times[0],trip_times[-1]
    delta = (vmax-vmin)/((len(trip_times)))
    tick_positions = trip_times 
    fig.tight_layout()
    cax = fig.add_axes([0.3, -0.055, 0.4, 0.02], zorder=3)
    cb = ColorbarBase(cax,cmap=cmap,norm=norm, orientation='horizontal',ticks=tick_positions)
    cbar_ticks = [str(x) for x in trip_times] #in minutes
    cb.ax.set_xticklabels(cbar_ticks)
    cb.ax.tick_params(labelsize=55)
    cb.ax.set_xlabel('Tiempo de viaje (minutos) \n Compañías con vehículo de tipo '+vehiculos[tipo].upper(), fontsize=55,labelpad=13)
    colors_cbar = [i for i in new_iso_colors[::-1]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_list',colors_cbar,N=n_bin) #Colores continuos
    norm = mpl.colors.Normalize(vmin=trip_times[0], vmax=trip_times[-1])
    cax1 = fig.add_axes([0.3, -0.03, 0.4, 0.02], zorder=3)
    cb1 = ColorbarBase(cax1,cmap=cmap,norm=norm, orientation='horizontal',ticks=tick_positions)
    cb1.set_ticks([])
    cb1.outline.set_visible(True)
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=.2,dpi=300, facecolor = 'w')
    
title = ''
descripton = ''
figsize = (28, 20)
trip_times = [4,6,8,10]
iso_colors_reds = ox.get_colors(n=len(trip_times), cmap='Reds', start=0.3, return_hex=True)
iso_colors_blues = ox.get_colors(n=len(trip_times), cmap='Blues', start=0.3, return_hex=True)
bounds = (-73.222,-36.89,-72.928,-36.71)
vehiculos = {'fuego':'q','forestal':'f','hazmat':'h','rescate':'r'}
G = nx.read_gpickle("C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/multi_graph.gpickle")


tipo = 'rescate'
save_dir ='C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/graficos/cobertura_calles_new_'+tipo+'.png'
fac = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/FINAL/Resultados_shp/new_fac_1_facs.shp')
facilities = [node for node in fac if (int(node['properties']['id'])>10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]
facilities = [node for node in fac if (int(node['properties']['id'])<10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
new_center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]
comunas = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/Calles/comuna_mapa.shp')

new_roads_plot(title,descripton,figsize,G,trip_times,iso_colors_blues,
               iso_colors_reds,'t_18',save_dir,center_node,new_center_node,bounds)

tipo = 'rescate'
save_dir ='C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/graficos/cobertura_calles_reloc_'+tipo+'.png'
fac = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/FINAL/Resultados_shp/reloc_fac_1_facs.shp')
facilities = [node for node in fac if (int(node['properties']['id'])>10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]
facilities = [node for node in fac if (int(node['properties']['id'])<10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
new_center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]

new_roads_plot(title,descripton,figsize,G,trip_times,iso_colors_blues,
               iso_colors_reds,'t_18',save_dir,center_node,new_center_node,bounds)
tipo = 'fuego'
save_dir ='C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/graficos/cobertura_calles_reloc_and_new_'+tipo+'.png'
fac = fiona.open('C:/Users/Sebastian/Desktop/Investigacion/MT Pregrado - Bomberos/Datos/FINAL/Resultados_shp/relocnew_fac1_facs.shp')
facilities = [node for node in fac if (int(node['properties']['id'])>10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]
facilities = [node for node in fac if (int(node['properties']['id'])<10000) & (int(node['properties']['abierto'])>0) & (int(node['properties'][vehiculos[tipo]])>0)]
new_center_node = [ox.get_nearest_node(G,(node['geometry']['coordinates'][1],node['geometry']['coordinates'][0])) for node in facilities]

new_roads_plot(title,descripton,figsize,G,trip_times,iso_colors_blues,
               iso_colors_reds,'t_18',save_dir,center_node,new_center_node,bounds)
