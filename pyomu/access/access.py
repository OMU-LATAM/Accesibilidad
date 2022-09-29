import pandas as pd
from pandas._libs.lib import is_integer
import numpy as np
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import seaborn as sns

import mapclassify
import contextily as ctx

from tqdm.notebook import tqdm
tqdm.pandas()

import time

import unidecode
import itertools
from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.decomposition import PCA

import osmnx as ox
import networkx as nx
import pandana

from pandana.loaders import osm as osm_pandana

import datetime as datetime
from datetime import date
from datetime import timedelta

pd.set_option('display.max_columns', None)

import googlemaps

import pytz
from tzwhere import tzwhere 
from pytz import timezone

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from shapely.geometry import Point, Polygon

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor # ColorFormat, 
from PIL import Image, ImageDraw

import os
import glob

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import IPython
from IPython.display import set_matplotlib_formats


from PIL import Image # pip install Pillow
from PIL import ImageOps

import seaborn as sns
from pandas import DataFrame

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyomu.utils import utils
from pyomu import pyomu

def create_matrix(origin,                   
                  destination, 
                  id_origin = '', 
                  id_destination = '', 
                  latlon = True,                  
                  normalize = False, 
                  res = 8,
                  duplicates = False):
    
    '''
    En base a un GeoDataFrame de origines y uno de destinos crea una matriz de origin-destino para el cálculo de las distancias y/o tiempos.
    Parámetros:
    
    origin = GeoDataFrame de origines    
    destination = GeoDataFrame de destinos
    id_origin = ID del GeoDataFrame de origenes (opcional)
    id_destination = ID del GeoDataFrame de destinos (opcional)
    latlon = Especifica si la función devuelve una matriz con campos de latitud o longitud o con un campo de origin y uno de destino con lat/lon tipo texto (ej. "-31.06513, -64.28675")
    normalize = Si normalize es True, toma los origines y destinos desde el centroide del hexágono h3 con la resolución correspondiente. Este permite minimizar la cantidad de 
    consultas de tiempos de viaje en observaciones que están a una distancia muy cercana. Esto permite reducir los costos de consulta en Google Maps.
    res = Resolución de los hexágonos h3 si normalize = True.
    duplicates = False: Genera una matriz de origin y destinos sin duplicados. True: mantiene las tablas origin y destination sin modificar y puede traer duplicados si existen
    
    Salida: Matriz de origines y destinos
   '''
    
    origin_columns = origin.columns.tolist()
    destination_columns = destination.columns.tolist()
    origin_columns.remove('geometry')
    destination_columns.remove('geometry')
    
    lat_o, lon_o, lat_d, lon_d = 'lat_o', 'lon_o', 'lat_d', 'lon_d'
    
    origin = utils.h3.create_latlon(df = origin, normalize=True, res=res, hex='hex_o', var_latlon='origin', lat=lat_o, lon=lon_o).drop(['geometry'], axis=1)
    
    destination = utils.h3.create_latlon(df = destination, normalize=True, res=res, hex='hex_d', var_latlon='destination', lat=lat_d, lon=lon_d).drop(['geometry'], axis=1)
    
    cols_latlon_o, cols_latlon_d = [], []
    if latlon:        
        if not normalize:
            lat_o, lon_o, lat_d, lon_d = 'lat_o', 'lon_o', 'lat_d', 'lon_d'
        else:
            lat_o, lon_o, lat_d, lon_d = 'lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'
        cols_latlon_o = [lat_o, lon_o]
        cols_latlon_d = [lat_d, lon_d]
    
    if not duplicates:
        if not normalize:
            origin = origin[['hex_o', 'origin', 'origin_norm']+cols_latlon_o].drop_duplicates().reset_index(drop=True)
            destination = destination[['hex_d', 'destination', 'destination_norm']+cols_latlon_d].drop_duplicates().reset_index(drop=True) 
        else:
            origin = origin[['hex_o', 'origin_norm']+cols_latlon_o].drop_duplicates().reset_index(drop=True)
            destination = destination[['hex_d', 'destination_norm']+cols_latlon_d].drop_duplicates().reset_index(drop=True) 
            
        cols = ['hex_o', 'hex_d']
    
    else:
        cols = []
        if len(id_origin)>0:
            cols += [id_origin]
        if len(id_destination)>0:
            cols += [id_destination]

        if id_origin != 'hex_o':
            cols += ['hex_o']
        if id_destination != 'hex_d':
            cols += ['hex_d']
                
    origin['aux'] = 1
    destination['aux'] = 1
    
    od_matrix = pd.merge(
                        origin,
                        destination,
                        on='aux').drop(['aux'], axis=1)
    
    del origin['aux']
    del destination['aux']
    
    if (not latlon)&(duplicates):
        od_matrix = od_matrix.drop(['lat_o', 'lon_o', 'lat_d', 'lon_d', 'lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'], axis=1)
    
    for i in origin_columns:
        if (not i in cols)&(i in od_matrix.columns):
            cols += [i]
    
    for i in destination_columns:
        if (not i in cols)&(i in od_matrix.columns):
            cols += [i]
    
    
    for i in od_matrix.columns:
        if not i in cols:
            cols += [i]
    
    od_matrix = od_matrix[cols]
    
    
    return od_matrix


def measure_distances(idmatrix, node_from, node_to, G, lenx):
    '''
    Función de apoyo de measure_distances_osm
    '''
    
    if idmatrix % 2000 == 0:
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} processing {int(idmatrix)} / {lenx}')
        
    try:
        ret = nx.shortest_path_length(G, node_from, node_to, weight='length')        
    except:
        ret = np.nan
    return ret


def h3togeo(x):    
        return str(h3.h3_to_geo(x)[0]) +', '+ str(h3.h3_to_geo(x)[1])
    
def h3dist(x):
    return h3.h3_distance(x.h3_o, x.h3_d) / 10

def measure_distances_osm_from_matrix(df, 
                                      origin = '', 
                                      destination = '',
                                      lat_o = '',
                                      lon_o = '',
                                      lat_d = '',
                                      lon_d = '',           
                                      h3_o = '',
                                      h3_d = '',
                                      processing = 'pandana',
                                      ):
    cols = df.columns.tolist()
    
    if len(lat_o)==0: lat_o = 'lat_o'
    if len(lon_o)==0: lon_o = 'lon_o'
    if len(lat_d)==0: lat_d = 'lat_d'
    if len(lon_d)==0: lon_d = 'lon_d'
    
    if (lon_o not in df.columns)|(lat_o not in df.columns):      
        if (origin not in df.columns)&(len(h3_o)>0):
            origin='origin'
            df[origin] = df[h3_o].apply(h3togeo)    
        df['lon_o'] = df[origin].str.split(',').apply(lambda x: x[1]).str.strip().astype(float)        
        df['lat_o'] = df[origin].str.split(',').apply(lambda x: x[0]).str.strip().astype(float)
    
    if (lon_d not in df.columns)|(lat_d not in df.columns):  
        if (destination not in df.columns)&(len(h3_d)>0):
            destination='destination'
            df[destination] = df[h3_d].apply(h3togeo)    
        df['lon_d'] = df[destination].str.split(',').apply(lambda x: x[1]).str.strip().astype(float)    
        df['lat_d'] = df[destination].str.split(',').apply(lambda x: x[0]).str.strip().astype(float)


    ymin, xmin, ymax, xmax = min(df['lat_o'].min(), df['lat_d'].min()), \
                             min(df['lon_o'].min(), df['lon_d'].min()), \
                             max(df['lat_o'].max(), df['lat_d'].max()), \
                             max(df['lon_o'].max(), df['lon_d'].max())
    xmin -= .2
    ymin -= .2
    xmax += .2
    ymax += .2
    
    var_distances = []

    for mode in ['drive', 'walk']:
        print('')
        print(f'Coords OSM {mode} - Download map') 
        print('ymin, xmin, ymax, xmax', ymin, xmin, ymax, xmax)

        print('')

        if processing != 'pandana':


            G = ox.graph_from_bbox(ymax,
                                   ymin,
                                   xmax, 
                                   xmin, 
                                   network_type=mode)

            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)

            nodes_from = ox.distance.nearest_nodes(G, 
                                                   df[lon_o].values, 
                                                   df[lat_o].values, 
                                                   return_dist=True)

            df['node_from'] = nodes_from[0]

            nodes_to = ox.distance.nearest_nodes(G, 
                                                 df[lon_d].values, 
                                                 df[lat_d].values, 
                                                 return_dist=True)

            df['node_to'] = nodes_to[0]

            if 'idmatrix' not in df.columns:
                df = df.reset_index(drop=True).reset_index().rename(columns={'index':'idmatrix'})
            df[f'distance_osm_{mode}'] = df.apply(lambda x : measure_distances(x['idmatrix'],
                                                                                             x['node_from'], 
                                                                                             x['node_to'], 
                                                                                             G = G, 
                                                                                             lenx = len(df)), 
                                                       axis=1)
        else:

            network = osm_pandana.pdna_network_from_bbox(ymin, xmin, ymax,  xmax, network_type=mode)  

            df['node_from'] = network.get_node_ids(df[lon_o], df[lat_o]).values
            df['node_to'] = network.get_node_ids(df[lon_d], df[lat_d]).values
            df[f'distance_osm_{mode}'] = network.shortest_path_lengths(df['node_to'].values, df['node_from'].values) 

        var_distances += [f'distance_osm_{mode}']
        df[f'distance_osm_{mode}'] = (df[f'distance_osm_{mode}'] / 1000).round(2)

        print('')

    df.loc[(df.distance_osm_drive*1.3) < df.distance_osm_walk, 'distance_osm_walk'] = df.loc[(df.distance_osm_drive*1.3) < df.distance_osm_walk, 'distance_osm_drive']

    df.loc[df.distance_osm_drive>2000, 'distance_osm_drive'] = np.nan
    df.loc[df.distance_osm_walk>2000, 'distance_osm_walk'] = np.nan
    
    df = df[cols+var_distances]
    
    if (len(h3_o)>0)&(len(h3_d)>0):
        df['distance_h3'] = df[[h3_o, h3_d]].apply(h3dist, axis=1)
    
    return df


def measure_distances_osm(origin, 
                          id_origin, 
                          destination, 
                          id_destination,                           
                          normalize=False, 
                          res = 8,                           
                          processing='pandana', 
                          equipement_bring_closest = False, 
                          equipment_closest_qty = 3, 
                          closest_distance = [800, 1500, 2000],
                          equipment_type = '',
                          trips_file = 'trips_file_tmp',
                          current_path=Path(),
                          city=''):
    
    '''
    Calcula distancias en Open Street Maps para los modos de transporte entre dos capas cartográficas de origenes y destinos.
    En el caso de que alguna capa sea de polígonos, toma en cuenta el centroide de cada polígono.
    Devuelve una od_matrix de origenes y destinos con las distancias calculadas para cada modo.
    
    Parámetros
    origin = GeoDataFrame que representa a los puntos de origen
    id_origin = ID del GeoDataFrame origin
    destination = GeoDataFrame que representa a los puntos de destino
    id_destination = ID del GeoDataFrame destino
    driving = True    
    walking = True     
    processing= Se especifica donde procesar la consulta, puede ser a través de osmnx o pandana. Por defecto usa 'pandana'. Ambas busquedas utilizan Open Street Maps.
    equipement_bring_closest = Si trae solo los destinos más cercanos. False trae todos los destinos, True. Ordena por distancia y trae los más cercanos según la variable equipment_closes_qty
    equipment_closest_qty = Cantidad de destinos a traer si equipement_bring_closest es True
    equipment_type = Si hay más de un tipo de equipamiento en la tabla, por ejemplo Escuelas y Hospitales en una variable tipo_establecimientos (ej. equipment_type = 'tipo_establecimientos'). Se puede enviar una lista de variables.
    trips_file = Nombre de archivo temporal donde se guardan las consultas. Por defecto 'trips_file_tmp'
    current_path = Directorio de trabajo. Por defecto: Path()
    
    Salida: Matriz de origenes y destino con los campos nuevos de distancia calculados en Open Street Maps para los modos solicitados. 
    
    '''
    
    driving = True
    walking = True

    
    if len(city)>0:
        trips_file = current_path / 'tmp' / f'{city}_{trips_file}_osm.csv'
    else:
        trips_file = current_path / 'tmp' / f'{trips_file}_osm.csv'

    add_file = ''
    if driving: add_file+='_drive'
    if walking: add_file+='_walk'
    if normalize: add_file+='_norm'

    utils.create_result_dir(current_path=current_path)

    modes = []
    if driving:
        modes += ['drive']
    if walking:
        modes += ['walk']

    if not normalize:
        lat_o, lon_o, lat_d, lon_d = 'lat_o', 'lon_o', 'lat_d', 'lon_d'
        geo_origin, geo_destination = 'origin', 'destination'
    else:
        lat_o, lon_o, lat_d, lon_d = 'lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'
        geo_origin, geo_destination = 'origin_norm', 'destination_norm'


    od_matrix_all = create_matrix(origin, 
                                  destination, 
                                  id_origin=id_origin, 
                                  id_destination = id_destination, 
                                  normalize = False, 
                                  res = res, 
                                  duplicates = True, 
                                  latlon = False)

    od_matrix = create_matrix(origin, 
                              destination, 
                              id_origin = id_origin, 
                              id_destination = id_destination, 
                              latlon = True, 
                              normalize = normalize, res=res)

    trips = pd.DataFrame([])

    print('Archivo temporal', trips_file)

    if Path(trips_file).is_file():  

        trips = pd.read_csv(trips_file)

        trips['osm'] = 1

        if (geo_origin in trips.columns)&(geo_destination in trips.columns):

            od_matrix = od_matrix.merge(trips[[geo_origin, 
                                               geo_destination, 
                                               'osm']],
                                                how='left', 
                                                on=[geo_origin, 
                                                    geo_destination])

            od_matrix = od_matrix[od_matrix.osm.isna()]
            del od_matrix['osm']
            del trips['osm']



    if len(od_matrix) > 0:
        od_matrix = measure_distances_osm_from_matrix(od_matrix, 
                                                      lat_o = lat_o, 
                                                      lon_o = lon_o, 
                                                      lat_d = lat_d, 
                                                      lon_d = lon_d,
                                                      processing = processing)

    od_matrix = pd.concat([trips, od_matrix], ignore_index=True)

    if 'lat_o' in od_matrix.columns:
        od_matrix = od_matrix.drop(['lat_o', 'lon_o', 'lat_d', 'lon_d'], axis=1)
    if 'lat_o_norm' in od_matrix.columns:
        od_matrix = od_matrix.drop(['lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'], axis=1)
    if (not normalize)&('origin_norm' in od_matrix.columns):
        od_matrix = od_matrix.drop(['origin_norm', 'destination_norm'], axis=1)


    od_matrix.to_csv(trips_file, index=False)

    od_matrix = od_matrix_all.merge(od_matrix, how='left', on=['hex_o', 'hex_d', geo_origin, geo_destination])

    if equipement_bring_closest:

        if len(equipment_type) == 0:
            od_matrix['equipment_type_tmp'] = 1
            equipment_type = ['equipment_type_tmp']

        if type(equipment_type) == str:
            equipment_type = [equipment_type]

        od_matrix = od_matrix.sort_values(equipment_type + [id_origin, 'distance_osm_walk'])
        od_matrix['distance_order'] = od_matrix.groupby(equipment_type + [id_origin]).transform('cumcount')
        od_matrix['distance_order'] = od_matrix['distance_order'] + 1


        if len(closest_distance)>0:
            if type(closest_distance) == str:
                closest_distance = [closest_distance]
            for i in closest_distance:
                od_matrix_tmp = od_matrix[od_matrix.distance_osm_walk<=(i/1000)].groupby(equipment_type + [id_origin], as_index=False).size().rename(columns={'size':f'qty_est_{i}m'})                
                od_matrix = od_matrix.merge(od_matrix_tmp, on=equipment_type + [id_origin], how='left')
                od_matrix[f'qty_est_{i}m'] = od_matrix[f'qty_est_{i}m'].fillna(0).astype(int)

        od_matrix = od_matrix[od_matrix['distance_order']<=equipment_closest_qty].reset_index(drop=True)

        if 'equipment_type_tmp' in od_matrix.columns:
            od_matrix = od_matrix.drop(['equipment_type_tmp'], axis=1)


    print('Proceso OSM finalizado')

    return od_matrix


def trip_info_googlemaps(coord_origin, 
                         coord_destin, 
                         trip_datetime, 
                         trip_datetime_corrected,
                         gmaps,
                         transit = True, 
                         driving = True, 
                         walking = False, 
                         bicycling = False):
    
    '''
    Realiza una consulta a la API de Google Maps teniendo en cuenta un origen y un destino
    
    Parámetros:
    coord_origin = latitud/longitud (xy) del origen del viaje en formato texto. Ej. '-31.4247, -64.15922'
    coord_destin = = latitud/longitud (xy) del destino del viaje en formato texto. Ej. '-31.4247, -64.15922'
    trip_datetime = Fecha y hora de la consulta del viaje en formato datetime
    gmaps = Objeto gmaps para acceder a la API de googlemaps
    transit = Para obtener distancias y tiempos en transporte público (True/False). Por defecto True.
    driving = Para obtener distancias y tiempos en automovil (True/False). Por defecto True.
    walking = Para obtener distancias y tiempos caminando (True/False). Por defecto False.
    bicycling = Para obtener distancias y tiempos en bicicleta (True/False). Por defecto False.
    
    Salida: nuevo dataframe con las distancias y tiempos calculados para ese origen y destino
    
    '''
    
    alternatives = False
    
    
    
    Transit, Driving, Walking, Bicycling = '', '', '', ''
    
    
    
    for _ in range(0, 3):
        try:

            if transit:
                Transit = gmaps.directions(coord_origin,
                                           coord_destin,
                                           mode='transit', 
                                           departure_time=trip_datetime_corrected, 
                                           alternatives=alternatives, 
                                           traffic_model="best_guess")


            if driving:
                Driving = gmaps.directions(coord_origin,
                                           coord_destin,
                                           mode='driving', 
                                           departure_time=trip_datetime_corrected, 
                                           alternatives=alternatives, 
                                           traffic_model="best_guess")

            if walking:
                Walking = gmaps.directions(coord_origin,
                                           coord_destin,
                                           mode='walking', 
                                           departure_time=trip_datetime_corrected, 
                                           alternatives=alternatives, 
                                           traffic_model="best_guess")


            if bicycling:
                Bicycling = gmaps.directions(coord_origin,
                                           coord_destin,
                                           mode='bicycling', 
                                           departure_time=trip_datetime_corrected, 
                                           alternatives=alternatives, 
                                           traffic_model="best_guess")
            gmaps_ok = 1
            break
        except:
            time.sleep(5)            
            gmaps_ok = 0

    transit_trips, driving_trips, walking_trips, bicycling_trips = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    
    if gmaps_ok == 1:        

        # Transit
        v1, v2, v3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

        if len(Transit) != 0:
            t1, t2, t3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
            t1 = pd.DataFrame.from_dict(pd.json_normalize(Transit))
            for i in range(0, len(t1)):

                t1.loc[i,"trip_datetime"] = trip_datetime                
                t1.loc[i,"geo_origin"] = coord_origin
                t1.loc[i,"geo_destination"] = coord_destin
                t1.loc[i,"alternative"] = i

                t2 = pd.DataFrame.from_dict(pd.json_normalize(Transit[i]['legs']))
                for l in range(0, len(t2)):
                    t2.loc[l,"trip_datetime"] = trip_datetime                    
                    t2.loc[l,"geo_origin"] = coord_origin
                    t2.loc[l,"geo_destination"] = coord_destin
                    t2.loc[l,"alternative"] = i
                    t2.loc[l,"legs"] = l

                    t3 = pd.DataFrame.from_dict(pd.json_normalize(Transit[i]['legs'][l]['steps']))        
                    for s in range(0, len(t3)):
                        t3.loc[s,"trip_datetime"] = trip_datetime                        
                        t3.loc[s,"geo_origin"] = coord_origin
                        t3.loc[s,"geo_destination"] = coord_destin
                        t3.loc[s,"alternative"] = i
                        t3.loc[s,"legs"] = l
                        t3.loc[s,"steps"] = s

                    v3 = pd.concat([v3, t3], ignore_index=True)
                v2 = pd.concat([v2, t2], ignore_index=True)
            v1 = pd.concat([v1, t1], ignore_index=True)



            if 'transit_details.line.vehicle.name' not in v3.columns:
                v3['transit_details.line.vehicle.name'] = np.nan

            if 'transit_details.line.vehicle.type' not in v3.columns:
                v3['transit_details.line.vehicle.type'] = np.nan



            v3 = v3[['trip_datetime', 
                'geo_origin', 
                'geo_destination', 
                'alternative', 
                'steps', 
                'travel_mode', 
                'distance.value', 
                'duration.value',              
                'transit_details.line.vehicle.name',
                'transit_details.line.vehicle.type'
                   ]]

            v3 = pd.concat([v3, 
                            pd.get_dummies(v3['travel_mode']),
                            pd.get_dummies(v3['transit_details.line.vehicle.type'])], axis=1)

            transit_modes = []
            for i in v3['travel_mode'].unique().tolist():
                v3[f'transit_{i.lower()}.distance'] = v3['distance.value'] * v3[i]
                v3[f'transit_{i.lower()}.duration'] = v3['duration.value'] * v3[i]
                v3[f'transit_{i.lower()}.steps'] = 1 * v3[i]

                transit_modes += [f'transit_{i.lower()}.distance', f'transit_{i.lower()}.duration', f'transit_{i.lower()}.steps']

            steps = v3.groupby(['trip_datetime', 'geo_origin', 'geo_destination', 'alternative'])[transit_modes].sum().reset_index()

            if 'departure_time.text' not in v2.columns:
                v2['departure_time.text'] = np.nan

            if 'arrival_time.text' not in v2.columns:
                v2['arrival_time.text'] = np.nan


            transit_trips = v2[['trip_datetime', 
                                'geo_origin', 
                                'geo_destination', 
                                'alternative',                             
                                'departure_time.text',                        
                                'arrival_time.text',
                                'distance.value',                    
                                'duration.value',                     
                                ]].copy()

            transit_trips = transit_trips.merge(steps, how='left', on=['trip_datetime', 
                                                                    'geo_origin', 
                                                                    'geo_destination', 
                                                                    'alternative'])
            transit_trips = transit_trips.merge(
                                    v3.loc[(v3.steps==0)&(v3.travel_mode=='WALKING'), ['trip_datetime', 
                                                                                       'geo_origin', 
                                                                                       'geo_destination', 
                                                                                       'alternative', 
                                                                                       'distance.value', 
                                                                                       'duration.value']].rename(columns={'distance.value': 'transit_walking.distance.origin',
                                                                                                                          'duration.value': 'transit_walking.duration.origin'}),
                                    how='left', 
                                    on=['trip_datetime', 'geo_origin', 'geo_destination', 'alternative'])

            transit_trips.columns = [i.replace('.value', '') for i in transit_trips.columns ]
            transit_trips.columns = [i.replace('.text', '') for i in transit_trips.columns ]

            transit_trips = transit_trips.rename(columns={'arrival_time': 'transit_arrival_time', 
                                                         'departure_time':'transit_departure_time',
                                                         'distance':'transit_distance',
                                                         'duration': 'transit_duration'})


        # Driving
        d1, d2, d3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        if len(Driving) != 0:    
            t1, t2, t3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
            t1 = pd.DataFrame.from_dict(pd.json_normalize(Driving))
            for i in range(0, len(t1)):
                t1.loc[i,"trip_datetime"] = trip_datetime                
                t1.loc[i,"geo_origin"] = coord_origin
                t1.loc[i,"geo_destination"] = coord_destin
                t1.loc[i,"alternative"] = i

                t2 = pd.DataFrame.from_dict(pd.json_normalize(Driving[i]['legs']))
                for l in range(0, len(t2)):
                    t2.loc[l,"trip_datetime"] = trip_datetime                
                    t2.loc[l,"geo_origin"] = coord_origin
                    t2.loc[l,"geo_destination"] = coord_destin
                    t2.loc[l,"alternative"] = i
                    t2.loc[l,"legs"] = l

                    t3 = pd.DataFrame.from_dict(pd.json_normalize(Driving[i]['legs'][l]['steps']))        
                    for s in range(0, len(t3)):
                        t3.loc[s,"trip_datetime"] = trip_datetime                
                        t3.loc[s,"geo_origin"] = coord_origin
                        t3.loc[s,"geo_destination"] = coord_destin
                        t3.loc[s,"alternative"] = i
                        t3.loc[s,"legs"] = l
                        t3.loc[s,"steps"] = s

                    d3 = pd.concat([d3, t3], ignore_index=True)

                d2 = pd.concat([d2, t2], ignore_index=True)
            d1 = pd.concat([d1, t1], ignore_index=True)

            if 'duration_in_traffic.value' not in d2.columns:            
                d2['duration_in_traffic.value'] = np.nan


            driving_trips = d2[['trip_datetime',
                               'geo_origin', 'geo_destination', 'alternative',                   
                               'distance.value', 
                               'duration.value', 
                               'duration_in_traffic.value']]
            driving_trips.columns = [i.replace('.value', '') for i in driving_trips.columns ]
            driving_trips = driving_trips.rename(columns={'distance':'driving_distance', 'duration': 'driving_duration', 'duration_in_traffic':'driving_duration_in_traffic'})

        # Walking
        w1, w2, w3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        if len(Walking) != 0:
            t1, t2, t3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
            t1 = pd.DataFrame.from_dict(pd.json_normalize(Walking))
            for i in range(0, len(t1)):
                t1.loc[i,"trip_datetime"] = trip_datetime                
                t1.loc[i,"geo_origin"] = coord_origin
                t1.loc[i,"geo_destination"] = coord_destin
                t1.loc[i,"alternative"] = i

                t2 = pd.DataFrame.from_dict(pd.json_normalize(Walking[i]['legs']))
                for l in range(0, len(t2)):
                    t2.loc[l,"trip_datetime"] = trip_datetime                
                    t2.loc[l,"geo_origin"] = coord_origin
                    t2.loc[l,"geo_destination"] = coord_destin
                    t2.loc[l,"alternative"] = i
                    t2.loc[l,"legs"] = l

                    t3 = pd.DataFrame.from_dict(pd.json_normalize(Walking[i]['legs'][l]['steps']))        
                    for s in range(0, len(t3)):
                        t3.loc[s,"trip_datetime"] = trip_datetime                
                        t3.loc[s,"geo_origin"] = coord_origin
                        t3.loc[s,"geo_destination"] = coord_destin
                        t3.loc[s,"alternative"] = i
                        t3.loc[s,"legs"] = l
                        t3.loc[s,"steps"] = s

                    w3 = pd.concat([w3, t3], ignore_index=True)
                w2 = pd.concat([w2, t2], ignore_index=True)
            w1 = pd.concat([w1, t1], ignore_index=True)

            walking_trips = w2[['trip_datetime',
                                'geo_origin', 
                                'geo_destination', 
                                'alternative', 
                                'distance.value',     
                                'duration.value']]
            walking_trips.columns = [i.replace('.value', '') for i in walking_trips.columns ]
            walking_trips = walking_trips.rename(columns={'distance':'walking_distance', 
                                                          'duration': 'walking_duration'})

        # Biciclying
        b1, b2, b3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

        if len(Bicycling) != 0:
            t1, t2, t3 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
            t1 = pd.DataFrame.from_dict(pd.json_normalize(Bicycling))
            for i in range(0, len(t1)):
                t1.loc[i,"trip_datetime"] = trip_datetime                
                t1.loc[i,"geo_origin"] = coord_origin
                t1.loc[i,"geo_destination"] = coord_destin
                t1.loc[i,"alternative"] = i

                t2 = pd.DataFrame.from_dict(pd.json_normalize(Bicycling[i]['legs']))
                for l in range(0, len(t2)):
                    t2.loc[l,"trip_datetime"] = trip_datetime                
                    t2.loc[l,"geo_origin"] = coord_origin
                    t2.loc[l,"geo_destination"] = coord_destin
                    t2.loc[l,"alternative"] = i
                    t2.loc[l,"legs"] = l

                    t3 = pd.DataFrame.from_dict(pd.json_normalize(Bicycling[i]['legs'][l]['steps']))        
                    for s in range(0, len(t3)):
                        t3.loc[s,"trip_datetime"] = trip_datetime                
                        t3.loc[s,"geo_origin"] = coord_origin
                        t3.loc[s,"geo_destination"] = coord_destin
                        t3.loc[s,"alternative"] = i
                        t3.loc[s,"legs"] = l
                        t3.loc[s,"steps"] = s

                    b3 = pd.concat([b3, t3], ignore_index=True)
                b2 = pd.concat([b2, t2], ignore_index=True)
            b1 = pd.concat([b1, t1], ignore_index=True)

            bicycling_trips = b2[['trip_datetime',
                                'geo_origin', 
                                'geo_destination', 'alternative',
                                'distance.value', 
                                'duration.value']]
            bicycling_trips.columns = [i.replace('.value', '') for i in bicycling_trips.columns ]
            bicycling_trips = bicycling_trips.rename(columns={'distance':'bicycling_trips_distance', 
                                                          'duration': 'bicycling_trips_duration'})

        if len(transit_trips)==0:    
            transit_trips['trip_datetime'] = np.nan
            transit_trips['geo_origin'] = np.nan
            transit_trips['geo_destination'] = np.nan
            transit_trips['alternative']  = np.nan
        if len(driving_trips)==0:
            driving_trips['trip_datetime'] = np.nan
            driving_trips['geo_origin'] = np.nan
            driving_trips['geo_destination'] = np.nan
            driving_trips['alternative']  = np.nan
        if len(walking_trips)==0:
            walking_trips['trip_datetime'] = np.nan
            walking_trips['geo_origin'] = np.nan
            walking_trips['geo_destination'] = np.nan
            walking_trips['alternative']  = np.nan
        if len(bicycling_trips)==0:
            bicycling_trips['trip_datetime'] = np.nan
            bicycling_trips['geo_origin'] = np.nan
            bicycling_trips['geo_destination'] = np.nan
            bicycling_trips['alternative']  = np.nan


        if len(transit_trips) == 0:
            transit_trips = driving_trips.copy()
        else:
            transit_trips = transit_trips.merge(driving_trips, how='left', on=[ 'trip_datetime',
                                                                                'geo_origin', 
                                                                                'geo_destination', 
                                                                                'alternative' ])
        if len(transit_trips) == 0:
            transit_trips = walking_trips.copy()
        else:
            transit_trips = transit_trips.merge(walking_trips, how='left', on=[ 'trip_datetime',
                                                                                'geo_origin', 
                                                                                'geo_destination', 
                                                                                'alternative' ])
        if len(transit_trips) == 0:
            transit_trips = bicycling_trips.copy()
        else:
            transit_trips = transit_trips.merge(bicycling_trips, how='left', on=[ 'trip_datetime',
                                                                                'geo_origin', 
                                                                                'geo_destination', 
                                                                                'alternative' ])

        for i in transit_trips.columns:
            if i.find('duration') > -1:
                transit_trips[i] = (transit_trips[i] / 60).round(2)
            if i.find('distance') > -1:
                transit_trips[i] = (transit_trips[i] / 1000).round(2)
            if i.find('.') > -1:
                transit_trips = transit_trips.rename(columns={i: i.replace('.', '_')})

        if not alternatives:        
            transit_trips = transit_trips.drop(['alternative'], axis=1)

        if len(transit_trips) == 0:
            transit_trips = pd.DataFrame([[trip_datetime, coord_origin, coord_destin]], columns=['trip_datetime', 'geo_origin', 'geo_destination'])
    
    else:
        print('******')
        print('Hay un error con la API de Google Maps, verifique si funciona internet o hay algún problema con la key')
        print('******')
    return transit_trips, gmaps_ok

def gmaps_matrix(od_matrix, geo_origin, geo_destination, trip_datetime, mode, gmaps):
    '''
    Realiza una consulta a la API de Google Maps teniendo en cuenta un dataframe de origenes y destinos. 
    Las consultas se realizan para una matriz de origenes y destinos y devuelve solo tiempos y distancias del viaje.
    
    Parámetros:
    od_matrix = Matriz de origenes y destinos. Debe contener un campo geo_origen y otro geo_destination en formato lat/lon texto para realizar el proceso
    trip_datetime = Fecha y hora de la consulta del viaje en formato datetime
    mode = Según el modo que se quiera consultar, las opciones son: transit, driving, walking, bicycling
    gmaps = Objeto gmaps para acceder a la API de googlemaps
    
    Salida: Matriz de origenes y destinos con los resultados de la consulta a Google Maps
    '''

    trip_datetime_corrected = utils.correct_datetime(trip_datetime, lat=float(od_matrix[geo_origin].head(1).values[0].split(',')[0]), lon=float(od_matrix[geo_origin].head(1).values[0].split(',')[1]))
    
    list_origin = od_matrix[geo_origin].unique().tolist()
    list_destination = od_matrix[geo_destination].unique().tolist()

    try:
        result = gmaps.distance_matrix(list_origin,
                                       list_destination, 
                                       mode=mode,
                                       departure_time=trip_datetime_corrected)
    
    
        t1 = pd.DataFrame.from_dict(pd.json_normalize(result))
        origin_addresses = t1['origin_addresses'][0]
        destination_addresses = t1['destination_addresses'][0]

        trips = pd.DataFrame([])
        n = 0
        for x in t1['rows'][0]:     
            tx = pd.DataFrame.from_dict(pd.json_normalize(x['elements']))    

            tx['origin_address'] = origin_addresses[n]
            tx[geo_origin] = list_origin[n]
            tx = pd.concat([tx,
                   pd.DataFrame(list_destination, columns=[geo_destination]),
                   pd.DataFrame(destination_addresses, columns=['destination_address'])], axis=1)

            tx['trip_datetime'] = trip_datetime

            if not 'distance.value' in tx.columns:
                tx['distance.value'] = np.nan
            if not 'duration.value' in tx.columns:
                tx['duration.value'] = np.nan

            trips = pd.concat([trips, tx], ignore_index=True)
            n+=1

        if len(trips)>0:
            var_list = ['trip_datetime', geo_origin, geo_destination, 'origin_address', 'destination_address', 'distance.value', 'duration.value']
            var_list_new = ['trip_datetime', geo_origin, geo_destination, 'origin_address', 'destination_address', f'{mode}_distance', f'{mode}_duration']
            if mode == 'driving':
                trips['duration_in_traffic.value'] = round(trips['duration_in_traffic.value'] / 60, 2)
                var_list = var_list+['duration_in_traffic.value']
                var_list_new = var_list_new + [f'{mode}_duration_in_traffic']

            trips = trips[var_list]
            trips.columns = var_list_new

            trips[ f'{mode}_distance'] = round(trips[ f'{mode}_distance'] / 1000,2)
            trips[ f'{mode}_duration'] = round(trips[ f'{mode}_duration'] / 60,2)

    except:
        print('')
        print(f'** Error en la API de Google Maps. Revise que esté funcionando correctamente la key utilizada _ modo {mode}')
        print('')
        trips = pd.DataFrame([])
    
    return trips 
    
    
   
    
def trip_info_googlemaps_matrix(od_matrix, 
                                geo_origin,
                                geo_destination,
                                trip_datetime, 
                                gmaps,
                                transit = True, 
                                driving = True, 
                                walking = False, 
                                bicycling = False):
    '''
    Función auxiliar para la creación de la matrix de origenes y destinos con distintos modos de transporte    
    '''
    
    transit_trips, driving_trips, walking_trips, bicycling_trips = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    
    if transit:
        transit_trips = gmaps_matrix(od_matrix, geo_origin, geo_destination, trip_datetime, 'transit', gmaps)
    if driving:
        driving_trips = gmaps_matrix(od_matrix, geo_origin, geo_destination, trip_datetime, 'driving', gmaps)
    if walking:
        walking_trips = gmaps_matrix(od_matrix, geo_origin, geo_destination, trip_datetime, 'walking', gmaps)
    if bicycling:
        bicycling_trips = gmaps_matrix(od_matrix, geo_origin, geo_destination, trip_datetime, 'bicycling', gmaps)

    if len(transit_trips)==0:    
        transit_trips['trip_datetime'] = np.nan
        transit_trips[geo_origin] = np.nan
        transit_trips[geo_destination] = np.nan
        transit_trips['origin_address']  = np.nan
        transit_trips['destination_address']  = np.nan
    if len(driving_trips)==0:
        driving_trips['trip_datetime'] = np.nan
        driving_trips[geo_origin] = np.nan
        driving_trips[geo_destination] = np.nan
        driving_trips['origin_address']  = np.nan
        driving_trips['destination_address']  = np.nan
    if len(walking_trips)==0:
        walking_trips['trip_datetime'] = np.nan
        walking_trips[geo_origin] = np.nan
        walking_trips[geo_destination] = np.nan
        walking_trips['origin_address']  = np.nan
        walking_trips['destination_address']  = np.nan
    if len(bicycling_trips)==0:
        bicycling_trips['trip_datetime'] = np.nan
        bicycling_trips[geo_origin] = np.nan
        bicycling_trips[geo_destination] = np.nan
        bicycling_trips['origin_address']  = np.nan
        bicycling_trips['destination_address']  = np.nan


    if len(transit_trips) == 0:
        transit_trips = driving_trips.copy()
    else:    
        driving_trips = driving_trips.drop(['origin_address', 'destination_address'], axis=1)
        transit_trips = transit_trips.merge(driving_trips, how='left', on=[ 'trip_datetime',
                                                                            geo_origin, 
                                                                            geo_destination])
    if len(transit_trips) == 0:
        transit_trips = walking_trips.copy()
    else:
        walking_trips = walking_trips.drop(['origin_address', 'destination_address'], axis=1)
        transit_trips = transit_trips.merge(walking_trips, how='left', on=[ 'trip_datetime',
                                                                            geo_origin, 
                                                                            geo_destination])
    if len(transit_trips) == 0:
        transit_trips = bicycling_trips.copy()
    else:    
        bicycling_trips = bicycling_trips.drop(['origin_address', 'destination_address'], axis=1)
        transit_trips = transit_trips.merge(bicycling_trips, how='left', on=[ 'trip_datetime',
                                                                            geo_origin, 
                                                                            geo_destination])
    return transit_trips




def trips_gmaps_process(od_matrix, 
                        geo_origin,
                        geo_destination,
                        trip_datetime,                 
                        key, 
                        transit=True,
                        driving=True,
                        walking=False,
                        bicycling=False,
                        normalize=True,
                        res = 8,
                        trips_file = 'trips_file_tmp',
                        full_day=False,
                        only_distance_duration=False,
                        current_path=Path(),
                        city=''):
    
    '''
    Función auxiliar para el proceso de consulta a la API de Google Maps. Se llama de trips_gmaps_from_od y de trips_gmaps_from_matrix
    '''
    
# if True:
#     od_matrix = od_matrix_tmp.head(2).copy() 
#     geo_origin=geo_origin
#     geo_destination=geo_destination 
#     trip_datetime = list_trip_datetime 
#     transit=transit
#     driving=driving
#     walking=walking
#     bicycling=bicycling
#     normalize=normalize
#     res = res    
#     full_day=full_day
#     only_distance_duration=only_distance_duration
#     current_path=current_path
#     trips_file = 'trips_file_tmp'

    utils.create_result_dir(current_path=current_path)    
    
    
    if len(city)>0: trips_file = city+'_'+trips_file
    
    modos = ''
    answer = ''
    if transit: modos += 'Transporte Público, '
    if driving: modos += 'Automovil, '
    if walking: modos += 'Caminata, '
    if bicycling: modos += 'Bicicleta, '
    modos = modos[:-2]

    if transit: trips_file += '_transit'
    if driving: trips_file += '_drive'
    if walking: trips_file += '_walk'
    if bicycling: trips_file += '_bike'

    if only_distance_duration:
        trips_file += '_matrix'
    if normalize:
        trips_file += '_norm'
    
    if 'trip_datetime' in od_matrix.columns: del od_matrix['trip_datetime']
        
    list_trip_datetime = trip_datetime
    if type(trip_datetime) != list:
        list_trip_datetime = [list_trip_datetime]
        
    od_matrix_ = od_matrix.copy()
    od_matrix = pd.DataFrame([])
    for i in list_trip_datetime:
        od_matrix_['trip_datetime'] = pd.to_datetime(i)
        od_matrix = pd.concat([od_matrix, od_matrix_], ignore_index=True)    
    
    if full_day:
        
        od_matrix_full_day = pd.DataFrame([])
        for n in range(0, 24):                    
            od_matrix['trip_datetime'] = od_matrix.trip_datetime.apply(lambda dt: dt.replace(hour=n))
            od_matrix['hour'] = n
            od_matrix_full_day = pd.concat([od_matrix_full_day,
                                            od_matrix], ignore_index=True)
        od_matrix_all = od_matrix_full_day.drop_duplicates().copy()
        
    else:
        od_matrix_all = od_matrix.copy()
    od_matrix_all = od_matrix_all.drop_duplicates()
    _error = False
    trips_all = pd.DataFrame([])
    for trip_datetime in list_trip_datetime:
        trips_file_ = current_path / 'tmp' / Path(f'{trips_file}_{str(trip_datetime)[:10]}.csv'.replace(':', '_').replace(' ', '_'))

        print(' Archivo temporal:', trips_file_)
        if Path(trips_file_).is_file(): 
            try:
                trips_ = pd.read_csv(trips_file_)                 
            except:
                print('-------Error levantando', trips_file_)
                
                trips_all = pd.DataFrame([])
                _error = True
                break
                
            trips_['trip_datetime'] = pd.to_datetime(trips_['trip_datetime'])
            
            trips_['gmaps'] = 1
            
            trips_all = pd.concat([trips_all, trips_], ignore_index=True)
            
    if len(trips_all)>0:
        
        od_matrix_agg = od_matrix_all.merge(trips_all[[geo_origin, 
                                                       geo_destination, 
                                                       'trip_datetime', 
                                                       'gmaps']],
                                            how='left', 
                                            on=[geo_origin, 
                                                geo_destination, 
                                                'trip_datetime'])
        
        od_matrix_agg = od_matrix_agg[(od_matrix_agg.gmaps.isna())&(od_matrix_agg[geo_origin]!=od_matrix_agg[geo_destination])]
    
        del od_matrix_agg['gmaps']
        del trips_all['gmaps']
    else:
        od_matrix_agg = od_matrix_all.copy()

    trips_all_new = trips_all.copy()
    
    if len(od_matrix_agg) > 0:
        
        if 'hour' in od_matrix_agg:
            od_matrix_agg = od_matrix_agg.sort_values(['trip_datetime', 'hour'])
            del od_matrix_agg['hour']
        else:
            od_matrix_agg = od_matrix_agg.sort_values(['trip_datetime'])
                
        try:
            gmaps = googlemaps.Client(key)
            gmaps_ok = True

        except:
            print('')
            print('Hay un error en la configuración de la API de Google Maps')
            print('Verifique que el API Key sea correcto y esté valido y que haya conexión a internet')
            print('')
            gmaps_ok = False
            od_matrix = pd.DataFrame([])

        if gmaps_ok:

            if trip_datetime > datetime.datetime.now():

                weekDaysMapping = ("Lunes", 
                                   "Martes",
                                   "Miércoles", 
                                   "Jueves",
                                   "Viernes", 
                                   "Sábado",
                                   "Domingo")
                monthsMapping = ("", 
                                 "Enero", 
                                 "Febrero", 
                                 "Marzo", 
                                 "Abril", 
                                 "Mayo", 
                                 "Junio", 
                                 "Julio", 
                                 "Agosto", 
                                 "Septiembre", 
                                 "Octubre", 
                                 "Noviembre", 
                                 "Diciembre")

                cost = transit * .01 + driving * .015 + walking * .01 + bicycling*.01
                qty_queries = len(od_matrix_agg[(od_matrix_agg[geo_origin]!=od_matrix_agg[geo_destination])])*(transit * 1 + driving * 1 + walking * 1 + bicycling*1)
                qty_queries_str = '{:,}'.format(qty_queries)
                len_matrix = len(od_matrix)
                len_matrix = '{:,}'.format(len_matrix)
                print('')
                print(f' Para una matriz de origenes y destinos de {len_matrix} viajes se van a realizar {qty_queries_str} consultas en la Api de Google Maps')
                print('')
                print(f' Se van a consultar los modos: {modos} a un costo estimado de USD {round(len(od_matrix_agg)* cost, 2)}')                    
                print('')
                
                print('Las consultas se realizarán para los siguientes días:')
                print('')
                if not full_day:
                    for ii in od_matrix_agg.trip_datetime.unique():                    
                        ii = pd.to_datetime(ii)
                        print(f'        {weekDaysMapping[ii.weekday()]} {ii.day} de {monthsMapping[ii.month]} de {ii.year} a las {str(ii.hour).zfill(2)}:{str(ii.minute).zfill(2)} hs.')
                else:
                    for ii in od_matrix_agg.trip_datetime.dt.date.unique():                    
                        ii = pd.to_datetime(ii)
                        print(f'        {weekDaysMapping[ii.weekday()]} {ii.day} de {monthsMapping[ii.month]} de {ii.year} para las 24 horas del día')
                print('')
                if not only_distance_duration:                    
                    tiempo_estimado = qty_queries * 0.006 
                    
                    if tiempo_estimado > 60:                                         
                        tiempo_estimado = tiempo_estimado / 60
                        tiempo_estimado = f'{int(tiempo_estimado)}:{int(60*(tiempo_estimado - int(tiempo_estimado)) )} horas'
                    else:                        
                        tiempo_estimado = str(int(tiempo_estimado)) + ' minutos'

                    print(f' El tiempo estimado para correr este proceso es de {tiempo_estimado}')
                    print('')


                answer = input("  Ingrese si para continuar ")
                print('')
                if answer.lower() == 'si':
                
                    print('')
                    print(f'Las consultas quedan guardadas en el archivo temporal')
                    print('')
                    
                    gmaps_ok = 1
                    
                    coord_origin = od_matrix_agg.head(1)[geo_origin].values[0]

                    for trip_datetime in od_matrix_agg.trip_datetime.unique():

                        trip_datetime = pd.to_datetime(trip_datetime)

                        trips_file_ = current_path / 'tmp' / Path(f'{trips_file}_{str(trip_datetime)[:10]}.csv'.replace(':', '_').replace(' ', '_'))
                        
                        if len(trips_all) > 0:                            
                            trips = trips_all[trips_all.trip_datetime.dt.date == trip_datetime.date()]
                        else:
                            trips = pd.DataFrame([])
                        
                        if not only_distance_duration:

                            trip_datetime_corrected = utils.correct_datetime(trip_datetime, lat=float(coord_origin.split(',')[0]), lon=float(coord_origin.split(',')[1]))

                            n = 0                
                            for _, i in od_matrix_agg[(od_matrix_agg.trip_datetime==trip_datetime)&(od_matrix_agg[geo_origin]!=od_matrix_agg[geo_destination])].iterrows():        
                                
#                                 print(i[geo_origin], i[geo_destination])

                                trips_new, gmaps_ok = trip_info_googlemaps(i[geo_origin], 
                                                                             i[geo_destination], 
                                                                             trip_datetime = pd.to_datetime(i.trip_datetime), 
                                                                             trip_datetime_corrected = trip_datetime_corrected,
                                                                             gmaps = gmaps, 
                                                                             transit=transit,
                                                                             driving=driving,
                                                                             walking=walking,
                                                                             bicycling=bicycling)
                                
                                if gmaps_ok == 0:
                                    break
                                
                                trips_new = trips_new.rename(columns={'geo_origin':geo_origin, 'geo_destination':geo_destination})
                                
                                sk = save_key(key, (transit+((driving)*1.5)+walking+bicycling), current_path)
                                
                                trips = pd.concat([trips,
                                                   trips_new], 
                                                   ignore_index=True)
                                
                                trips_all = pd.concat([trips_all,
                                                   trips_new], 
                                                   ignore_index=True)
                                
                                n += 1

                                if (n % 100) == 0:

                                    n_queries = n*(transit * 1 + driving * 1 + walking * 1 + bicycling*1)
                                    n_queries = '{:,}'.format(n_queries)

                                    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} procesando {n_queries} de {qty_queries}')
                                    trips.to_csv(trips_file_, index=False)
                            
                            trips.to_csv(trips_file_, index=False)
                            
                            if gmaps_ok == 0:
                                break

                        else:
                        
                            print(f'Procesando {str(pd.to_datetime(trip_datetime))}')

                            trips_new = trip_info_googlemaps_matrix(od_matrix_agg[(od_matrix_agg.trip_datetime==trip_datetime)&(od_matrix_agg[geo_origin]!=od_matrix_agg[geo_destination])], 
                                                                    geo_origin = geo_origin,
                                                                    geo_destination = geo_destination,
                                                                    trip_datetime = trip_datetime, 
                                                                    gmaps = gmaps,
                                                                    transit = transit, 
                                                                    driving = driving, 
                                                                    walking = walking, 
                                                                    bicycling = bicycling)
                            
                            trips_new = trips_new.rename(columns={'geo_origin':geo_origin, 'geo_destination':geo_destination})
                            
                            sk = save_key(key, (transit+((driving)*1.5)+walking+bicycling)*len(trips_new), current_path)

                            trips = pd.concat([trips,
                                               trips_new], 
                                               ignore_index=True)
                            
                            trips_all = pd.concat([trips_all,
                                                   trips_new], 
                                                   ignore_index=True)

                            trips.to_csv(trips_file_, index=False)
                    
                    if gmaps_ok == 1:
                        print('Proceso finalizado')
                    else:
                        print('Error en la API')
                    print('')
                    print('')
                else:
                    trips_all = ''
                    od_matrix_all = pd.DataFrame([])
            else:
                print('')
                print('La fecha/hora del proceso debe ser una fecha/hora mayor a la actual')
                print('')

                trips_all = ''
                od_matrix_all = pd.DataFrame([])


    else:
        
        print('')
        print('')
        print(f'Este proceso ya se corrió con anterioridad. Las consultas están guardadas en los archivos temporales')
        print(f'Puede borrar estos archivo si quiere correr nuevamente el proceso para este mismo día')
        print('')
    
    if len(trips_all) > 0:
        
        od_matrix_all = od_matrix_all.merge(trips_all, 
                                           how='left', 
                                           on=[geo_origin, geo_destination, 'trip_datetime'])

    if geo_origin in od_matrix_all.columns: 
        
        for i in od_matrix_all.columns:
            if od_matrix_all[i].dtype == 'float':        
                od_matrix_all.loc[od_matrix_all[geo_origin] == od_matrix_all[geo_destination], i] = od_matrix_all.loc[od_matrix_all[geo_origin] == od_matrix_all[geo_destination], i].fillna(0)

    return od_matrix_all


def trips_gmaps_from_od(origin, 
                        id_origin, 
                        destination, 
                        id_destination, 
                        population,
                        trip_datetime,                 
                        key, 
                        transit=True,
                        driving=True,
                        walking=False,
                        bicycling=False,
                        normalize=True,
                        res = 8,                        
                        full_day=False,
                        only_distance_duration=False,
                        trips_file = 'trips_file_tmp',
                        current_path=Path(),
                        city=''):
    
    '''
    Calcula distancias en modos de transporte entre dos capas cartográficas de origenes y destinos.
    En el caso de que alguna capa sea de polígonos, toma en cuenta el centroide de cada polígono.
    Devuelve una od_matrix de origenes y destinos con las distancias calculadas para cada modo.

    Parámetros
    origin = GeoDataFrame que representa a los puntos de origen
    id_origin = ID del GeoDataFrame origin
    destination = GeoDataFrame que representa a los puntos de destino
    id_destination = ID del GeoDataFrame destino
    trip_datetime = Fecha/hora de la consulta en formato datetime.
    key = key requerida para consultar la API de Google Maps
    transit = Para obtener distancias y tiempos en transporte público (True/False). Por defecto True.
    driving = Para obtener distancias y tiempos en automovil (True/False). Por defecto True.
    walking = Para obtener distancias y tiempos caminando (True/False). Por defecto False.
    bicycling = Para obtener distancias y tiempos en bicicleta (True/False). Por defecto False.
    normalize = Si normalize es True, toma los origenes y destinos desde el centroide del hexágono h3 con la resolución correspondiente. Este permite minimizar la cantidad de consultas de tiempos de viaje en observaciones que están a una distancia muy cercana. Esto permite reducir los costos de consulta en Google Maps.
    res = Resolución de los hexágonos h3 si normalize = True.    
    full_day = Si se realiza consulta para un día completo o para una sola hora del día. Por defecto es False
    only_distance_duration = Si se quiere consultar solo tiempos y distancias en la matriz. La corrida de este proceso es mucho más rápida pero no incluye todas las variables, como distancias de caminata para acceder al transporte público, tiempos en cada modo, etc.
    trips_file = Nombre de archivo temporario para guardar las consultas a Google Maps. Por defecto: 'trips_file_tmp',
    current_path = Directorio de trabajo
    
    Salida: matriz de origenes y destinos con las variables de tiempo y distancias calculadas para los modos solicitados

    '''
    
#     destination = gpd.sjoin(destination, origin[[id_origin, 'geometry']].rename(columns={id_origin:'tmp_origin'})).drop(['index_right', 'tmp_origin'], axis=1).reset_index(drop=True)
#     if len(destination) > 0:
    
    if not normalize:
        lat_o, lon_o, lat_d, lon_d = 'lat_o', 'lon_o', 'lat_d', 'lon_d'
        geo_origin, geo_destination = 'origin', 'destination'
    else:
        lat_o, lon_o, lat_d, lon_d = 'lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'
        geo_origin, geo_destination = 'origin_norm', 'destination_norm'

    od_matrix_all_ = create_matrix(origin,                                   
                                  destination, 
                                  id_origin = id_origin, 
                                  id_destination = id_destination, 
                                  latlon=False, 
                                  normalize=False, 
                                  duplicates=True,
                                  res = res)

    list_trip_datetime = trip_datetime
    if type(trip_datetime) != list:
        list_trip_datetime = [list_trip_datetime]

    od_matrix_all = pd.DataFrame([])
    for i in list_trip_datetime:
        od_matrix_all_['trip_datetime'] = i
        od_matrix_all = pd.concat([od_matrix_all, od_matrix_all_], ignore_index=True)

    if full_day:
        od_matrix_all_ = od_matrix_all.copy()
        od_matrix_full_day = pd.DataFrame([])
        for n in range(0, 24):                    
            od_matrix_all_['trip_datetime'] = od_matrix_all_.trip_datetime.apply(lambda dt: dt.replace(hour=n))
            od_matrix_all_['hour'] = n
            od_matrix_all = pd.concat([od_matrix_all,
                                       od_matrix_all_], ignore_index=True)
        od_matrix_all = od_matrix_all.sort_values(['trip_datetime', 'hour'])

        del od_matrix_all['hour']

        od_matrix_all = od_matrix_all.drop_duplicates().reset_index(drop=True)        



    od_matrix = create_matrix(origin,                               
                              destination, 
                              id_origin = id_origin, 
                              id_destination = id_destination, 
                              latlon=False,
                              normalize=normalize,
                              duplicates=False,
                              res = res)


    od_matrix = trips_gmaps_process(od_matrix = od_matrix, 
                                    geo_origin=geo_origin,
                                    geo_destination=geo_destination,
                                    trip_datetime = trip_datetime,                 
                                    key = key, 
                                    transit=transit,
                                    driving=driving,
                                    walking=walking,
                                    bicycling=bicycling,
                                    normalize=normalize,
                                    res = res,
                                    trips_file = trips_file,
                                    full_day=full_day,
                                    only_distance_duration=only_distance_duration,
                                    current_path=current_path,
                                    city=city)

    if (not normalize)&(f'{geo_origin}_norm' in od_matrix.columns):
        od_matrix = od_matrix.drop([f'{geo_origin}_norm', f'{geo_destination}_norm'], axis=1)
        od_matrix_all = od_matrix_all.drop([f'{geo_origin}_norm', f'{geo_destination}_norm'], axis=1)



    if len(od_matrix) > 0:

        od_matrix = od_matrix_all.merge(od_matrix, how='left', on=['hex_o', 'hex_d', 'trip_datetime', geo_origin, geo_destination])
        od_matrix['hour'] = od_matrix.trip_datetime.dt.hour
        od_matrix = od_matrix.sort_values(['trip_datetime', 'hour'])
        del od_matrix['hour']

    cols = [i for i in [id_origin, id_destination, 'hex_o', 'hex_d', 'origin', 'destination', 'origin_norm', 'destination_norm', 
                        'area_m2', population, 'PCA_1', 'NSE_5', 'NSE_3', 'weight', 'weight%', 
                        'distance_osm_drive', 'distance_osm_walk',
                         'trip_datetime', 'transit_departure_time', 'transit_arrival_time',
                         'transit_distance', 'transit_duration', 'transit_walking_distance', 'transit_walking_duration', 'transit_walking_steps',
                         'transit_transit_distance', 'transit_transit_duration', 'transit_transit_steps', 'transit_walking_distance_origin',
                         'transit_walking_duration_origin', 'driving_distance',
                         'driving_duration', 'driving_duration_in_traffic',
                         'walking_distance', 'walking_duration',
                         'bicycling_distance', 'bicycling_duration'] if i in od_matrix.columns]  
    od_matrix = od_matrix[cols]
#     else:
#         print('No coinciden los destinos con los origenes')
#         od_matrix = pd.DataFrame([])
    
    return od_matrix
    
def trips_gmaps_from_matrix(od_matrix, 
                            trip_datetime,                 
                            key, 
                            geo_origin = 'origin',
                            geo_destination = 'destination',                        
                            transit=True,
                            driving=True,
                            walking=False,
                            bicycling=False,
                            normalize=True,
                            res = 8,
                            trips_file = 'trips_file_tmp',                        
                            current_path=Path(),
                            city=''):
    '''
    Realiza consulta a la API de Google Maps a partir de una matriz ya existente. El proceso es similar a trips_gmaps_from_od pero con la diferencia que la matriz ya viene como parámetro.
    
    Parámetros
    od_matrix = GeoDataFrame que representa a la matriz de origenes y destinos
    key = key requerida para consultar la API de Google Maps
    geo_origin = Variable con lat/lon origen. Por defecto: 'origin',
    geo_destination = Variable con lat/lon destino. Por defecto: 'destination',
    trip_datetime = Fecha/hora de la consulta en formato datetime.
    transit = Para obtener distancias y tiempos en transporte público (True/False). Por defecto True.
    driving = Para obtener distancias y tiempos en automovil (True/False). Por defecto True.
    walking = Para obtener distancias y tiempos caminando (True/False). Por defecto False.
    bicycling = Para obtener distancias y tiempos en bicicleta (True/False). Por defecto False.
    normalize = Si normalize es True, toma los origenes y destinos desde el centroide del hexágono h3 con la resolución correspondiente. Este permite minimizar la cantidad de consultas de tiempos de viaje en observaciones que están a una distancia muy cercana. Esto permite reducir los costos de consulta en Google Maps.
    res = Resolución de los hexágonos h3 si normalize = True.        
    trips_file = Nombre de archivo temporario para guardar las consultas a Google Maps. Por defecto: 'trips_file_tmp',
    current_path = Directorio de trabajo
    
    Salida: matriz de origenes y destinos con las variables de tiempo y distancias calculadas para los modos solicitados
    '''

    if normalize:        
        geo_origin = geo_origin + '_norm'        
        geo_destination = geo_destination + '_norm'        

        od_matrix['lat_o_norm'] = od_matrix['hex_o'].apply(utils.h3.add_geometry, bring='lat').round(5)
        od_matrix['lon_o_norm'] = od_matrix['hex_o'].apply(utils.h3.add_geometry, bring='lon').round(5)
        od_matrix['lat_d_norm'] = od_matrix['hex_d'].apply(utils.h3.add_geometry, bring='lat').round(5)
        od_matrix['lon_d_norm'] = od_matrix['hex_d'].apply(utils.h3.add_geometry, bring='lon').round(5)
        od_matrix[geo_origin] = od_matrix[f'lat_o_norm'].round(5).astype(str) + ', ' + od_matrix[f'lon_o_norm'].round(5).astype(str)
        od_matrix[geo_destination] = od_matrix[f'lat_d_norm'].round(5).astype(str) + ', ' + od_matrix[f'lon_d_norm'].round(5).astype(str)
        od_matrix = od_matrix.drop(['lat_o_norm', 'lon_o_norm', 'lat_d_norm', 'lon_d_norm'], axis=1)


    
    od_matrix_agg = od_matrix.groupby(['hex_o', 'hex_d', geo_origin, geo_destination], as_index=False).size().drop(['size'], axis=1)
    
    od_matrix_agg = trips_gmaps_process(od_matrix = od_matrix_agg, 
                                        geo_origin=geo_origin,
                                        geo_destination=geo_destination,
                                        trip_datetime = trip_datetime,                 
                                        key = key, 
                                        transit=transit,
                                        driving=driving,
                                        walking=walking,
                                        bicycling=bicycling,
                                        normalize=normalize,
                                        res = res,
                                        trips_file = trips_file,
                                        current_path=current_path,
                                        city=city)
    
    
    
    
    if len(od_matrix_agg) > 0:
        od_matrix = od_matrix.merge(od_matrix_agg, on=['hex_o', 'hex_d', geo_origin, geo_destination])
        

    return od_matrix

def save_key(key, Qty=0, current_path=Path(), key_file = 'save_key.csv'):
    '''
    Guarda cantidad de consultas en Google Maps en un archivo auxiliar    
    '''
    save_key_file = current_path / 'tmp' / key_file
    n = 0
    new_key = ''
    for i in range(0,39, 5):
        if not n % 2:
            new_key += key[i:i+5]
        else:
            new_key += '....'    
        n += 1
    new_key

    try:

        tmp = current_path / 'tmp'
        if save_key_file.is_file():        
            save_key = pd.read_csv(save_key_file)  
        else:
            save_key = pd.DataFrame([[new_key, str(date.today())[:7], 0]], columns=['key', 'month', 'Qty'])

        if len(save_key[(save_key.key==new_key)&(save_key.month==str(date.today())[:7])]) == 0:
            save_key = pd.concat([save_key, 
                                  pd.DataFrame([[new_key, str(date.today())[:7], 0]], columns=['key', 'month', 'Qty'])], ignore_index=True)

        save_key.loc[(save_key.key==new_key)&(save_key.month==str(date.today())[:7]), 'Qty'] += Qty

        save_key['usd'] = save_key['Qty'] * .01

        save_key.to_csv(save_key_file, index=False)
    except:
        print('except')
        save_key = pd.DataFrame([[new_key, str(date.today())[:7], 0]], columns=['key', 'month', 'Qty'])
        save_key.to_csv(save_key_file, index=False)
    
    save_key['X'] = ''    
    save_key.loc[(save_key.key==new_key)&(save_key.month==str(date.today())[:7]), 'X'] = 'X'
    
    return save_key

def distances_to_equipments(origin,
                            destination, 
                            id_origin,
                            id_destination,                            
                            trip_datetime,
                            key,                           
                            geo_origin = 'origin',
                            geo_destination = 'destination',
                            processing='pandana',
                            equipement_bring_closest = True,
                            equipment_closest_qty = 2,
                            equipment_type = '',
                            normalize=True,
                            closest_distance=[800, 1500, 2000],
                           current_path=Path(),
                           city=''):
    '''
    Calcula las distancias entre una capa geográfica de origenes y un capa de destinos que corresponda a algún tipo de equipamiento (ej. escuelas, hospitales, etc)
    En una primera instancia se calcula una matriz de distancia en OpenStreetMaps desde el centroide de la capa de origen hacia el establecimiento (capa destino)
    Una vez calculadas las distancias para cada origen se obtienen los equipment_closest_qty más cercanos. 
    Una vez seleccionados los más cercanos se realiza la consulta de tiempos de viaje en transporte público o caminando utilizando Google Maps.
    
    Parámetros
    origin = capa de origen (Ejemplo, capa de hexágonos h3)
    destination = capa de destinos
    id_origin = id capa de origen
    id_destination = id capa de destino
    trip_datetime = fecha y hora para realizar la consulta en Google Maps
    key = key para la API de google maps
    geo_origin = Variable con lat/lon origen. Por defecto: 'origin',
    geo_destination = Variable con lat/lon destino. Por defecto: 'destination',
    equipement_bring_closest = Si trae los más cercanos. Por defecto True,
    equipment_closest_qty = Cantidad de equipamientos más cercanos de cada origen. Por defecto 2.
    equipment_type = Si los equipamientos se encuentran categorizados según alguna variable (ej. escuelas primarias y escuelas secundarias). Se van a traer los equipamiento más cercano para cada tipo en forma independiente.
    normalize = Si se quiere utilizar el centroide del hexágono h3 de destino en vez de la ubicación exácta. Esto permite reducir la cantidad y el costo de las consultas a la API de Google Maps.
    closest_distance = Genera una variable con la cantidad de establecimientos en un rango de distancia para cada origen. Por defecto: closest_distance = [800, 1500, 2000] 
    
    Salida
    Matriz de tiempos y distancias de viaje a los establecimientos más cercanos.
    '''

    
#     destination = gpd.sjoin(destination, origin[[id_origin, 'geometry']].rename(columns={id_origin:'tmp_origin'})).drop(['index_right', 'tmp_origin'], axis=1).reset_index(drop=True)
#     if len(destination) > 0:
    
    print('Calcula distancias en Open Street Maps')
    od_matrix_osm = measure_distances_osm(origin = origin, 
                                          id_origin = id_origin, 
                                          destination = destination, 
                                          id_destination = id_destination,           
                                          processing=processing, 
                                          equipement_bring_closest = equipement_bring_closest,
                                          equipment_closest_qty = equipment_closest_qty,
                                          closest_distance = closest_distance,
                                          equipment_type = equipment_type,
                                          current_path=current_path,
                                          city=city)



    print('')
    print('Calcula tiempos en transporte público con Google Maps')
    print('')
    od_matrix_transit =   trips_gmaps_from_matrix(od_matrix = od_matrix_osm.copy(), 
                                                    trip_datetime = trip_datetime,                 
                                                    key = key, 
                                                    geo_origin = geo_origin,
                                                    geo_destination = geo_destination,                        
                                                    transit=True,
                                                    driving=False,
                                                    walking=False,
                                                    bicycling=False,
                                                    normalize=normalize,
                                                    current_path=current_path,
                                                    city=city)

    print('')
    if 'transit_duration' in od_matrix_transit.columns:
        od_matrix_transit['total_duration'] = od_matrix_transit['transit_duration']

        od_matrix_transit['modo'] = 'walk'
        od_matrix_transit.loc[(od_matrix_transit.transit_transit_duration.notna())&(od_matrix_transit.transit_transit_duration != 0), 'modo'] = 'transit'

        if 'distance_osm_walk' in od_matrix_transit.columns:

            tmp = od_matrix_transit.loc[(od_matrix_transit.transit_duration.notna())&(od_matrix_transit.transit_transit_duration.isna()), :].groupby(['origin_norm', 'destination_norm', 'transit_distance', 'transit_duration'], as_index=False).size().drop(['size'], axis=1)

            tmp['kmh_walk'] = round(tmp['transit_distance'] / (tmp['transit_duration'] / 60), 2)

            kmh_walk = 4.8
            if len(tmp)>=100:            
                kmh_walk = round(tmp.kmh_walk.mean(), 1)

            od_matrix_transit['duration_osm_walk'] = round(od_matrix_transit['distance_osm_walk'] / kmh_walk * 60, 1)

            od_matrix_transit.loc[(od_matrix_transit.total_duration.isna())|(od_matrix_transit.total_duration==0), 'total_duration'] = od_matrix_transit['duration_osm_walk']

            if len(equipment_type)==0: equipment_type=[]
            if type(equipment_type)==str: equipment_type=[equipment_type]

            od_matrix_transit = od_matrix_transit.sort_values(equipment_type + [id_origin, 'total_duration'])
            od_matrix_transit['duration_order'] = od_matrix_transit.groupby(equipment_type + [id_origin]).transform('cumcount')
            od_matrix_transit['duration_order'] = od_matrix_transit['duration_order'] + 1

            od_matrix_transit = od_matrix_transit[od_matrix_transit.duration_order==1].reset_index(drop=True)

            var_osm = od_matrix_osm.columns.tolist()
            qty_var = []
            for x in closest_distance:
                var = f'qty_est_{x}m'
                var_osm.remove(var)
                qty_var.append(var)

            if 'distance' in var_osm: var_osm.remove('distance')
            if 'distance_order' in var_osm: var_osm.remove('distance_order')
            if 'duration' in var_osm: var_osm.remove('duration')
            if 'origin' in var_osm: var_osm.remove('origin')
            if 'destination' in var_osm: var_osm.remove('destination')
            if 'distance_osm_walk' in var_osm: var_osm.remove('distance_osm_walk')

            vars_od = [geo_origin, geo_destination]
            if normalize:
                vars_od += [f'{geo_origin}_norm', f'{geo_destination}_norm']

            vars_od = [i for i in vars_od if i not in var_osm]
            od_matrix_transit = od_matrix_transit[var_osm+vars_od+['trip_datetime', 'modo', 'distance_osm_walk', 'total_duration']+qty_var]

            od_matrix_transit = od_matrix_transit.rename(columns={'distance_osm_walk': 'distance', 'total_duration':'duration'})

    cols = [i for i in od_matrix_transit.columns if i not in [id_origin, id_destination, 'hex_o', 'hex_d', 'origin', 'destination', 'origin_norm', 'destination_norm']]
    cols = [id_origin, id_destination, 'hex_o', 'hex_d', 'origin', 'destination', 'origin_norm', 'destination_norm'] + cols
    od_matrix_transit = od_matrix_transit[cols]

    return od_matrix_transit
    
    
def calc_pop(hx, 
             ring_size=1, 
             df='',
             population=''):    
    rings = h3.k_ring(hx, ring_size)    
    res = df[df.hex.isin(rings)][population].sum().astype(int)
    return res

def calc_green_area(hx, 
                    ring_size=1, 
                    df=''):
    rings = h3.k_ring(hx, ring_size)    
    res = df[df.hex.isin(rings)]['green_area_m2_pc'].sum().round(2)
    return res

def var_spatial_lag(hx, 
                   ring_size=1, 
                   var='', 
                   df=''):
    rings = h3.k_ring(hx, ring_size) 
    rings.remove(hx)
    res = round(df[df.hex.isin(rings)][var].mean(), 2)
    return res

def var_spatial_kde(hx, 
                   ring_size=2, 
                   var='', 
                   df='',
                   kde_fex=.4):
    
    fex = 1
    kde_val = []
    kde_w = []
    remove = []
    kde_fex = 1-kde_fex
   
    for i in range(0, ring_size+1):
        
        rings = list(h3.k_ring(hx, i))  

        for x in remove:
            rings.remove(x)

        kde_val += [ round(df.loc[df.hex.isin(rings), var].mean(), 2)]
        kde_w += [fex]
        
        fex = fex*kde_fex
        remove += rings
        
       
        res = round(np.average(kde_val, weights=kde_w), 2)
    return res

def calculate_green_space(df, 
                          city_crs, 
                          population, 
                          max_distance = [1200, 2000], 
                          green_space='', 
                          osm_tags={'leisure': ['park', 'playground', 'nature_reserve', 'recreation_ground']},
                          current_path=Path(),
                          city='',
                          run_always=False):
    '''
    Calculo el área de espacio verde y el espacio verde per-cápita en un radio determinado.
    
    df = capa de hexágonos h3. Tiene que contener un campo de población
    city_crs = Proyección adecuada en metros
    population = variable de población de la capa de hexágonos
    max_distance = Distancias a las que se quiere realizar el cálculo. Es el área a partir del centroide del hexágono. Por defecto max_distance = [1000, 2000]
    green_space = Capa geográfica con parques o espacios públicos. De encontrarse esta capa vacia se obtiene de OpenStreetMaps con los parámetros osm_tags={'leisure': ['park', 'playground', 'nature_reserve', 'recreation_ground']} 
    
    Salida
    Capa geográfica con el area y area per cap de espacios verdes en los rangos seleccionados para cada hexágono.
    '''

    file = current_path / 'Resultados_files' / f'{city}_greens.geojson'
    
    if (Path(file).is_file())&(not run_always):
        df = gpd.read_file(file)
        print('El cálculo de espacios verdes ya existía en', file)
        print('Si cambió algún parámetro o quiere correr el proceso de nuevo cambia run_always=True')
    else:

        if type(max_distance) != list:
            max_distance = [max_distance]

        # Traigo spacios verdes y públicos de OSM
        if len(green_space)==0:       
            green_space = pyomu.utils.bring_osm(df, tags = osm_tags)
            green_space = green_space[(green_space.geom_type == 'Polygon')|(green_space.geom_type == 'MultiPolygon')]

        green_space[f'green_area_m2'] = green_space.to_crs(city_crs).area.round(1)
        green_space[f'green_area_m2'] = green_space[f'green_area_m2'].fillna(0)
        green_space = green_space[green_space[f'green_area_m2']>=5000]

        green_space['aux'] = 1
        green_space = green_space.dissolve(by='aux')
        green_space = gpd.overlay(df[['hex', population, 'geometry']], green_space[['osmid', 'geometry']], how='intersection')
        green_space['osmid_order'] = green_space.groupby(['osmid']).transform('cumcount')
        green_space[f'green_area_m2'] = green_space.to_crs(city_crs).area.round(1)
        
        df = df.merge(green_space[['hex', 'green_area_m2']], how='left', on='hex')
        df[f'green_area_m2'] = df[f'green_area_m2'].fillna(0)


        for max_dist in max_distance:        

            print('Calculo para distancia de', max_dist)
            
            ring_size = round(max_dist / (h3.edge_length(resolution=9, unit='m') * 2))
            
            # Calculo la población en el rango de distancia
            df[f'pop_in_{max_dist}'] = df.hex.apply(calc_pop, 
                                                    ring_size=ring_size,
                                                    df=df.copy(),
                                                    population=population)
            
            # calculo areas verdes per capita para cada espacio verde en el hexágono
            df['green_area_m2_pc'] = df['green_area_m2'] / df[f'pop_in_{max_dist}']
            
            df.loc[df.green_area_m2_pc == np.inf, 'green_area_m2_pc'] = 0
            
            # Sumo los m2 de áreas verdes para cada hexágano según el rango de distancia
            df[f'green_area_m2_pcap_in_{max_dist}'] = df.hex.apply(calc_green_area, 
                                                                      ring_size=ring_size,
                                                                      df=df.copy())
            
            # Calculo el promedio de los vecinos para suavizar
            df[f'green_area_m2_pcap_in_{max_dist}_lag'] = df.hex.apply(var_spatial_lag, 
                                                                          ring_size=1, 
                                                                          var=f'green_area_m2_pcap_in_{max_dist}',
                                                                          df=df.copy())       
            
            df[f'green_area_m2_pcap_in_{max_dist}_kde'] = df.hex.apply(var_spatial_kde, 
                                                                      ring_size=1, 
                                                                      var=f'green_area_m2_pcap_in_{max_dist}',
                                                                      df=df.copy(),
                                                                      kde_fex=.4)       
            
            
            df = df.drop([f'pop_in_{max_dist}', 'green_area_m2_pc'], axis=1)
            
            df.to_file(file)
        
    return df