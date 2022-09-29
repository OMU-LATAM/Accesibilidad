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
warnings.filterwarnings("ignore")
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
from pyomu.vizuals import vizuals as viz

    
def calculate_nse(df, vars, population, var_result='PCA_1', scaler_type='Standard', q=[5, 3], order_nse='', show_map=False):
    '''
    Realiza el análisis de Componentes Principales (PCA) para un conjunto de variables en un dataframe. Este resultado se utiliza para la
    creación de una variable de nivel socioeconómico
    
    Parámetros: 
    df = dataframe que contiene las variables que se utilizan para calcular el PCA y nivel socieconómico. El dataframe puede ser de hexágonos, 
    radios censales u otra geometría con datos de los hogares o población.
    vars = variables que se van a utilizar para el calculo de PCA
    population = variable población.
    var_result= variable donde se alojará el resultado del PCA. Por defecto será 'PCA_1'
    scaler_type=Tipo de normalización para el PCA. Por defecto será 'Standard'
    q = Cantidad de agrupaciones para el nivel socioeconómico resultado. Por defecto serán quintiles y terciles = [5, 3]
    show_map = Muestra mapa resultado, por defecto es False
    
    Salida: DataFrame similar al de entrada con nuevas variables que contienen el PCA y los grupos solicitados (i.e. 'PCA_1', 'NSE_5', 'NSE_3')
    '''
    
    df = df[df[vars].notna().all(axis=1)].reset_index(drop=True)
    df.loc[df[population].isna(), population] = 0
    
    if type(var_result) == str:
        var_result = [var_result]
    
    for i in var_result:
        if i in df.columns:
            df.drop([i], axis=1, inplace=True)
    
    data_1 = df[vars]
    if scaler_type!='None':
        if scaler_type=='Standard':        
            scaler = StandardScaler()
        if scaler_type=='Robust':        
            scaler = RobustScaler()
        
        data_1 = scaler.fit_transform(data_1)

    # Importamos la clase PCA del modulo decomposition de la librería Sklearn.

    # Instanciamos la clase pidiendo que conserve la cantidad de componentes requerida
    pca_2cp = PCA(n_components=len(var_result), svd_solver='full')

    # Con el método fit() calculammos los componentes principales
    principalComponents = pca_2cp.fit_transform(data_1)
    
    principalDf = pd.DataFrame(data = principalComponents, 
                               columns = var_result)

    df_result=pd.concat([df, principalDf], axis=1)

    print('variance ratio', round(pca_2cp.explained_variance_ratio_[0], 2))
    print('(% de la variancia explicada por el componente 1)')
    print('')
    
    
    # Calcula el NSE ponderado por la población (puedo calcular NSE para varios q según parámetros)
    if type(q) == int: 
        q = [q]
        
    for i in q:    
        
        df_result[f'NSE_{i}'] = utils.weighted_qcut(df_result[var_result[0]], df_result[population], i, labels=False)
        df_result[f'NSE_{i}'] = df_result[f'NSE_{i}'] + 1
    

    if len(order_nse) > 0:
        for i in range(0, len(order_nse)):
            
            lst = [x+1 for x in range(0, q[i])]
            ord = order_nse[i]        
            ord = [f'{x+1} - {ord[x]}' for x in range(0, len(ord))]        
            df_result[f'NSE_{q[i]}'] = df_result[f'NSE_{q[i]}'].replace(lst, ord)
            
    
    if show_map:
        
        df_result['NSE_X'] = df_result[f'NSE_{i}'].replace({'1 - Alto':'Alto', '2 - Medio-Alto':'Medio-Alto', '3 - Medio':'Medio',  '4 - Medio-Bajo':'Medio-Bajo', '5 - Bajo':'Bajo'})
        if df_result['NSE_X'].dtype == 'O':
            df_result['NSE_X'] = pd.CategoricalIndex(df_result['NSE_X'], categories= ['Alto', 'Medio-Alto', 'Medio', 'Medio-Bajo', 'Bajo'])
        
        fig, ax = plt.subplots(dpi=150, figsize=(5, 5))
        df_result.to_crs(3857).plot(column='NSE_X', 
                                    ax=ax, 
                                    cmap='Reds',
                                    categorical=True,
                                    legend=True,                            
                                    legend_kwds={'loc': 'best', 'frameon': True, 'edgecolor':'white', 'facecolor':None, "title":'NSE', 'title_fontsize':8, 'fontsize':7})

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=3)
        ax.axis('off');
        del df_result['NSE_X']
    
    return df_result
    
    


def distribute_population(gdf, 
                          id_gdf, 
                          hexs, 
                          id_hexs, 
                          population, 
                          pca, 
                          crs, 
                          q=[5, 3], 
                          order_nse='',
                          verify_overlay=True,
                          verify_overlay_df = '',
                          verify_overlay_df_id = '',
                          tags_osm = {'leisure':True, 'aeroway':True, 'natural':True},
                          show_map=False):

    
    
    '''
    Distribuye la población y el resultado del PCA de una capa geográfica principal a una segunda capa geográfica. Por ejemplo,
    permite distribuir la población y el PCA calculado para una capa geográfica censal en una capa geográfica de hexágonos.
    La población la distribuye según el área ocupada de cada polígono en relación a los polígonos de la segunda capa y el PCA para la 
    segunda capa geográfica lo calcula como un promedio ponderado (usando la variable población) de los PCAs de la capa principal.
    A su vez, permite verificar si existe superposición con otra capa geográfica si es posible identificar areas donde no hay población.
    Por ejemplo, esto sirve si se pueden identificar espacios verdes o públicos para asignar mejor la población en áreas donde no existen este tipo de
    equipamientos. De no existir esta capa geográfica, se pueden obtener y utilizar una capa obtenida de Open Street Maps. Por defecto obtiene los elementos
    relacionados con recreación, aeronavegación y espacios naturales.
    
    gdf = Capa geográfica principal (por ejemplo, una capa de radios censales)
    id_gdf = ID de la capa principal
    hexs = Capa geográfica secundaria (por ejemplo, una capa de hexágonos)
    id_hexs = ID de la capa de secundaria
    population = variable donde está la población de cada polígono en la capa principal
    pca = variable donde está el pca de cada polígono en la capa principal
    crs = Proyección correspondiente a la capa en metros (es importante que no sea un proyección de en grados)
    q=[5, 3] = Resultado de las variables de nivel socioeconómico que se quiere contruir. Por defecto se construyen quintiles y terciles. q = [5, 3]
    verify_overlay = Verifica que exista superposición con otra capa geográfica donde se puede identificar que no hay población (por ejemplo parques o espacios naturales)
    verify_overlay_df = GeoDataFrame con la capa geográfica que se quiere verificar la superposición
    verify_overlay_df_id = ID del GeoDataFrame con la capa geográfica que se quiere verificar la superposición
    tags_osm = Si se quiere obtener la información de superposición de Open Street Mapas. Por defecto trae los siguientes elementos: 
    tags_osm = {'leisure':True, 'aeroway':True, 'natural':True},
    show_map=Permite mostrar un mapa con el resultado del nuevo NSE calculado para la capa geográfica secundaria.
    
    Salida: Devuelve la capa geográfica secundaria con la distribución de las variables de población, PCA y NSE calculadas teniendo en cuenta la capa geográfica principal 
    '''
    
    gdf.loc[gdf[population].isna(), population] = 0
    
    shape = gpd.overlay(hexs[[id_hexs, 'geometry']], gdf[[id_gdf, population, pca, 'geometry']], how='intersection')

    shape['area_interception'] = shape.to_crs(crs).area

    if verify_overlay:
        if len(verify_overlay_df)==0:
            verify_overlay_df_id = 'osmid'
            verify_overlay_df = utils.bring_osm(gdf, tags = {'leisure':True, 'aeroway':True, 'natural':True}, list_types = ['Polygon', 'MultiPolygon'] )

    shape_space_not_available = gpd.overlay(shape[[id_hexs, id_gdf, 'geometry']], verify_overlay_df[['osmid', 'geometry']], how='intersection')
    shape_space_not_available = shape_space_not_available[[id_hexs, id_gdf, 'geometry']].dissolve(by=[id_hexs, id_gdf]).reset_index()
    shape_space_not_available['area_not_available'] = shape_space_not_available.to_crs(crs).area


    shape = shape.merge(shape_space_not_available[[id_hexs, id_gdf, 'area_not_available']], on=[id_hexs, id_gdf], how='left').fillna(0)

    shape['area_interception_result'] = (shape['area_interception'] - shape['area_not_available']) 
    
    shape['area_gdf_result'] = shape.groupby(id_gdf).area_interception_result.transform(sum)

    shape['distribute_population'] = shape['area_interception_result'] / shape['area_gdf_result']

    shape[population] = (shape[population] * shape['distribute_population']).round()
    

    df_result = shape.groupby(id_hexs, as_index=False)[population].sum().merge(
                        shape[shape[population]>0].groupby(id_hexs).apply(lambda x: np.average(x[pca], 
                                                                                               weights=x[population])).reset_index().rename(columns={0:pca}), 
                                                                                               how='left' )

    
    df_result = hexs[[id_hexs, 'geometry']].merge(df_result, how='left')
    df_result['area_m2'] = df_result.to_crs(crs).area.round()
    df_result['density_ha'] = round(df_result[population] / (df_result['area_m2']/10000),1)
    df_result = df_result[[id_hexs, 'area_m2', population, pca, 'geometry']]
    
    # Calcula el NSE ponderado por la población (puedo calcular NSE para varios q según parámetros)

    if type(q) == int: 
        q = [q]
    
    for i in q:            
        df_result.loc[df_result[pca].notna(), f'NSE_{i}'] = utils.weighted_qcut(df_result.loc[df_result[pca].notna(), pca], df_result.loc[df_result[pca].notna(), population], i, labels=False)
        df_result.loc[df_result[pca].notna(), f'NSE_{i}'] = df_result.loc[df_result[pca].notna(), f'NSE_{i}'] + 1
    
    
    for i in range(0, len(order_nse)):            
        lst = [x+1 for x in range(0, q[i])]      
        ord = order_nse[i]        
        ord = [f'{x+1} - {ord[x]}' for x in range(0, len(ord))]        
        df_result.loc[df_result[pca].notna(), f'NSE_{q[i]}'] = df_result.loc[df_result[pca].notna(), f'NSE_{q[i]}'].replace(lst, ord)
    
    
    if show_map:
        i = q[0]
        
        fig = plt.figure(figsize=(15,15), dpi=100)
        
        sns.set_style("white")
        
        fig.suptitle('NSE', fontsize=16)
        
        ax = fig.add_subplot(2,2,1)
        
        gdf.to_crs(3857).plot(column=f'NSE_{i}', 
                                    ax=ax, 
                                    cmap='Reds',
                                    categorical=True,
                                    legend=True,                            
                                    legend_kwds={'loc': 'best', 
                                                 'frameon': True, 
                                                 'edgecolor':'white', 
                                                 'facecolor':None, 
                                                 'title':'NSE', 
                                                 'title_fontsize':8, 
                                                 'fontsize':7})
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=5)
        
        ax.axis('off');

        ax = fig.add_subplot(2,2,2)
        
        
        df_result['NSE_X'] = df_result[f'NSE_{i}'].replace({'1 - Alto':'Alto', '2 - Medio-Alto':'Medio-Alto', '3 - Medio':'Medio',  '4 - Medio-Bajo':'Medio-Bajo', '5 - Bajo':'Bajo'})
        if df_result['NSE_X'].dtype == 'O':
            df_result['NSE_X'] = pd.CategoricalIndex(df_result['NSE_X'], categories= ['Alto', 'Medio-Alto', 'Medio', 'Medio-Bajo', 'Bajo'])
                
        df_result.to_crs(3857).plot(column=f'NSE_X', 
                                    ax=ax, 
                                    cmap='Reds',
                                    categorical=True,
                                    legend=True,                            
                                    legend_kwds={'loc': 'best', 
                                                 'frameon': True, 
                                                 'edgecolor':'white', 
                                                 'facecolor':None, 
                                                 'title':'NSE', 
                                                 'title_fontsize':8, 
                                                 'fontsize':7})
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=5)
        
        ax.axis('off');
        del df_result['NSE_X']
        
        
    
    return df_result




def assign_weights(amenities):
    '''
    Asigna ponderación a los objetos espaciales de Open Street Maps. Esta ponderación se va a utilizar para la clusterización e identificación de principales atracctores de actividad.
    Parámetros
        amenities = GeoDataFrame obtenido de OSMNX con objetos espaciales de Open Street Maps.

    Salida: agrega variable weight con la ponderación asignada a cada objeto espacial.
    '''
    
    
    cat0 = ['college',
            'university',
            'hospital',
            'clinic',        
            'bank',
            'bus_station',
            'ferry_terminal',
            'mall',
            'centro comercial',
            'multiple uso comercial',
            'juzgados',
            'courthouse',
            'public_building',
            'government',
            'register_office',
            'defensa'
            ]

    cat1 = ['education',
            'school',
            'primaria',
            'library',
            'laboratory',        
            'factory',
            'community_hall',
            'community_centre',
            'post_office'
            ]

    cat2 = [
            'police',
            'Prefectura',
            'first_aid',        
            'supermarket',
            ]


    cat3 = ['auditorium',
            'arts',
            'theatre',
            'cinema',
            'exhibition_centre',
            'comunity_center',
            'conference_center',
            'convention_centre',
            'music_venue',
            'comercio',
            'store',
            'shop',
            'comercial',
            'boutique',      
            'childcare',
            'kindergarten',
            'jardin infantil',
            'church',
            'place_of_worship',
            'coworking_space',
            'doctors',      
            'embassy',
            'emergency_service',
            'entertainment',   
            'fast_food',
            'cafeteria',
            'restaurant',
            'bar',
            'gambling',
            'gimnasium',
            'gym',
            'social_club',
            'sport_centre',
            'laboratorio',
            'odontologia',
            'office',
            'oficina',
            'payment_centre'
            'peluqueria'
            'pharmacy',
            'recreation',        
            'pharmacy'        
    ]

    tags_weighted = pd.DataFrame()
    w = [100, 50, 20, 10]
    n = 0
    for i in [cat0, cat1, cat2, cat3]:
        tags_ = pd.DataFrame(i, columns=['tag_type'])
        tags_['weight'] = w[n]
        tags_weighted = pd.concat([tags_weighted, 
                                   tags_], ignore_index=True)
        n += 1

    amenities['tag_type'] = amenities.tag_type.apply(utils.normalize_words)

    for i in tags_weighted.sort_values('weight', ascending=True).itertuples():   
        
        amenities.loc[amenities.tag_type.str.contains(i.tag_type), 'weight'] = i.weight

    amenities['weight'] = amenities['weight'].fillna(1)
    
    amenities = amenities[amenities.weight > 0].reset_index(drop=True)

    return amenities.sort_values('tag_type').reset_index(drop=True)


def activity_density(amenities, city_crs, cantidad_clusters = 15, show_map = False): 
    '''
    Teniendo en consideración un GeoDataFrame de equipamientos se identifican los clusters de alta densidad de establecimientos.
    Se sugiere la obtención del GeoDataFrame de Open Stree Maps y se la asigna una ponderación a cada establecimiento (ver funciones bring_osm y assign_weights). 
    Este GeoDataFrame puede ser reemplazo por alguna otra fuente. De no existir la variable de ponderación (weight), se le asigna 1 a cada registro.
    
    Parámetros:
    amenities = GeoDataFrame con los equipamientos
    city_crs = proyección en metros de la ciudad de análisis para el cálculo de distancias entre establecimientos
    cantidad_clusters = Cantidad de clusters que se quiere obtener. Por defecto 15.
    show_map = Muestra mapa resultado del proceso con los clusters de actividad. Por defecto False
    
    Salida: Devuelve un nuevo GeoDataFrame con los principales clusters de actividad
    '''

    eps_ = [250, 500, 750, 1000]
    samples_ = list(range(500, 3000, 200)) 

    amenities['geometry'] = amenities['geometry'].representative_point()
    
    if 'weight' not in amenities.columns:
        amenities['weight'] = 1

    amenities['x'] = amenities.to_crs(city_crs).geometry.centroid.x
    amenities['y'] = amenities.to_crs(city_crs).geometry.centroid.y
    X = amenities.reindex(columns = ['x','y']).values
    W = amenities.weight

    scores = pd.DataFrame([])

    for eps, samples in tqdm(list(itertools.product(eps_, samples_)), desc='Densidad de actividad'):

        cluster_name = f'cluster_{eps}_{samples}'

        clustering = DBSCAN(eps=eps, 
                            min_samples=samples 
                            ).fit(X, sample_weight=W)

        amenities[cluster_name] = clustering.labels_

        try:
            sc = metrics.silhouette_score(X, clustering.labels_).round(2)
            exception = False
        except:
            exception = True
            sc = -1

        if len(amenities[amenities[cluster_name]>-1])>0:
            cant_clusters = len(amenities.loc[(amenities[cluster_name]>-1),cluster_name].unique())

            result = amenities[amenities[cluster_name]>-1].groupby(cluster_name, as_index=False).agg({'weight':'sum', 'x': 'mean', 'y':'mean'}).sort_values('weight', ascending=False)

            result['weight'] = result['weight'] / amenities['weight'].sum() * 100

            scores = pd.concat([scores, 
                                pd.DataFrame([[cluster_name, 
                                               eps, 
                                               samples, 
                                               sc, 
                                               cant_clusters, 
                                               result.head(1).weight.values[0], 
                                               result.tail(1).weight.values[0], 
                                               exception]], 
                                            columns=['cluster', 
                                                     'eps', 
                                                     'samples', 
                                                     'score', 
                                                     'cant_clusters', 
                                                     'max_weight', 
                                                     'min_weight', 
                                                     'exception'])], ignore_index=True)

    scores = scores[(scores.cant_clusters>=cantidad_clusters)].sort_values(['eps', 'cant_clusters']).reset_index(drop=True)

    cluster_name = scores.head(1).cluster.values[0]

    result = amenities[amenities[cluster_name]>-1].groupby(cluster_name, as_index=False).agg({'weight':'sum'}).sort_values('weight', ascending=False)
    result = result.merge(
                    amenities[amenities[cluster_name]>-1].groupby(cluster_name).apply(lambda x: np.average(x['x'], weights=x['weight'])).reset_index().round(3).rename(columns={0:'x'}))
    result = result.merge(
                    amenities[amenities[cluster_name]>-1].groupby(cluster_name).apply(lambda x: np.average(x['y'], weights=x['weight'])).reset_index().round(3).rename(columns={0:'y'}))

    
    result['weight%'] = (result['weight'] / result['weight'].sum() * 100).round(1) 
    
    result = result.sort_values('weight', ascending=False).head(cantidad_clusters).reset_index().rename(columns={'index':'cluster'})
    
    result = gpd.GeoDataFrame(
                    result, geometry=gpd.points_from_xy(result['x'], result['y']), crs=city_crs).to_crs(4326)
    
    
    if show_map:
        fig = plt.figure(figsize=(15,15), dpi=100)

        ax = fig.add_subplot(2,2,1)
        amenities.to_crs(3857).plot(ax=ax, alpha=0)
        amenities[amenities[cluster_name]>-1].to_crs(3857).plot(ax=ax, column=cluster_name, categorical = True, lw=.1, alpha=.4)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=3)
        ax.set_title(cluster_name, fontsize=8)
        ax.axis('off');


        ax = fig.add_subplot(2,2,2)
        amenities.to_crs(3857).plot(ax=ax, alpha=0)
        result.to_crs(3857).plot(ax=ax, column=cluster_name, categorical = True, lw=.1, alpha=.6, markersize = result['weight%']*5)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=3)
        ax.set_title(cluster_name, fontsize=8)
        ax.axis('off');
        

    return result[['cluster', 'weight', 'weight%', 'geometry']], scores, amenities