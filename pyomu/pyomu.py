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
from pyomu.access import access
from pyomu.vizuals import vizuals
from pyomu.nse import nse   

    
    
def calculate_nse_in_hexagons(censo,
                              id_censo = '',                          
                              population='',
                              vars_nse = '', 
                              city_crs = '',
                              current_path = Path(),
                              city='',
                              res=8,
                              run_always=True,
                              show_map=True):
    
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    
    utils.create_result_dir(current_path=current_path)
    
    if (not Path(current_path / 'Resultados_files' / f'{city}_hexs{res}.geojson').is_file())|(run_always):


        censo = nse.calculate_nse(censo, 
                              vars_nse, 
                              population=population, 
                              show_map=False)

        hexs = utils.h3.create_h3(censo, 
                         res=res, 
                         show_map=False)

        hexs = nse.distribute_population(gdf=censo, 
                                     id_gdf=id_censo, 
                                     hexs=hexs, 
                                     id_hexs='hex', 
                                     population=population, 
                                     pca='PCA_1', 
                                     crs=city_crs, 
                                     q=[5, 3],
                                     order_nse = [['Alto', 'Medio-Alto', 'Medio', 'Medio-Bajo', 'Bajo'],
                                                  ['Alto', 'Medio', 'Bajo']],
                                     show_map=show_map)

        hexs.to_file(current_path / 'Resultados_files' / f'{city}_hexs{res}.geojson')
        print('')
        print(f'Se guardó el archivo {city}_hexs{res}.geojson en', current_path / f'{city}_hexs{res}.geojson')
        print('')
    else:
        hexs = gpd.read_file(current_path / 'Resultados_files' / f'{city}_hexs{res}.geojson')
        
    hexs.loc[hexs[population].isna(), population] = 0

    return hexs.reset_index(drop=True)
    
    
def calculate_activity_density(hexs,
                               tags = {'amenity':True},
                               cantidad_clusters = 8,
                               city_crs = '',
                               current_path = Path(),
                               city='',                              
                               run_always=True):
    
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    
    utils.create_result_dir(current_path=current_path)
    
    if (not Path(current_path / 'Resultados_files' / f'{city}_activity_density.geojson').is_file())|(run_always):


        amenities = utils.bring_osm(hexs, tags = tags)
        amenities = nse.assign_weights(amenities)
        densidad_actividad_result, scores, amenities2 = nse.activity_density(amenities, 
                                                                  city_crs, 
                                                                  cantidad_clusters = cantidad_clusters,                                                               
                                                                  show_map = True)

        densidad_actividad_result.to_file(current_path / 'Resultados_files' / f'{city}_activity_density.geojson')
        print('')
        print('Se guardó el archivo hexs.geojson en', current_path / f'{city}_activity_density.geojson')
        print('')
    else:
        densidad_actividad_result = gpd.read_file(current_path / 'Resultados_files' / f'{city}_activity_density.geojson')

    return densidad_actividad_result
    
    
def calculate_od_matrix_all_day(origin, 
                                id_origin, 
                                destination, 
                                id_destination,                                 
                                trip_datetime, 
                                population,
                                key, 
                                samples_origin = 12,
                                samples_destination = 6,
                                transit=False,
                                driving=True,
                                walking=False,
                                bicycling=False,
                                current_path=Path(), 
                                city = '',
                                normalize=False,
                                run_always=True):
    
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    
    utils.create_result_dir(current_path=current_path)
    


    if (not (current_path / 'tmp' / f'{city}_origin_sample.geojson').is_file())|(run_always):    
        origin_sample = origin.sample(samples_origin, random_state=2).copy()
        origin_sample.to_file(current_path / 'tmp' / f'{city}_origin_sample.geojson')
        destination_sample = destination.sample(samples_destination, random_state=2).copy()
        destination_sample.to_file(current_path / 'tmp' / f'{city}_destination_sample.geojson')
    else:    
        origin_sample =  gpd.read_file(current_path / 'tmp' / f'{city}_origin_sample.geojson')
        destination_sample =  gpd.read_file(current_path / 'tmp' / f'{city}_destination_sample.geojson')


    od_matrix_all_day = access.trips_gmaps_from_od(origin = origin_sample, 
                                            id_origin = id_origin, 
                                            destination = destination_sample, 
                                            id_destination = id_destination, 
                                            population = population,
                                            trip_datetime = trip_datetime,                 
                                            key = key, 
                                            transit=transit,
                                            driving=driving,
                                            walking=walking,
                                            bicycling=bicycling,
                                            full_day=True,
                                            normalize=normalize,
                                            only_distance_duration=True,
                                            current_path=current_path,
                                            city=city)


    if len(od_matrix_all_day) > 0:
        od_matrix_all_day.to_csv(current_path / 'Resultados_files' / f'{city}_od_matrix_all_day.csv', index=False)
        print('')
        print('Se guardó el archivo od_matrix_all_day.geojson en', current_path / f'{city}_od_matrix_all_day.csv')
        print('')
    
    if len(od_matrix_all_day) > 0:
        indicators_all = vizuals.indicators_all_day(od_matrix_all_day, current_path, city=city)
    
    return od_matrix_all_day
    
    











