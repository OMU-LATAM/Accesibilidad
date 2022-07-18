import pandas as pd
import numpy as np
import geopandas as gpd
from pandas._libs.lib import is_integer
import unidecode
import osmnx as ox
import networkx as nx
import pandana
from pandana.loaders import osm as osm_pandana

from pathlib import Path

import datetime as datetime
from datetime import date
from datetime import timedelta

import pytz
from tzwhere import tzwhere 
from pytz import timezone


def weighted_qcut(values, weights, q, **kwargs):
    '''
    Calculo terciles, deciles, terciles u otra corto teniendo en cuenta una variable de ponderación
    Parámetros
    values = lista de valores
    weights = lista con la ponderación. 
    q = Cantidad de grupos resultado
    **kwargs = parámetros para la función pd.cut 
    
    Salida: lista con un resultado numérico correspondiente a la agrupación solicitada. Ej. Para quintiles devuelve valores [1,2,3,4,5]
    
    '''
    
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()


def normalize_words(x):
    return unidecode.unidecode(x).lower()



def correct_datetime(trip_datetime, lat, lon):        
    '''
    Corrige la diferencia horaria para la consulta en Google Maps teniendo en cuenta la zona de consulta y la zona donde se está corriendo el proceso
    
    Parámetros
    trip_datetime = datetime en el que se quiere correr el proceso
    lat = latitud donde se va a correr el proceso
    lon =  longitud donde se va a correr el proceso
    
    Salida: devuelve una variable datatime corregida para correr el proceso según la diferencia horaria
    '''
    try:
        tz = tzwhere.tzwhere()
        timeZoneStr = tz.tzNameAt(lat, lon)
        timeZoneObj = timezone(timeZoneStr)
        now_time = datetime.datetime.now(timeZoneObj)
        now_time = now_time.time().hour    
        hour_dif = now_time - datetime.datetime.now().time().hour     
        hour_dif = trip_datetime + timedelta(hours=(hour_dif*-1))
    except:
        hour_dif = trip_datetime
    return hour_dif

def reindex_df(df, weight_col, div=0):
    """expand the dataframe to prepare for resampling
    result is 1 row per count per sample"""
    
    if div > 0:
        df[weight_col] = df[weight_col] / div
        
    df = df.reindex(df.index.repeat(df[weight_col]))
    df.reset_index(drop=True, inplace=True)
    return(df)