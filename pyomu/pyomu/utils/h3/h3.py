import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon
import contextily as ctx
from pathlib import Path
import numpy as np

def geometry_to_h3(geometry, res=8):        
    '''
    Devuelve el hexágono h3 que contiene el punto (lat, lng) para una resolución dada. Llama a la función geo_to_h3 de la librería h3.
    
    Parámetros:    
    lat = latitud
    lng = longitud
    res = resolución esperada del hexágono h3
    
    Salida: Código H3 del hexágono al que pertenece la geometria
    '''
    return h3.geo_to_h3(lat=geometry.y, lng=geometry.x, resolution=res)

def h3_from_row(row, res, x, y):
    '''
    Toma una fila, un nivel de resolucion de h3 y los nombres que contienen las coordenadas xy y devuelve un id de indice h3
    
    Parámetros:
    row = Fila de un GeoDataFrame
    res = resolución h3
    x = coordenada x
    y = coordenada y
    
    Salida: Código H3 del hexágono al que pertenece la geometria
    '''
        
    ret = np.nan
    
    if (row[y] != np.nan)  & (row[x] != np.nan):
        ret = h3.geo_to_h3(row[y], row[x], resolution=res)
    
    return ret

def h3_indexing(df, res, lat='lat', lon='lon', var=None):
    """
    Esta funcion toma una tabla con dos pares de coordenadas para origen y destino. 
    Según n nivel de resolucion h3, devuelve la tabla con los ids de h3
    
    Parámetros:
    df = GeoDataFrame
    res = Resolución h3
    lat= campo latitud
    lon= campo longitud
    var= nombre de la variable salida (por defecto el nombre de variable es f'h3_res_{res}')
    
    Salida: GeoDataFrame con nuevo campo (con el nombre de la variable var)
    
    """
    
    if len(var)==0: var=f'h3_res_{res}' 
    
    df[var] = df.apply(h3_from_row, axis=1, args=[res, lon, lat])
    
    return df

def add_geometry(row, bring='polygon'):

    '''
    Devuelve una tupla de pares lat/lng que describen el polígono de la celda. Llama a la función h3_to_geo_boundary de la librería h3.
    
    Parámetros:     
    row = código del hexágono en formato h3    
    bring = define si devuelve un polígono, latitud o longitud
    
    Salida: geometry resultado
    '''
    points = h3.h3_to_geo_boundary(row, True)
    
    points = Polygon(points)
    if bring == 'lat':
        points = points.representative_point().y
    if bring == 'lon':
        points = points.representative_point().x
    
    return points

def bring_children_(x, res=8):    
    '''
    Trae los hijos de un hexágono h3
    
    Parámetros:
    x = código del hexágono en formato h3
    res = resolución de los hexágonos hijos
    
    Salida: lista con los códigos h3 solicitados
    '''
    return list(h3.h3_to_children(x, res))

def bring_children(gdf, res=8):    
    '''
    Dado un GeoDataFrame de hexágonos h3, la función devuelve un GeoDataFrame con los hexágonos hijos.
    
    Parámetros:
    gdf = GeoDataFrame de hexágonos h3 en una determinada resolución
    res = Resolución del nuevo GeoDataFrame con los hexágonos en resolución mayor. La resolución tiene que ser mayor a la resolución del GeoDataframe original.    
    
    Salida: nuevo GeoDataFrame con un registro por cada hexágono correpondiente a los h3 hijos.
    '''
    gdf_list = gdf.hex.apply(bring_children_, res=res)
    gdf_new = []
    for i in gdf_list.tolist(): gdf_new+=list(i)
    gdf_new = pd.DataFrame(gdf_new, columns=['hex'])
    gdf_new['geometry'] = gdf_new.hex.apply(add_geometry)
    gdf_new = gpd.GeoDataFrame(gdf_new, crs='EPSG:4326')
    return gdf_new

def create_h3(gdf, res = 8, show_map=False):
    '''
    Crea un GeoDataFrame de hexágonos h3 en un determinada resolución a partir de una GeoDataFrame de polígonos o puntos.
    
    Parámetros:
    gdf_ = GeoDataFrame de entrada
    res = Resolución h3 que se quiere convertir.
    show_map = (True/False) muestra mapa con el mapa original y el nuevo de hexágonos superpuestos
    
    Salida: Nuevo GeoDataFrame en formato de hexágonos H3 que cubre el mismo área geográfico que el GeoDataFrame de entrada.
    '''
    gdf_ = gdf.copy()
    gdf_['geometry'] = gdf_['geometry'].representative_point()
    gdf_['hex'] = gdf_.geometry.apply(geometry_to_h3, args=[res-3])
    gdf_ = gdf_[['hex']].drop_duplicates().reset_index(drop=True)
    gdf_['geometry'] = gdf_['hex'].apply(add_geometry)
    gdf_ = gpd.GeoDataFrame(gdf_, crs='EPSG:4326')

    gdf_ = bring_children(gdf_, res=res)
    
    gdf_ = gpd.sjoin(gdf_, gdf)[['hex', 'geometry']].drop_duplicates().reset_index(drop=True)
    
    if show_map:
        fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
        gdf.to_crs(3857).plot(ax=ax, alpha=.7, edgecolor='navy', color='None', lw=.1)
        gdf_.to_crs(3857).plot(ax=ax, alpha=.7, edgecolor='red', color='None', lw=.1)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=3)
        ax.axis('off')
    
    return gdf_

def create_latlon(df, normalize=False, res=8, hex='hex', lat='', lon='', var_latlon=''):
    
    delete_latlon = False
    
    if (len(lat)==0)|(len(lon)==0):
        delete_latlon = True
        lat = 'lat'
        lon = 'lon'
    
    cols = df.columns.tolist()
    new_cols = []

    
    df[lat] = df.representative_point().y.round(5)
    df[lon] = df.representative_point().x.round(5)
    
    df[hex] = df.apply(h3_from_row, axis=1, args=[res, lon, lat])
    
    new_cols += [hex, lat, lon]

    if normalize:    
        df[f'{lat}_norm'] = df[hex].apply(add_geometry, bring='lat').round(5)
        df[f'{lon}_norm'] = df[hex].apply(add_geometry, bring='lon').round(5)
        new_cols += [f'{lat}_norm', f'{lon}_norm']

    if len(var_latlon)>0:
        df[var_latlon] = df[lat].astype(str) + ', ' + df[lon].astype(str)
        new_cols += [var_latlon]
        if normalize:
            df[f'{var_latlon}_norm'] = df[f'{lat}_norm'].astype(str) + ', ' + df[f'{lon}_norm'].astype(str)
            new_cols += [f'{var_latlon}_norm']

    for i in new_cols:        
        if i not in cols:
            cols = cols + [i]

    df = df[cols]
    
    if delete_latlon:
        df = df.drop([lat, lon], axis=1)
        if normalize:
            df = df.drop([f'{lat}_norm', f'{lon}_norm'], axis=1)

    return df