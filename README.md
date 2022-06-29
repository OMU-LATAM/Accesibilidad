# Welcome to PyOMU

This library presents a set of tools to perform urban accessibility analysis

### Antecedentes

En el marco de la actualización del Observatorio de Movilidad Urbana (https://omu-latam.org/), se está trabajando en torno a métricas de acceso universal tomando provecho de los avances tecnológicos en el análisis de datos. Esta librería presenta las herramientas utilizadas en el modelo análitico utilizado para analizar la accesibilidad urbana en ciudades de América Latina. Se busca que este modelo sea aplicable a diferentes contextos y replicable por personas con conocimientos técnicos en datos y programación. Los indicadores y mapas resultantes podrán ser una herramienta útil para identificar brechas de acceso en una ciudad y diseñar las correspondientes intervenciones.

Las herramientas desarrolladas para la construcción de un modelo analítico que permite analizar y elaborar indicadores de accesibilidad utiliza fuentes de datos de acceso público con información socioeconómica, cartografía e información de oferta y demanda de viajes. Este modelo se aplica a ciudades de América Latina con el principal objetivo de desarrollar indicadores para ciudades de toda la región que sirvan para comprender mejor la accesibilidad en las distintas ciudades, brindando un mejor entendimiento del sistema urbano y de la movilidad para analistas y tomadores de decisión. Los resultados forman parte de la tercera edición del Observatorio de Movilidad Urbana (OMU), desarrollado por CAF, Banco de Desarrollo de América Latina en conjunto con  BID, Banco Interamericano de Desarrollo. Los indicadores elaborados en este documento: accesibilidad, cobertura del transporte público e índice de congestión; se enmarcan dentro de una matriz de 23 indicadores, bajo 4 pilares fundamentales: acceso universal, eficiencia y calidad, seguridad y movilidad verde, cuyo objetivo es la evaluación y monitoreo de métricas relevantes para la movilidad urbana de las ciudades de la región.

## Para la instalación de esta librería se puede utilizar:

```sh
pip install pyomu
```

## Para la utilización de esta librería:

```sh
from pyomu install pyomu
```

### Datos de entrada

Como dato de entrada principal se requiere la cartografía censal en el menor nivel de desagregación posible (ej. radios, secciones o manzanas censales), la población en cada polígono y una serie de variables censales seleccionadas que serán utilizadas para la construcción de un indicador de nivel socioeconómico. Se recomienda incluir variables relacionadas con la calidad de los materiales de la vivienda, calidad o acceso a servicios (agua, cloacas, electricidad, etc), acceso a bienes, nivel educativo, etc. 

### Principales funciones

#### Nivel socioeconómico

calculate_nse: cálculo de nivel socioeconómico en base a datos censales

#### Utilización cartografía de hexágonos (en base a librería h3)

create_h3: en base a la cartografía censal crea una nueva capa de hexágonos

distribute_population: distribuye la población censal y nse hacia los hexágonos 

#### Identifica áreas de alta densidad de actividad (como proxy de atractores de viaje)

bring_osm: trae equipamientos de OpenStreetMaps
assign_weights: asigna ponderadores a los equipamientos
activity_density: crea cluster de actividad para utilizar como proxy de atractores de viajesss

#### Cálculo distancias y tiempos de viaje para una matriz de origenes y destinos

measure_distances_osm: genera una matriz de origenes y destinos y calcula las distancias en la red vial de Open Street Maps
trips_gmaps_from_od: genera una matriz de origenes y destinos y calcula los tiempos distancias en la red vial de Google Maps para distintos modos de transporte
trips_gmaps_from_matrix: en base a una matrix de origenes y destinos existente calcula los tiempos distancias en la red vial de Google Maps para distintos modos de transporte
distances_to_equipments: identifica equipamientos más cercanos y calcula cantidad de establecimientos en un rango de distancia y calcula los tiempos de viaje en transporte público al equipamiento más cercanos (ej. escuela más cercana desde cada radio censal)
calculate_green_space: cálcula m2 de espacios verdes en un rango de distancia y m2 de espacios verdes per-cápita en un rango de distancia


# Ejemplos

```python
from pyomu import pyomu

censo = pyomu.calculate_nse(censo, 
                            X, 
                            population='cant_pers', 
                            show_map=False)
    
hexs = pyomu.create_h3(censo, 
                       res=8, 
                       show_map=False)

hexs = pyomu.distribute_population(gdf=censo, 
                                   id_gdf='RADIO_LINK', 
                                   hexs=hexs, 
                                   id_hexs='hex', 
                                   population='cant_pers', 
                                   pca='PCA_1', 
                                   crs=city_crs, 
                                   q=[5, 3],
                                   order_nse = [['Alto', 'Medio-Alto', 'Medio', 'Medio-Bajo', 'Bajo'],
                                                ['Alto', 'Medio', 'Bajo']],
                                   show_map=True)

```
censo es un DataFrame con la cartografía y variables censales.
X contiene las variables que serán utilizadas para calcular el Nivel Socioeconómico




