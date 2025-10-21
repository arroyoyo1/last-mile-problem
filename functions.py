import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import folium



def mapping():
    stop_ids = np.load('datos/stop_ids.npy') # cargamos el array de stop_ids

    # aquí empieza el mapeo de string a int (para que el algoritmo funcione)
    stop2int = {stop_id: i for i, stop_id in enumerate(stop_ids)}
    # y este es el mapeo de int a string (este no era mil necesario pero chance estaría bien para mostrar la ruta final)
    int2stop = {i: stop_id for i, stop_id in enumerate(stop_ids)}

    # guardamos los mapeos en archivos json para no tener que hacerlos cada que corramos el algoritmo
    path_map1 = 'datos/stop2int.json'
    path_map2 = 'datos/int2stop.json'

    with open(path_map1, 'w') as f:
        json.dump(stop2int, f, indent=4)
        
    with open(path_map2, 'w') as f:
        json.dump(int2stop, f, indent=4)
        
    return stop2int, int2stop


def cluster_stops(k):
    df = pd.read_csv('inputs.csv') # cargamos los datos a un df de pandas y los mapeos
    with open('datos/stop2int.json', 'r') as f:
        stop2int = json.load(f) # load lo convieret automáticamente a un diccionario
    
    # coordenadas de las paradas
    coords = df[['stop_lat', 'stop_lng']].values # values lo convierte a un array de np    
    # weights para el algoritmo (volumen de los paquetes)
    sample_weights = df['package_volume_cm3'].values
    
    # llevamos a cabo la clusterización
    kmeans = KMeans(n_clusters = k, random_state = 6174, n_init = 10)
    kmeans.fit(coords, sample_weight=sample_weights)
    labels = kmeans.labels_ # etiquetas de cluster para cada parada

    # para organizarlos, creamos una lista de listas para los clusters
    clusters = [[] for vehiculo in range(k)] # si k = 3, clusters = [[], [], []]
    for indice, stop_id in enumerate(df['stop_id']): # enumerate hace (0, 'AD')
        num_cluster = labels[indice]
        mapped_id = stop2int.get(stop_id)
        if mapped_id is not None:
            if mapped_id != 0:
                clusters[num_cluster].append(mapped_id)

    # scatterplot clusters
    # añadimos las etiquetas de cluster al df para poder graficar
    df['cluster'] = labels
    
    # creamos las categorías de volumen de paquetes
    df['volume_category'] = pd.cut(df['package_volume_cm3'], 
    bins = [0, 10000, 20000, df['package_volume_cm3'].max() + 1],
    labels = ['<10k', '10k-20k', '>20k'], right = False)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x = 'stop_lng', y = 'stop_lat', hue = 'cluster',         
        style = 'volume_category', palette = 'flare', s = 100, alpha = 0.7              
    )
    plt.title('Puntos de Entrega Clusterizados', fontsize=16)
    plt.xlabel('longitud')
    plt.ylabel('latitud')
    plt.legend(title='clusters y volumen')
    plt.grid(True)
    plt.show()

    # mapa clusters
    depot_coords = [34.116928, -118.250428] # para centrar el mapa
    delivery_map = folium.Map(location = depot_coords, tiles = "cartodb positron", zoom_start = 65) # mapa base

    # añadimos un marcador para el depósito
    folium.Marker(
        depot_coords, popup = 'UCLA5',
        icon = folium.Icon(color = 'red', icon = 'industry', prefix = 'fa')
    ).add_to(delivery_map)

    # lista de colores para los clusters
    num_clusters = df['cluster'].nunique()
    colors = sns.color_palette('flare', num_clusters).as_hex()
    
    # añadimos cada parada al mapa
    for i, row in df.iterrows():
        folium.CircleMarker(
            location = [row['stop_lat'], row['stop_lng']], radius=5,
            color=colors[row['cluster']], fill=True, fill_color=colors[row['cluster']],
            fill_opacity=0.7, popup = f"ID: {row['stop_id']}<br>Cluster: {row['cluster']}<br>Volumen: {row['package_volume_cm3']:.2f} cm³"
        ).add_to(delivery_map)
        
    # guardamos el mapa en un html
    map_file = './datos/delivery_map.html'
    delivery_map.save(map_file)

    return clusters




def generate_solutions(clusters):
    chromosome = []
    for cluster in clusters:
        # hacemos diferentes permutaciones con el resultado de cada cluster 
        perm = random.sample(cluster, len(cluster))
        route = [0] + perm + [0] # ruta completa
        chromosome.append(route) 

    return chromosome


def init_population(population_size, clusters):
    population = []
    for chromo in range(population_size):
        chromosome = generate_solutions(clusters)
        population.append(chromosome)

    return population

if __name__ == '__main__':

    # pruebas mapping()
    # stop2int, int2stop = mapping()

    """
    para verificar mapping()
    print(f"UCLA5 - {stop2int.get('UCLA5')}")
    print(f"0 - {int2stop.get(0)}")
    print(f"'AD' - {stop2int.get('AD')}")
    print(f"10 - '{stop2int .get('10')}'") 
    """
    
    # pruebas cluster_stops()
    num_vehiculos = 3
    clusters_result = cluster_stops(k = num_vehiculos)
    
    """
    # para verificar la clusterización
    for i, cluster in enumerate(clusters_result): # (0, [1, 5, 8]) donde i=0 y cluster=lista
        print(f"\n cluster {i+1}, {len(cluster)} paradas: {cluster}")
    """

    # pruebas para inicializar soluciones
    # pop_size = 50
    # population = init_population(pop_size, clusters_result)
    
    """
    print(f"{len(population[0])} rutas")
    # primer cromosoma (population[0])
    for i, route in enumerate(population[0]): # (0, [0, 5, 1, 8, 0])
        print(f"ruta {i+1}, longitud {len(route)}, {route}")
    """