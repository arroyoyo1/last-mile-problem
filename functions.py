import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import random


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
    
    # etiquetas de cluster para cada parada
    labels = kmeans.labels_
    
    # para organizarlos, creamos una lista de listas para los clusters
    clusters = [[] for vehiculo in range(k)] # si k = 3, clusters = [[], [], []]
    for indice, stop_id in enumerate(df['stop_id']): # enumerate hace (0, 'AD')
        num_cluster = labels[indice]
        mapped_id = stop2int.get(stop_id)
        if mapped_id is not None:
            if mapped_id != 0:
                clusters[num_cluster].append(mapped_id)

    return clusters


def generate_solutions(clusters):
    chromosome = []
    for cluster in clusters:
        # hacemos diferentes permutaciones con el resultado de cada cluster 
        perm = random.sample(cluster, len(cluster))
        # print(perm)
        route = [0] + perm + [0] # ruta completa
        chromosome.append(route) 
        # print(individual)
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
    pop_size = 50
    population = init_population(pop_size, clusters_result)
    
    """
    print(f"{len(population[0])} rutas")
    # primer cromosoma (population[0])
    for i, route in enumerate(population[0]): # (0, [0, 5, 1, 8, 0])
        print(f"ruta {i+1}, longitud {len(route)}, {route}")
    """