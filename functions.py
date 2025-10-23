import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import folium

time_matrix = np.load('datos/time_matrix.npy')  # matriz de tiempos de viaje entre paradas
data = pd.read_csv('inputs.csv')
time_window_data = data[["time_window_start_utc","time_window_end_utc"]]


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

def traffic_func(t_hour, A_m, A_e, mu_m, mu_e, sigma):
    t = t_hour % 24  # tiempo de 0 a 24
    # modelado estocastico del trafico dependiendo de la hora
    peak_m = A_m * np.exp(-(((t-mu_m)**2))/(2*sigma**2))
    peak_e = A_e * np.exp(-(((t-mu_e)**2))/(2*sigma**2))
    return 1.0 + peak_m + peak_e
     

def time_traffic(i, j, t_depart_hour, time_matrix, traffic_params):
    # tiempo de viaje i a j con trafico
    base = float(time_matrix[i, j])
    # parametros de trafico
    mult = traffic_func(
        t_hour = t_depart_hour,
        A_m = traffic_params['A_m'],
        A_e = traffic_params['A_e'],
        mu_m = traffic_params['mu_m'],
        mu_e = traffic_params['mu_e'],
        sigma = traffic_params['sigma']
    )
    # tiempo calculado mas trafico
    return base * mult

# cambiar el formato en la ventana de tiempo del csv
def to_seconds_of_day(ts): 
    if pd.isna(ts):
        return np.nan
    return float((ts.hour * 3600) + (ts.minute * 60) + ts.second)

def load_time_windows(csv_path: str, n_nodes: int):
    # leer las ventanas y aplica la funcion para cambiar el formato
    df = pd.read_csv(
        csv_path,
        parse_dates=['time_window_start_utc', 'time_window_end_utc'],
    )

    starts = df['time_window_start_utc'].apply(to_seconds_of_day).to_numpy(dtype=float)
    ends   = df['time_window_end_utc'].apply(to_seconds_of_day).to_numpy(dtype=float)

    # ajusatr a n_nodes
    if len(starts) < n_nodes:
        pad = np.full((n_nodes - len(starts),), np.nan, dtype=float)
        starts = np.concatenate([starts, pad])
        ends   = np.concatenate([ends,   pad])
    else:
        starts = starts[:n_nodes]
        ends   = ends[:n_nodes]

    return starts, ends

def simulate_route(route, start_time_hour, service_time, time_matrix, traffic_params, tw_start_s=None, tw_end_s=None):
    
    # funcion para obtener el tiempo de servicio, en formato de dataframe o de valor unico
    # esto es por si queremos cambirar el tiempo de servicio por nodo
    # service_time = 300.0
    # O
    # service_time = np.array([0, 300, 300, 0, 0, ...])
    # por si queremos modificar lo tiempo de servicio por nodo
    def svc(j):
        return service_time[j] if hasattr(service_time, "__len__") else float(service_time)

    # listas para acumular resultados
    current_time = float(start_time_hour)    # horas decimales
    total_drive = 0.0
    total_service = 0.0
    total_wait = 0.0
    missed_nodes = []

    use_tw = (tw_start_s is not None) and (tw_end_s is not None)

    # esto es lo que devuelve cuando se evalua el chromosoma
    # se ve asi con el ejemplo de ruta [0, 1, 2, 0]
    # {'drive': np.float64(648.0753908993297), 'wait': 0.0, 'service': 600.0, 'missed': 0, 
    # 'missed_nodes': [], 'end_time_hour': np.float64(16.34668760858315)}
    # Total de tiempo para el cromosoma = 0:20:48.075391
    # en eso se traduce, esta en segundos, y se calcula la hora final, sin segundos
    if len(route) < 2:
        return {"drive": 0.0, "wait": 0.0, "service": 0.0,
                "missed": 0, "missed_nodes": [], "end_time_hour": current_time % 24.0}

    for idx in range(len(route) - 1):
        i = route[idx]
        j = route[idx + 1]
        t_depart = current_time

        # funcion trafico
        t_travel_s = time_traffic(i, j, t_depart, time_matrix, traffic_params)
        # llegada en segundos
        t_curr_s = current_time * 3600.0
        t_arrival_s = t_curr_s + t_travel_s

        # ignorar el servicio en el nodo 0
        if j == 0:
            t_depart_j_s = t_arrival_s  # regresar al depósito, sin servicio, sin espera
            total_drive += t_travel_s
            current_time = (t_depart_j_s / 3600.0) % 24.0
            continue

        if use_tw and np.isfinite(tw_start_s[j]) and np.isfinite(tw_end_s[j]):
            arr_mod = t_arrival_s % 86400.0
            start_j = float(tw_start_s[j])
            end_j   = float(tw_end_s[j])

            # manejar ventana que cruza medianoche
            if end_j <= start_j:
                end_cmp = end_j + 86400.0
                arr_cmp = arr_mod + 86400.0 if arr_mod < start_j else arr_mod
            else:
                end_cmp = end_j
                arr_cmp = arr_mod

            if arr_cmp < start_j:
                # espera hasta apertura
                wait = start_j - arr_cmp
                t_start_service_s = t_arrival_s + wait
                t_depart_j_s = t_start_service_s + svc(j)
                total_wait += wait
                total_service += svc(j) 
            elif arr_cmp > end_cmp:
                # tarde, no se entrega (sin servicio)
                missed_nodes.append(j)
                t_depart_j_s = t_arrival_s
            else:
                # dentro de ventana
                t_depart_j_s = t_arrival_s + svc(j)
                total_service += svc(j) 
        else:
            # sin ventana, servir al llegar
            t_depart_j_s = t_arrival_s + svc(j)
            total_service += svc(j)     

        total_drive += t_travel_s
        current_time = (t_depart_j_s / 3600.0) % 24.0

    return {
        "drive": total_drive,
        "wait": total_wait,
        "service": total_service,
        "missed": len(missed_nodes),
        "missed_nodes": missed_nodes,
        "end_time_hour": current_time
    }

def simulate_chromosome(chromosome, start_time, service_time, time_matrix, traffic_params):
    results = []
    for k, route in enumerate(chromosome):
        m = simulate_route(route, float(start_time[k]), service_time, time_matrix, traffic_params)
        results.append(m)
    return results


def objective_function(chromosome, start_time, service_time, time_matrix, traffic_params,
                       distance_matrix, g_k=1.0, M=1e6):
    """
    Z = sum(g_k * dist_tot_k) + M * unf
    Retorna el costo total (a minimizar)
    """
    results = simulate_chromosome(chromosome, start_time, service_time, time_matrix, traffic_params)
    total_distance = 0.0
    unf = 0  # entregas incumplidas, o restricciones violadas
    
    for route in chromosome:
        # calcular distancia total de cada ruta
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            total_distance += distance_matrix[i, j]
    
    # acumular restricciones incumplidas (si existen)
    unf += sum(r.get("missed", 0) for r in results)
    
    Z = g_k * total_distance + M * unf
    return Z



def fitness(chromosome, start_time, service_time, time_matrix, traffic_params,
            distance_matrix, g_k=1.0, M=1e6):
    """
    fitness = 1 / Z
    cuanto menor sea el costo, mayor el fitness.
    """
    Z = objective_function(chromosome, start_time, service_time, time_matrix,
                           traffic_params, distance_matrix, g_k, M)
    return 1.0 / (Z + 1e-9)

def mutation(chromosome, prob):
    #Aplica una mutación a un cromosoma dado, con cierta probabilidad. 
    #Intercambia dos nodos dentro de una ruta aleatoria, dentro de las rutas de un cromosoma
    mutacion = chromosome
    if random.random() < prob:
        #Elegir una ruta aleatoria 
        route_idx = random.randint(0, len(mutacion) - 1)
        route = mutacion[route_idx]

        #Solo elegir rutas con longitud mayor a 3
        if len(route) > 3: 
            #Elegir dos índices al azar
            rand_i, rand_j = random.sample(range(1, len(route) - 1), 2)

            #Intercambiar los nodos 
            route[rand_i], route[rand_j] = route[rand_j], route[rand_i]

            #Reemplazar la ruta
            mutacion[route_idx] = route
    #Regresa la ruta modificada
    return mutacion

def crossover(chromo1, chromo2, prob):
    import copy
    #Aplica el cruce de comosomas entre dos de ellos, donde 
    #se intercambian rutas completas entre dos cromosomas
    hijo1 = copy.deepcopy(chromo1)
    hijo2 = copy.deepcopy(chromo2)


    if random.random() < prob:
        route_idx = random.randint(0, len(chromo1) - 1) #Selecciona aleatoriamente indice de la ruta a swappear
        temp = hijo1[route_idx][1:-1] #Guardar temporalmente ruta del hijo1
        hijo1[route_idx][1:-1] = hijo2[route_idx][1:-1] #Swappear nodod de hijo1 por nodos de hijo2
        hijo2[route_idx][1:-1] = temp #Sustituir nodos de ruta2 por nodos de hijo1
    
    return hijo1, hijo2

def check_capacity(route, demands, vehicle_capacity):
    #Recibe la lista de nodos de una ruta, un array de 
    #demanda de cada nodo, y la capacidad máxima del vehículo
    total_demand = 0
    for node in route: 
        if node != 0: 
            total_demand += demands[node]
    return total_demand <= vehicle_capacity

def check_work_time(route, start_time_hour, service_time, time_matrix, traffic_params, max_time):
    #Revisar si el tiempo total de trabajo de un vehículo
    #no excede la jornada máxima permitida

    def svc(j):
        #Si service_time tiene atributo length
        if hasattr(service_time, "__len__"):
            #En caso de ser lista, retorna el service_time para el nodo j
            return float(service_time[j])
        else:
            #Retorna el service_time, si service_time es un solo valor (no una lista)
            return float(service_time)
    
    current_time = float(start_time_hour)
    for idx in range(len(route) - 1):
        i = route[idx]
        j = route[idx + 1]

        #Calcular el tiempo de viaje con tráfico
        travel_time_s = time_traffic(i, j, current_time, time_matrix, traffic_params)
        travel_time_h = travel_time_s / 3600
        current_time += travel_time_h

        #Usado para ignorar el servicio en depósito
        if j != 0: 
            current_time += svc(j) / 3600 
    
    #Si el tiempo total excede el máximo permitido, regresa True
    if current_time - start_time_hour <= max_time:
        return True
    else:
        return False

#Al parecer, esta función ya la implementa implicitamente la función objetivo
def check_time_windows(route, arrival_times, tw_start, tw_end):
    #Regresa la lista de nodos que no cumplen con las restricciones de tiempo o ventanas de tiempo
    missed_nodes = []
    for node, arrival in zip(route, arrival_times):
        if node == 0:
            continue
        start, end = tw_start[node], tw_end[node]
        if arrival < start or arrival > end:
            missed_nodes.append(node)
    return missed_nodes

def genetic_algorithm(clusters, population_size = 50, num_generations = 200, elitism_rate = 0.2, mutation_prob = 0.05, crossover_prob = 0.9, start_time = None, service_time = 300, 
                      time_matrix = None, distance_matrix = None, traffic_params = None, demands = None, vehicle_capacity = 50000, max_work_time = 8.0):
    population = init_population(population_size, clusters)

    num_elite = int(elitism_rate * population_size)

    if start_time is None: 
        start_time = []
        for _ in range(len(clusters)):
            start_time.append(8)

    #Loop principal 
    for gen in range(num_generations):
        fitness_values = []
        for chromo in population:
            fit = fitness(chromosome = chromo, 
                          start_time = start_time,
                          service_time= service_time, 
                          time_matrix = time_matrix, 
                          traffic_params=traffic_params, 
                          distance_matrix = distance_matrix)
            fitness_values.append((fit, chromo))
        
        fitness_values.sort(key = lambda x: x[0], reverse = True)
        ranked_population = []
        for pair in fitness_values: 
            fitness_val, chromo = pair
            ranked_population.append(chromo)
        
        #Elitism 
        new_population = ranked_population[:num_elite]

        #Generar nuevos individuos
        while len(new_population) < population_size:
            father_1 = random.choice(ranked_population[:num_elite])
            father_2 = random.choice(ranked_population[:num_elite])

            #Crossover
            son_1, son_2 = crossover(father_1, father_1, prob = crossover_prob)

            #Mutation
            son_1= mutation(son_1, prob = mutation_prob)
            son_2 = mutation(son_2, prob = mutation_prob)

            new_population.append(son_1)
            if len(new_population) < population_size:
                new_population.append(son_2)
        population = new_population
    
    #Resultados finales
    final_fitness = []
    for chromo in population:
        fit_value = fitness(chromo, start_time, service_time, time_matrix, traffic_params, distance_matrix)
        final_fitness.append((fit_value, chromo))
    
    best_sol = final_fitness[0][1]
    best_fit_val = final_fitness[0][0]

    return best_sol, best_fit_val

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
    # num_vehiculos = 3
    # clusters_result = cluster_stops(k = num_vehiculos)
    
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

    # pruebas para simular rutas con trafico
    """
    tw_start_s, tw_end_s = load_time_windows("inputs.csv", n_nodes=time_matrix.shape[0])

    traffic_params = dict(A_m=0.5, A_e=0.5, mu_m=8.0, mu_e=17.5, sigma=1.25)
    route = [0, 1, 2, 0]
    start_time_hour = 16.0 
    service_time = 300.0 # esto asume 5 min de espera por cada parada(servicio type shi)    

    res = simulate_route(route, start_time_hour, service_time,
                     time_matrix, traffic_params,
                     tw_start_s=tw_start_s, tw_end_s=tw_end_s)

    print(res)
    tot = res["drive"] + res["service"] + res["wait"]
    print("Total de tiempo para el cromosoma =", __import__("datetime").timedelta(seconds=tot))

    """

    #Pruebas para mutación
    '''
    chromosome = [
        [0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0],
        [0, 7, 8, 9, 0]
    ]
    mut = mutation(chromosome, prob = 0.9)
    print(mut)
    '''

    """
    #Pruebas para crossover 
    chromosome_1 = [
    [0, 1, 2, 3, 0],
    [0, 4, 5, 6, 0],
    [0, 7, 8, 9, 0]]

    chromosome_2 = [
    [0, 2, 4, 8, 0], 
    [0, 5, 9, 3, 0], 
    [0, 4, 6, 7, 0]]

    for _ in range(5):
        print(crossover(chromosome_1, chromosome_2, 0.9))
    """

