import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import random
import seaborn as sns
import folium
# import matplotlib
# matplotlib.use("Agg")  # backend sin GUI, evita Tk
import matplotlib.pyplot as plt
import datetime
import os
from matrices import matrices

cost_km = 7.5

def load_all_data():
    data = pd.read_csv('inputs.csv')
    distance_matrix = np.load('datos/distance_matrix.npy')
    time_matrix = np.load('datos/time_matrix.npy')
    
    with open('datos/stop2int.json', 'r') as f:
        stop2int = json.load(f)
    with open('datos/int2stop.json', 'r') as f:
        int2stop = json.load(f)
        
    return data, distance_matrix, time_matrix, stop2int, int2stop

def load_lookups(data, stop2int):
    # mapeamos stop_id a todos los datos asociados relevantes
    data_lookup = data.set_index('stop_id')
    
    service_times = {}
    volumes = {}
    tw_starts = {}
    tw_ends = {}
    
    depot_start_time = data['departure_time_utc'].iloc[0]
    h, m, s = map(int, depot_start_time.split(':'))
    depot_start_seconds = h * 3600 + m * 60 + s # convertimos datetime a segundos la hora de inicio de la ruta
    
    for stop_str, stop_int in stop2int.items():
        if stop_int == 0: # depósito
            service_times[0] = 0
            volumes[0] = 0
            tw_starts[0] = None
            tw_ends[0] = None
        else:
            try:
                stop_data = data_lookup.loc[stop_str]
                # maneja casos donde un stop_id aparezca más de una vez (ahorita no pasa en inputs pero en el
                # data set original si pasa) entonces estaría bien hacer el condicional para manejar el dataset
                if isinstance(stop_data, pd.DataFrame):
                    stop_data = stop_data.iloc[0]
                    
                service_times[stop_int] = stop_data['planned_service_time_seconds']
                volumes[stop_int] = stop_data['package_volume_cm3']
                tw_start_str = stop_data['time_window_start_utc']
                tw_end_str = stop_data['time_window_end_utc']
                
                # verifica si tiene ventanas o no
                if pd.isna(tw_start_str) or pd.isna(tw_end_str):
                    tw_starts[stop_int] = None
                    tw_ends[stop_int] = None
                else:
                    # convertimos datetime a segundos pero ahora para las tw
                    tw_starts[stop_int] = (datetime.datetime.fromisoformat(tw_start_str).hour * 3600 +
                                           datetime.datetime.fromisoformat(tw_start_str).minute * 60 +
                                           datetime.datetime.fromisoformat(tw_start_str).second)
                    tw_ends[stop_int] = (datetime.datetime.fromisoformat(tw_end_str).hour * 3600 +
                                         datetime.datetime.fromisoformat(tw_end_str).minute * 60 +
                                         datetime.datetime.fromisoformat(tw_end_str).second)
            except KeyError:
                # This might happen if stop_ids.npy is out of sync with inputs.csv omg!
                print(f"Warning: stop_id '{stop_str}' (int: {stop_int}) not found in inputs.csv. Using defaults.")
                service_times[stop_int] = 0
                volumes[stop_int] = 0
                tw_starts[stop_int] = None
                tw_ends[stop_int] = None
                
    return service_times, volumes, tw_starts, tw_ends, depot_start_seconds

"""time_matrix = np.load('datos/time_matrix.npy')  # matriz de tiempos de viaje entre paradas
distance_matrix = np.load('datos/distance_matrix.npy')  # matriz de distancias entre paradas
mean_time = data['planned_service_time_seconds'].mean()  # tiempo promedio de servicio por parada"""

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


def cluster_stops(k, data, stop2int):
    df = data.drop_duplicates(subset='stop_id').reset_index(drop=True)
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
        data = df, x = 'stop_lng', y = 'stop_lat', hue = 'cluster',         
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

    return clusters, df


def initialize_population(population_size, clusters):
    population = []
    for chromo in range(population_size):
        chromosome = []
        for cluster in clusters:
            # hacemos diferentes permutaciones con el resultado de cada cluster 
            perm = random.sample(cluster, len(cluster))
            route = [0] + perm + [0] # ruta completa
            chromosome.append(route)
        population.append(chromosome) # lista de listas

    return population


def traffic(t_hour, A_m = 0.5, A_e = 0.5, mu_m = 8.0, mu_e = 17.0, sigma = 1.25):
    t = t_hour % 24  # tiempo de 0 a 24
    # modelado estocastico del trafico dependiendo de la hora
    peak_m = A_m * np.exp(-(((t-mu_m)**2))/(2*sigma**2))
    peak_e = A_e * np.exp(-(((t-mu_e)**2))/(2*sigma**2))

    return 1.0 + peak_m + peak_e
     

def time_traffic(i, j, t_depart_hour, time_matrix, traffic_params):
    # tiempo de viaje i a j con trafico
    base = float(time_matrix[i, j])
    # parmetros de trafico
    mult = traffic(
        t_hour = t_depart_hour,
        A_m = traffic_params['A_m'],
        A_e = traffic_params['A_e'],
        mu_m = traffic_params['mu_m'],
        mu_e = traffic_params['mu_e'],
        sigma = traffic_params['sigma']
    )
    # tiempo calculado mas trafico
    return base * mult


def simulate_route(route, start_time_s, time_matrix, traffic_params, service_times_lookup, tw_starts_lookup, tw_ends_lookup):
    total_drive_s = 0
    total_service_s = 0
    total_wait_s = 0
    missed_nodes = []
    current_time_s = start_time_s
    
    for i in range(len(route) - 1):
        node_i = route[i]
        node_j = route[i+1]
        
        # calculamos el tiempo de viaje con tráfico y actualizamos el tiempo de llegada al nodo j
        t_depart_hour = (current_time_s / 3600.0) % 24.0
        t_travel_s = time_traffic(node_i, node_j, t_depart_hour, time_matrix, traffic_params)        
        t_arrival_s = current_time_s + t_travel_s
        
        # caso especial para el último nodo
        if node_j == 0:
            t_depart_j_s = t_arrival_s 
            total_drive_s += t_travel_s
            current_time_s = t_depart_j_s
            continue
            
        # datos asociados al nodo j
        service_time_j = service_times_lookup[node_j]
        tw_start_j = tw_starts_lookup[node_j]
        tw_end_j = tw_ends_lookup[node_j]
        
        use_tw = (tw_start_j is not None) and (tw_end_j is not None) # solo para verificar si se usan tw

        t_start_service_s = t_arrival_s
        wait_j = 0
        
        if use_tw:
            # creamos una variable comparable para el fin de la ventana
            tw_end_twin = tw_end_j
            # si el fin de la tw es antes o igual que el inicio, asume que es al día siguiente
            if tw_end_j <= tw_start_j:
                tw_end_twin += 86400  # añadir 24 horas en segundos

            # verificamos si hay tiempo de espera
            if t_arrival_s < tw_start_j:
                wait_j = tw_start_j - t_arrival_s
                t_start_service_s = tw_start_j # se espera hasta el inicio de la ventana
            
            # checa si llegó después del final de la ventana
            if t_start_service_s > tw_end_twin:
                missed_nodes.append(node_j)
                t_depart_j_s = t_arrival_s # ni modo mi chavo, te vas
                total_drive_s += t_travel_s
                current_time_s = t_depart_j_s
                continue # se salta el servicio
        
        # caso en que no hay ventanas de tiempo o llega dentro de la ventana
        t_depart_j_s = t_start_service_s + service_time_j
        total_drive_s += t_travel_s
        total_wait_s += wait_j
        total_service_s += service_time_j
        current_time_s = t_depart_j_s

    return {
        "drive": total_drive_s,
        "service": total_service_s,
        "wait": total_wait_s,
        "missed_nodes": missed_nodes
    }


def simulate_chromosome(chromosome, start_time_s, time_matrix, distance_matrix,traffic_params, service_times_lookup, tw_starts_lookup, tw_ends_lookup):
    metrics_list = []

    for route in chromosome:
        metrics = simulate_route(route, start_time_s, time_matrix, traffic_params, service_times_lookup, tw_starts_lookup, tw_ends_lookup)
        # añadimos distancia
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i], route[i+1]]
        metrics['distance'] = total_distance
        metrics['route'] = route
        metrics_list.append(metrics)

    return metrics_list


def summarize_by_vehicle(metrics_list):
    route_summaries = []
    sol_drive_s = 0
    sol_service_s = 0
    sol_wait_s = 0
    sol_dist_km = 0
    
    for i, metrics in enumerate(metrics_list):
        total_time_s = metrics['drive'] + metrics['service'] + metrics['wait']
        cost = metrics['distance'] * cost_km
        
        route_summary = {
            "vehicle_id": i,
            "route": metrics['route'],
            "drive_s": metrics['drive'],
            "service_s": metrics['service'],
            "wait_s": metrics['wait'],
            "total_time_s": total_time_s,
            "distance_km": metrics['distance'],
            "cost_mxn": cost,
            "missed_nodes": metrics['missed_nodes']
        }
        route_summaries.append(route_summary)
        
        sol_drive_s += metrics['drive']
        sol_service_s += metrics['service']
        sol_wait_s += metrics['wait']
        sol_dist_km += metrics['distance']
        
    sol_time_s = sol_drive_s + sol_service_s + sol_wait_s
    sol_cost = sol_dist_km * cost_km
    
    summary_report = {
        "routes": route_summaries,
        "summary": {
            "total_vehicles": len(route_summaries),
            "total_distance_km": sol_dist_km,
            "total_drive_s": sol_drive_s,
            "total_service_s": sol_service_s,
            "total_wait_s": sol_wait_s,
            "total_time_s": sol_time_s,
            "total_cost_mxn": sol_cost,
            "total_missed_stops": sum(len(r['missed_nodes']) for r in route_summaries)
        }
    }
    return summary_report


def check_capacity(chromosome, volume_lookup, max_capacity_cm3):
    unf_cap = 0
    for route in chromosome:
        # sumamos el volumen de cada paraad
        route_volume = sum(volume_lookup.get(stop, 0) for stop in route if stop != 0)
        if route_volume > max_capacity_cm3:
            unf_cap += 1
    return unf_cap


def check_work_time(summary_report, max_work_time_s):
    unf_work = 0
    for route_summary in summary_report['routes']:
        if route_summary['total_time_s'] > max_work_time_s:
            unf_work += 1
    return unf_work


def fitness_function(summary_report, capacity_violations, work_time_violations):
    # agarramos el costo total calculado como base
    total_cost = summary_report['summary']['total_cost_mxn']
    
    # añadimos penalizaciones para descartar estas soluciones
    penalty_capacity = capacity_violations * 1e6  
    penalty_work_time = work_time_violations * 1e6
    penalty_oot = summary_report['summary']['total_missed_stops'] * 1e6 # oot = out of time
    
    total_pen_cost = total_cost + penalty_capacity + penalty_work_time + penalty_oot
    fitness = 1.0 / (total_pen_cost)
    return fitness, total_pen_cost


def selection(pop_scores, num_parents):
    # sorteamos las soluciones basadas en la que mejor fitness tenga y de ahí para abajo
    pop_scores.sort(key = lambda x: x[0], reverse = True)
    
    # seleccionamos a las mejores soluciones basadas en un porcentaje de elección llamado elitism
    parents = [ind[1] for ind in pop_scores[:num_parents]]
    best_individual = pop_scores[0][1]
    best_fitness = pop_scores[0][0]
    best_cost = (1.0 / best_fitness)
    
    return parents, best_individual, best_fitness, best_cost

def one_point_crossover(parent1, parent2):
    num_routes = len(parent1) 
    if num_routes < 2:
        return parent1, parent2 # esto solo pasa si el número de padres es impar
        
    crossover = random.randint(1, num_routes - 1)
    
    child1 = parent1[:crossover] + parent2[crossover:]
    child2 = parent2[:crossover] + parent1[crossover:]
    
    return child1, child2

def mutation(chromosome, mutation_rate):
    mutated_chromosome = []
    for route in chromosome:
        if random.random() < mutation_rate and len(route) > 3: # necesitamos 2 cromosomas para mutar entre ellos
            # seleccionamos 2 índices random para mutar excepto el primero y último (ambos son el depósito)
            idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
            
            mutated_route = route[:] 
            mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1] # aquí sucede el swap
            mutated_chromosome.append(mutated_route)
        else:
            mutated_chromosome.append(route)
            
    return mutated_chromosome

# inputs, parámetros algoritmo, restricciones y parámetros tráfico
def genetic_algorithm(data, distance_matrix, time_matrix, stop2int, num_generations, population_size, 
    num_vehicles_k, mutation_rate, elitism_size, max_capacity_cm3, max_work_time_hours, traffic_params):

    # 1. extraemos los datos del dataset
    service_times, volumes, tw_starts, tw_ends, depot_start_s = load_lookups(data, stop2int)
    
    # 2. clusterizamos paradas
    clusters, dframe = cluster_stops(num_vehicles_k, data, stop2int)
    
    # 3. creamos la población
    population = initialize_population(population_size, clusters)
    
    max_work_time_s = max_work_time_hours * 3600
    
    best_solution_overall = None
    best_cost_overall = float('inf')
    history = [] # para trackear el progreso
    
    print("Algoritmo Genético para VRP")
    print(f"Generaciones: {num_generations}, Población: {population_size}, Vehículos: {num_vehicles_k}")
    
    for gen in range(num_generations):
        population_with_fitness = []
        
        # 4. evaluamos la población
        for individual in population:
            # simulamos cada ruta
            metrics_list = simulate_chromosome(individual, depot_start_s, time_matrix, distance_matrix, traffic_params, service_times, tw_starts, tw_ends)
            # resumen de cada ruta
            summary_report = summarize_by_vehicle(metrics_list)
            
            # checa las restricciones
            cap_violations = check_capacity(individual, volumes, max_capacity_cm3)
            work_violations = check_work_time(summary_report, max_work_time_s)
            
            # calcula el fitness
            fitness, penalized_cost = fitness_function(summary_report, cap_violations, work_violations)
            
            population_with_fitness.append((fitness, individual, penalized_cost, summary_report))

        # 5. elitismo
        num_parents = int(round(population_size * elitism_size))
        parents, best_ind_gen, best_fit_gen, best_cost_gen = selection(population_with_fitness, num_parents)
        
        # mantiene rastro de la mejor solución
        if best_cost_gen < best_cost_overall:
            best_cost_overall = best_cost_gen
            best_solution_overall = best_ind_gen

        history.append(best_cost_overall)
            
        # 6. crossover y mutación
        new_population = parents[:] # ahora usamos una nueva población pero con los mejores padres
        
        while len(new_population) < population_size:
            # nos aseguramos de que hay al menos 2 padres para samplear
            if len(parents) < 2:
                parents.append(random.choice(population_with_fitness)[1]) # añadir uno aleatorio si faltan
            
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = one_point_crossover(parent1, parent2)

            new_population.append(mutation(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutation(child2, mutation_rate))
                
        population = new_population

    print(f"Mejor costo encontrado: ${best_cost_overall:,.2f}")
    
    # encuentra el resumen de la mejor solución
    final_metrics = simulate_chromosome(best_solution_overall, depot_start_s, time_matrix, distance_matrix, traffic_params, service_times, tw_starts, tw_ends)
    final_summary_report = summarize_by_vehicle(final_metrics)

    return best_solution_overall, final_summary_report, history, dframe

if __name__ == "__main__":

    # con esta función nos encargamos de que independientemente de los datos que tengamos, el algoritmo jale,
    # entonces cuando hagamos la app no tengamos problema cuando los usuarios le piquen y le muevan parámetros
    def check_and_generate_data():

        if not os.path.exists('datos'): 
            os.makedirs('datos') # este directorio se va a sobreescribir con cada data set distinto

        # rutas de archivos
        dist_file = 'datos/distance_matrix.npy'
        time_file = 'datos/time_matrix.npy'
        ids_file = 'datos/stop_ids.npy'
        s2i_file = 'datos/stop2int.json'
        i2s_file = 'datos/int2stop.json'
        
        files_to_check = [dist_file, time_file, ids_file, s2i_file, i2s_file]
        
        # verificamos si falta algún archivo
        if not all(os.path.exists(f) for f in files_to_check):
            
            # generamos las matrices necesarias 
            dist_matrix, time_matrix, stop_ids = matrices()
            np.save(dist_file, dist_matrix)
            np.save(time_file, time_matrix)
            np.save(ids_file, stop_ids)

            # mapeos
            mapping()

    # entonces como digo, esto verificación y genera los datos necesarios independiente de cualquier cambio al data set
    check_and_generate_data()
    
    # cargamos los datos
    (data, dist_matrix, time_matrix, stop2int, int2stop) = load_all_data()

    if data is not None:
        # parámetros
        genes = 50  # genes not as in rutas sino de generaciones/iteraciones del algoritmo lol
        pop_size = 100
        mut_rate = 0.1
        elitism_rate = 0.2 
        num_vehicles = 2
        vehicle_cap = data['executor_capacity_cm3'].iloc[0] 
        work_hours = 8.0 
        traffic_parms = dict(A_m = 0.5, A_e = 0.5, mu_m = 8.0, mu_e = 17.0, sigma = 1.25)
        
        # algoritmo
        best_solution, best_summary, cost_history, df_viz_final = genetic_algorithm(
            data=data,
            distance_matrix = dist_matrix,
            time_matrix = time_matrix,
            stop2int = stop2int,
            num_generations = genes,
            population_size = pop_size,
            num_vehicles_k = num_vehicles,
            mutation_rate = mut_rate,
            elitism_size = elitism_rate,
            max_capacity_cm3 = vehicle_cap,
            max_work_time_hours = work_hours,
            traffic_params = traffic_parms
        )
        
        #  resultados
        print("\nMejor solución encontrada:")
        print(json.dumps(best_summary['summary'], indent=2))
        
        print("\nDetalle por Vehículo:")
        for route_info in best_summary['routes']:
            print(f"    Vehículo {route_info['vehicle_id']}:")
            print(f"    Paradas: {len(route_info['route']) - 2}")
            print(f"    Distancia: {route_info['distance_km']:.2f} km")
            print(f"    Tiempo Total: {str(datetime.timedelta(seconds=int(route_info['total_time_s'])))}")
            print(f"    Costo: ${route_info['cost_mxn']:,.2f}")
            print(f"    Paradas Fallidas: {len(route_info['missed_nodes'])}")
            
            # por si se quiere la ruta de números a ids leibles
            readable_route = [int2stop[str(stop)] for stop in route_info['route']] # int2stop usa keys de string
            print(f"Ruta: {', '.join(readable_route)}")
            
        # plot de costos de ruta
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history)
        plt.title('Convergencia del GA')
        plt.xlabel('t')
        plt.ylabel('Costo(t)')
        plt.grid(True)
        plt.show()

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
    tw_start_s, tw_end_s = load_time_windows()
    print(tw_start_s)
    print(tw_end_s)

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

    # pruebas de mapeo inverso
    """
    route_str = [int2stop[str(stop)] for stop in route_info['route']] 
    print(f"    Ruta: {' -> '.join(route_str)}")
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
