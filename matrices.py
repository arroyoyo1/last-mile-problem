import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

velocity = 60 # velocidad promedio

def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0  # radio de la tierra
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # resta entre coordenadas
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # fórmula haversine
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = radius * c
    return distance

def matrices():
    
    df = pd.read_csv("inputs.csv")
    stops_df = df[['stop_id', 'stop_lat', 'stop_lng']]

    # hacemos una lista de todas las coordenadas con su respectivo stop_id
    depot_lat, depot_lon = (20.613801, -100.402868) 
    coords = [(depot_lat, depot_lon)] + list(zip(stops_df['stop_lat'], stops_df['stop_lng'])) # zip empareja lat y lon
    stop_ids = ['Lucina'] + list(stops_df['stop_id'])
    
    num_locations = len(coords)
    distance_matrix = np.zeros((num_locations, num_locations))
    time_matrix = np.zeros((num_locations, num_locations))

    # llenamos las matrices
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:
                continue # diagonal de ceros
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            
            # calculamos y metemos la distancia
            dist = haversine(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = dist
            
            # calculamos y metemos el tiempo en segundos
            time_seconds = (dist / velocity) * 3600
            time_matrix[i, j] = time_seconds

    return distance_matrix, time_matrix, stop_ids

if __name__ == '__main__':
    dist_matrix, time_matrix, stop_ids = matrices()

    # guardamos matrices
    np.save('datos/distance_matrix.npy', dist_matrix)
    np.save('datos/time_matrix.npy', time_matrix)
    np.save('datos/stop_ids.npy', stop_ids)
    dist_matrix = np.load('datos/distance_matrix.npy')
    time_matrix = np.load('datos/time_matrix.npy')
    stop_ids = np.load("datos/stop_ids.npy")
    print(dist_matrix[:5, :5])
    print(time_matrix[:5, :5])
    print(stop_ids)

