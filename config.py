
# parámetros constantes del proyecto

BIG_M = 1e6

# parámetros del modelo de tráfico
A_M = 0.5        # extra_mañana
A_E = 0.5        # extra_tarde
MU_M = 8.0       # mu_mañana 
MU_E = 17.0      # mu_tarde
SIGMA = 0.75     # ancho de los picos (horas)

# operación del vehículo
DEFAULT_SPEED_KMH = 70.0    
COST_PER_KM = 2.21           # g_k ($ / km)
WORKDAY_SECONDS = 8 * 3600   # h_k (8 horas)

# coordenadas del depósito
DEPOT_COORD = (34.118289, -118.249262)  # (lat, lng)

# Nombres de archivos de salida generados por matrices.py
DIST_MATRIX_FILE = "dist_matrix.npy"        # km
TIME_BASE_MATRIX_FILE = "time_base_matrix.npy"  # seconds
MAPPING_FILE = "stop_mapping.json"          # mapping stop_id -> index
NODES_FILE = "nodes_list.json"              # list of (lat,lng) por índice

# Parámetros para la extracción de grafo OSM
OSM_BUFFER_M = 2000  # buffer (metros) alrededor del bbox para descargar la red

# Logging / performance
USE_TQDM = True  # mostrar barra de progreso al computar
