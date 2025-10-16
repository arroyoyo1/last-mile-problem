
# calcula distancias viales y tiempos base entre todos los stops usando OSMnx
# y las guarda en dist_matrix.npy (km), time_base_matrix.npy (s), mapping stop->idx (json)

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from config import (
    DIST_MATRIX_FILE, TIME_BASE_MATRIX_FILE, MAPPING_FILE, NODES_FILE,
    DEPOT_COORD, OSM_BUFFER_M, USE_TQDM
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

def read_inputs(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

def build_stop_mapping(df: pd.DataFrame, depot_coord=DEPOT_COORD):
    """
    Crea:
      - lista de nodos coords: index 0 = depot, 1..n = stops
      - mapping stop_id -> index (int)
    Retorna: nodes_list ([(lat,lng), ...]), stop2idx (dict), idx2stop (dict)
    """
    # Extraer stops únicos en el orden en que aparecen (assert no duplicados)
    if df['stop_id'].duplicated().any():
        raise ValueError("Se detectaron stop_id duplicados en inputs.csv (no debe ocurrir).")
    stop_ids = df['stop_id'].tolist()
    # ordenar único (ya que no se repiten están en orden dataset)
    # Prepend depot as index 0
    nodes = [depot_coord]  # index 0
    stop2idx = {}
    idx2stop = {}
    for i, sid in enumerate(stop_ids, start=1):
        lat = float(df.loc[df['stop_id'] == sid, 'stop_lat'].iloc[0])
        lng = float(df.loc[df['stop_id'] == sid, 'stop_lng'].iloc[0])
        nodes.append((lat, lng))
        stop2idx[sid] = i
        idx2stop[i] = sid
    # depot mapping (optional)
    idx2stop[0] = "DEPOT"
    stop2idx["DEPOT"] = 0
    return nodes, stop2idx, idx2stop

def build_osm_graph(nodes, buffer_m=OSM_BUFFER_M, network_type='drive'):
    """
    Descarga la subred vial que cubre todos los puntos usando graph_from_point centrado
    en el centroid. Este enfoque evita incompatibilidades de firma entre versiones
    de OSMnx (por ejemplo graph_from_bbox que cambia argumentos posicionales/keyword).
    """
    lats = [lat for lat, lng in nodes]
    lngs = [lng for lat, lng in nodes]

    # centroid de los puntos
    centroid_lat = sum(lats) / len(lats)
    centroid_lng = sum(lngs) / len(lngs)

    print(f"Descargando grafo OSM usando graph_from_point centrado en ({centroid_lat:.6f}, {centroid_lng:.6f}) con buffer {buffer_m} m ...")

    # Intento robusto con retries simples
    last_exc = None
    for attempt, dist in enumerate([buffer_m, int(buffer_m*1.5), int(buffer_m*2)], start=1):
        try:
            G = ox.graph_from_point((centroid_lat, centroid_lng), dist=dist, network_type=network_type)
            print(f"Grafo descargado con graph_from_point (dist={dist} m) en intento {attempt}.")
            # Añadir velocidades estimadas y tiempos si es posible
            try:
                ox.add_edge_speeds(G)
                ox.add_edge_travel_times(G)
            except Exception as e:
                print("Warning: add_edge_speeds / add_edge_travel_times falló:", e)
            return G
        except TypeError as te:
            # Si la firma de graph_from_point también está rara, capturamos y probamos otra cosa
            last_exc = te
            print(f"graph_from_point lanzó TypeError en intento {attempt}: {te}")
        except Exception as e:
            last_exc = e
            print(f"graph_from_point falló en intento {attempt} con dist={dist}: {e}")

    # Si llegamos aquí, intentar un último fallback más directo con graph_from_bbox usando kwargs (por si acaso)
    try:
        north = max(lats) + 0.01
        south = min(lats) - 0.01
        east  = max(lngs) + 0.01
        west  = min(lngs) - 0.01
        print("Intentando fallback final con graph_from_bbox usando kwargs...")
        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type=network_type)
        try:
            ox.add_edge_speeds(G)
            ox.add_edge_travel_times(G)
        except Exception:
            pass
        return G
    except Exception as final_e:
        raise RuntimeError(
            "No fue posible descargar la red OSM con graph_from_point ni graph_from_bbox. "
            "Revisa la versión de OSMnx, la conectividad a internet y las dependencias nativas. "
            f"Último error: {final_e}. Error previo: {last_exc}"
        )

def snap_nodes_to_graph(G, nodes):
    """
    Obtiene el nodo OSM más cercano para cada punto de coords.
    Usamos la versión vectorizada ox.nearest_nodes(G, X=lngs, Y=lats)
    para evitar errores de firma y acelerar el proceso.
    Retorna lista nodes_osm_nodes (id de nodo OSM) alineada con indices (0..n)
    """
    print("Snapear coordenadas a nodos OSM (nearest_nodes vectorizado)...")
    # separar listas
    lats = [lat for lat, lng in nodes]
    lngs = [lng for lat, lng in nodes]

    # OSMnx espera X=longitudes, Y=latitudes
    try:
        osm_nodes = ox.nearest_nodes(G, X=lngs, Y=lats)
        # ox.nearest_nodes puede devolver un numpy array; convertir a lista
        osm_nodes = list(osm_nodes)
        return osm_nodes
    except TypeError as te:
        # En caso de firma inesperada, intentar llamada por punto (compatibilidad)
        print("nearest_nodes firma inesperada, fallback punto por punto:", te)
        osm_nodes = []
        for lat, lng in tqdm(nodes) if USE_TQDM else nodes:
            try:
                osm_n = ox.nearest_nodes(G, lng, lat)  # lon, lat
            except Exception:
                # Último recurso: usar get_nearest_node si existe
                try:
                    osm_n = ox.get_nearest_node(G, (lat, lng))
                except Exception as e:
                    raise RuntimeError(f"No fue posible snapear punto {(lat,lng)}: {e}")
            osm_nodes.append(osm_n)
        return osm_nodes
    except Exception as e:
        raise RuntimeError("Error al snapear nodos a la red OSM: " + str(e))


def compute_matrices(G, osm_nodes):
    """
    Para cada fuente i (nodos OSM), corre Dijkstra (weight='length' para distancia en metros)
    y weight='travel_time' para tiempo en segundos cuando está disponible.
    Si travel_time no está disponible o falla, usamos un fallback:
       time_s = distance_m * 3.6 / DEFAULT_SPEED_KMH   (segundos)
    Devuelve matrices:
      - dist_matrix (km) shape (N,N)
      - time_base_matrix (s) shape (N,N)
    """
    N = len(osm_nodes)
    dist_m = np.full((N, N), np.inf, dtype=float)
    time_s = np.full((N, N), np.inf, dtype=float)

    print("Computando matrices de distancia y tiempo: para cada nodo OSM correr Dijkstra...")
    iterable = tqdm(range(N)) if USE_TQDM else range(N)
    for i in iterable:
        source = osm_nodes[i]
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='length')  # metros
        except Exception as e:
            print(f"Error Dijkstra (length) desde nodo {source}: {e}")
            lengths = {}

        # Intentar obtener times con travel_time; si falla, lo ignoramos (fallback abajo)
        try:
            times = nx.single_source_dijkstra_path_length(G, source, weight='travel_time')  # segundos
        except Exception:
            times = None

        for j in range(N):
            target = osm_nodes[j]
            d_m = lengths.get(target, None)
            t_s = None
            if times is not None:
                t_s = times.get(target, None)

            if d_m is None:
                dist_m[i, j] = np.inf
                time_s[i, j] = np.inf
            else:
                dist_m[i, j] = d_m / 1000.0  # metros -> km
                if t_s is None:
                    # fallback: usar velocidad por defecto (DEFAULT_SPEED_KMH)
                    # tiempo (s) = distancia_m * 3.6 / v_kmh
                    # porque: (d_m/1000 km) / (v_kmh km/h) * 3600 s/h = d_m * 3.6 / v_kmh
                    from config import DEFAULT_SPEED_KMH
                    time_s[i, j] = float(d_m * 3.6 / DEFAULT_SPEED_KMH)
                else:
                    time_s[i, j] = float(t_s)
    return dist_m, time_s


def save_outputs(out_dir: str, dist_matrix, time_base_matrix, nodes, stop2idx):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dist_path = Path(out_dir) / DIST_MATRIX_FILE
    time_path = Path(out_dir) / TIME_BASE_MATRIX_FILE
    mapping_path = Path(out_dir) / MAPPING_FILE
    nodes_path = Path(out_dir) / NODES_FILE

    np.save(dist_path, dist_matrix)
    np.save(time_path, time_base_matrix)

    # mapping stop->index (json)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(stop2idx, f, ensure_ascii=False, indent=2)

    # save nodes list (index -> (lat,lng))
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    print(f"Guardado dist_matrix → {dist_path}")
    print(f"Guardado time_base_matrix → {time_path}")
    print(f"Guardado mapping stop->idx → {mapping_path}")
    print(f"Guardado nodes list → {nodes_path}")

def main(csv_path: str, out_dir: str = ".", force_rebuild_graph: bool = False):
    print("Leyendo inputs.csv ...")
    df = read_inputs(csv_path)

    print("Construyendo mapeo stop_id -> index ...")
    nodes, stop2idx, idx2stop = build_stop_mapping(df, depot_coord=DEPOT_COORD)

    print("Construyendo grafo OSM (puede tardar unos segundos/minutos)...")
    G = build_osm_graph(nodes)

    print("Snap nodes a la red y obtener nodos OSM ...")
    osm_nodes = snap_nodes_to_graph(G, nodes)

    print("Precomputando matrices (distancias en km, tiempos base en segundos)...")
    dist_matrix, time_base_matrix = compute_matrices(G, osm_nodes)

    print("Guardando archivos...")
    save_outputs(out_dir, dist_matrix, time_base_matrix, nodes, stop2idx)
    print("¡Hecho!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute road distance & base travel times with OSMnx")
    parser.add_argument("--csv", type=str, default="inputs.csv", help="Path to inputs.csv (default ./inputs.csv)")
    parser.add_argument("--out", type=str, default=".", help="Output directory to save matrices and mapping")
    args = parser.parse_args()
    main(args.csv, args.out)
