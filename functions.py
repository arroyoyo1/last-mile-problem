import numpy as np
import json

def mapping():

    # cargamos el array de stop_ids
    stop_ids = np.load('datos/stop_ids.npy')

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

if __name__ == '__main__':

    # stop2int, int2stop = mapping()

    """
    # para verificar mapping()
    print(f"UCLA5 - {stop2int.get('UCLA5')}")
    print(f"0 - {int2stop.get(0)}")
    print(f"'AD' - {stop2int.get('AD')}")
    print(f"10 - '{stop2int .get('10')}'") 
    """