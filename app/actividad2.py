import heapq

# Definir el grafo del sistema de transporte masivo
grafo_transporte = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 1, 'D': 4},
    'C': {'A': 5, 'B': 1, 'D': 2},
    'D': {'B': 4, 'C': 2, 'E': 3},
    'E': {'D': 3}
}

def dijkstra(grafo, inicio, destino):
    cola, dist, ruta = [(0, inicio)], {n: float('inf') for n in grafo}, {}
    dist[inicio] = 0
    while cola:
        costo, nodo = heapq.heappop(cola)
        if nodo == destino: break
        for vecino, peso in grafo[nodo].items():
            nuevo_costo = costo + peso
            if nuevo_costo < dist[vecino]:
                dist[vecino], ruta[vecino] = nuevo_costo, nodo
                heapq.heappush(cola, (nuevo_costo, vecino))
    camino, actual = [], destino
    while actual: camino.append(actual); actual = ruta.get(actual)
    return camino[::-1], dist[destino]

camino, costo = dijkstra(grafo_transporte, 'B', 'D')
print(f"Ruta: {' -> '.join(camino)} con costo total de {costo}")