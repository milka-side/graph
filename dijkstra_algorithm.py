'''
Dijkstra's algorithm
'''
def min_index(distances: list[float], visited: list[int]) -> int:
    """
    Finds the index of the unvisited vertex with the smallest distance.

    Args:
        distances: A list of current shortest distances from the start node to each vertex.
        visited: A list of booleans where True indicates the vertex has already been visited.

    Returns:
        int: The index of the vertex with the minimum distance, or -1 if no reachable 
             unvisited vertices remain.
    """
    min_val = float('inf')
    minimum_index = -1
    for i, distance in enumerate(distances):
        if not visited[i] and distance < min_val:
            min_val = distance
            minimum_index = i
    return minimum_index

def dijksta_algorithm(graph: list[list[float]], start_vertex: int, end_vertex: int) -> tuple[float, list[int]]:
    """
    Finds the shortest path in a weighted graph using Dijkstra's algorithm.

    This function takes a graph represented as an adjacency matrix and returns
    the shortest distance and the path from the start vertex to the end vertex.

    Args:
        graph: A square matrix (list of lists) representing the graph. 
               graph[u][v] represents the weight of the edge from u to v.
               0 or float('inf') indicates no direct edge between vertices.
        start_vertex: The index of the starting vertex.
        end_vertex: The index of the destination vertex.

    Returns:
        tuple[float, list]: A tuple containing:
            - The shortest distance (float).
            - A list of integers representing the path from start to end.
            
        If no path exists, returns (float('inf'), []).

    Examples:
        >>> graph = [
        ...     [0, 4, 2],
        ...     [0, 0, 0],
        ...     [0, 1, 0]
        ... ]
        >>> dijksta_algorithm(graph, 0, 1)
        (3, [0, 2, 1])
        >>> graph_disconnected = [
        ...     [0, 0, 0],
        ...     [0, 0, 0],
        ...     [0, 0, 0]
        ... ]
        >>> dijksta_algorithm(graph_disconnected, 0, 2)
        (inf, [])
        >>> inf = float('inf')
        >>> graph_complex = [
        ...     [0, 10, 3, inf, inf],
        ...     [inf, 0, 1, 2, inf],
        ...     [inf, 4, 0, 8, 2],
        ...     [inf, inf, inf, 0, 7],
        ...     [inf, inf, inf, inf, 0]
        ... ]
        >>> dijksta_algorithm(graph_complex, 0, 3)
        (9, [0, 2, 1, 3])
    """
    n = len(graph)
    distances = [float('inf')] * n
    distances[start_vertex] = 0
    visited = [False] * n
    parents = [None] * n
    for _ in range(n):
        current_vertex = min_index(distances, visited)
        if current_vertex == -1 or distances[current_vertex] == float('inf'):
            break
        visited[current_vertex] = True
        for neighbor_index, weight in enumerate(graph[current_vertex]):
            if weight == float('inf') or weight == 0:
                continue
            new_distance = distances[current_vertex] + weight
            if new_distance < distances[neighbor_index]:
                distances[neighbor_index] = new_distance
                parents[neighbor_index] = current_vertex
    path = []
    current = end_vertex
    if distances[end_vertex] == float('inf'):
        return float('inf'), []
    while current is not None:
        path.append(current)
        current = parents[current]
    return distances[end_vertex], path[::-1]

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
