'''
Docstring for bellman_ford_algorithm
'''

def bellman_ford_algorithm(graph: list[list[float]], start_vertex: int, end_vertex: int) \
    -> tuple[float, list[int]]:
    """
    Finds the shortest path in a weighted graph using the Bellman-Ford algorithm.
    
    This algorithm can handle graphs with negative edge weights. It detects 
    negative weight cycles and raises a ValueError if one exists.

    Args:
        graph: A square matrix (list of lists) representing the graph.
               graph[u][v] represents the weight of the edge from u to v.
               0 or float('inf') indicates no direct edge.
        start_vertex: The index of the starting vertex.
        end_vertex: The index of the destination vertex.

    Returns:
        tuple[float, list[int]]: A tuple containing:
            - The shortest distance (float).
            - A list of integers representing the path from start to end.

    Raises:
        ValueError: If the graph contains a negative weight cycle reachable from the start vertex.

    Examples:
        >>> # Example 1: Graph with negative weight (no cycle)
        >>> # 0 -> 1 (cost 4)
        >>> # 0 -> 2 (cost 5)
        >>> # 2 -> 1 (cost -2) -> Best path to 1 is 0->2->1 (cost 3)
        >>> inf = float('inf')
        >>> graph_neg = [
        ...     [0, 4, 5],
        ...     [inf, 0, inf],
        ...     [inf, -2, 0]
        ... ]
        >>> bellman_ford_algorithm(graph_neg, 0, 1)
        (3, [0, 2, 1])

        >>> # Example 2: Negative Weight Cycle
        >>> # 0 -> 1 (1)
        >>> # 1 -> 2 (-5)
        >>> # 2 -> 0 (2)
        >>> # Cycle: 1 - 5 + 2 = -2 (negative loop)
        >>> graph_cycle = [
        ...     [0, 1, inf],
        ...     [inf, 0, -5],
        ...     [2, inf, 0]
        ... ]
        >>> try:
        ...     bellman_ford_algorithm(graph_cycle, 0, 2)
        ... except ValueError as e:
        ...     print(e)
        Graph contains a negative weight cycle

        >>> # Example 3: Unreachable node
        >>> graph_unreachable = [[0, inf], [inf, 0]]
        >>> bellman_ford_algorithm(graph_unreachable, 0, 1)
        (inf, [])
    """
    n = len(graph)
    distances = [float('inf')] * n
    distances[start_vertex] = 0
    parents = [None] * n
    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if distances[u] == float('inf'):
                continue
            for v, weight in enumerate(graph[u]):
                if weight == float('inf') or weight == 0:
                    continue
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    parents[v] = u
                    updated = True
        if not updated:
            break
    for u in range(n):
        if distances[u] == float('inf'):
            continue
        for v, weight in enumerate(graph[u]):
            if weight == float('inf') or weight == 0:
                continue
            if distances[u] + weight < distances[v]:
                raise ValueError("Graph contains a negative weight cycle")
    path = []
    curr = end_vertex
    if distances[end_vertex] == float('inf'):
        return float('inf'), []
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
        if len(path) > n:
            break

    return distances[end_vertex], path[::-1]

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
