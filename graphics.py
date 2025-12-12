'''Module for comparing effectiveness of algorithms'''
import csv
import timeit
import matplotlib.pyplot as plt

# =====================================================
# Зчитування й перетворення файлу в матрицю суміжності
# =====================================================

def load_csv_to_matrix(filename):
    """
    Reads a CSV file containing graph edges and returns an adjacency matrix.

    Expected CSV format (with or without headers):
    source, target, weight
    A, B, 10
    B, C, 5
    ...

    Returns:
    1. matrix (list of lists): NxN adjacency matrix where values represent edge weights.
       - No edge = float('inf')
       - Diagonal (self-loop) = 0.0
    2. nodes (list): List of node names mapping indices to labels.
       Example: nodes[0] == 'A', nodes[1] == 'B'.
    """
    edges = []
    unique_nodes = set()

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            try:
                if len(row) >= 3:
                    u, v = row[0].strip(), row[1].strip()
                    weight = float(row[2])
                    edges.append((u, v, weight))
                    unique_nodes.add(u)
                    unique_nodes.add(v)
            except ValueError:
                continue

    sorted_nodes = sorted(list(unique_nodes))
    node_to_index = {name: i for i, name in enumerate(sorted_nodes)}
    n = len(sorted_nodes)

    matrix = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 0.0

    for u, v, w in edges:
        idx_u = node_to_index[u]
        idx_v = node_to_index[v]
        matrix[idx_u][idx_v] = w
        # matrix[idx_v][idx_u] = w  # Якщо неорієнтований

    return matrix  #, sorted_nodes


# ==========================================
# АНАЛІЗ. ПОБУДОВА ГРАФІКІВ.
# ==========================================

def run_benchmark_with_bellman(
    generator_func,                         # Function(n, sparsity) -> матриця суміжності
    dijksta_algorithm,                      # Function(graph, start, end)
    algo_new,                               # Function(graph, start, end)
    n_nodes,                                # Int: кількість вершин через argparse
    bellman_ford_algorithm=None,            # Optional: алгоритм Беллмана-Форда
    output_file="project_discra.png"
):
    '''
    Функція для побудови графіків
    '''

    times_dijkstra = []
    times_new = []
    times_bellman = []

    sparsity_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    for density in sparsity_list:
        try:
            graph = generator_func(n_nodes, density)
            start = 0
            end = len(graph) - 1
        except Exception:
            print("Gen Error")
            return

        #1 Dijkstra
        t0 = timeit.default_timer()
        dijksta_algorithm(graph, start, end)
        times_dijkstra.append(timeit.default_timer() - t0)

        #2 New Algorithm
        t0 = timeit.default_timer()
        algo_new(graph, start, end)
        times_new.append(timeit.default_timer() - t0)

        #3 Bellman-Ford
        # if n_nodes > 500:
        #     times_bellman.append(None)
        # else:

        try:
            t0 = timeit.default_timer()
            bellman_ford_algorithm(graph, start, end)
            times_bellman.append(timeit.default_timer() - t0)
        except Exception as e:
            print(f"BF Error: {e}")
            times_bellman.append(None)


    plt.figure(figsize=(10, 6))

    if times_bellman:
        plt.plot(sparsity_list, times_bellman, 'g-^', label='Bellman-Ford (O(VE))')
    plt.plot(sparsity_list, times_dijkstra, 'r-o', label='Dijkstra')
    plt.plot(sparsity_list, times_new, 'b-s', label='New Algorithm')

    plt.yscale('log')

    plt.title(f'Performance Comparison - {n_nodes} Nodes')
    plt.xlabel('Sparsity')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.savefig(output_file)
    plt.show()
