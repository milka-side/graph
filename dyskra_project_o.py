"""
Graph Shortest Path Comparison Tool
-----------------------------------

This script generates or loads a weighted directed graph and compares the performance
and results of three shortest path algorithms:

1. Dijkstra's Algorithm
2. BMSSP (a custom multi-step shortest path algorithm)
3. Bellman-Ford Algorithm

Features:
- Generates random sparse directed graphs with configurable number of nodes,
density, and max edge weight.
- Loads graphs from CSV files with format: source, target, weight.
- Prints adjacency matrix of the graph.
- Finds shortest paths between source and target nodes for all three algorithms.
- Computes differences between path lengths for verification.
- Benchmarks execution time of all algorithms for varying graph densities and saves a log-time plot.

Usage:
    python BMSSP2.py [options]

Options:
    -n, --nodes            Number of nodes in the graph (default: 50)
    -d, --density          Probability of edge existence (default: 0.3)
    -w, --max-weight       Maximum random edge weight (default: 10)
    --seed                 Random seed for reproducibility
    --source               Source node index (default: 0)
    --target               Target node index (default: last node)
    -l, --recursion-depth  BMSSP recursion depth (default: 2)
    --print-matrix         Print adjacency matrix
    --csv                  Path to CSV file to load graph from
    --force-benchmark      Run benchmark even if CSV is provided
    --benchmark-output     Filename to save benchmark plot (default: benchmark_plot.png)
"""

from __future__ import annotations
from typing import List, Tuple, Set, Optional
import heapq
import random
import csv
import timeit
import math
import argparse
import matplotlib.pyplot as plt


Node = int
Weight = int
Edge = Tuple[Node, Node, Weight]
Graph = List[List[Tuple[Node, Weight]]]


# *** Loading from csv file ***
def load_csv_to_matrix(filename):
    """
    Reads a CSV file containing graph edges and returns an adjacency matrix.
    """
    edges = []
    unique_nodes = set()

    with open(filename, "r", encoding="utf-8") as f:
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

    sorted_nodes = sorted(unique_nodes)
    node_to_index = {name: i for i, name in enumerate(sorted_nodes)}
    n = len(sorted_nodes)

    matrix = [[float("inf")] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 0

    for u, v, w in edges:
        idx_u = node_to_index[u]
        idx_v = node_to_index[v]
        matrix[idx_u][idx_v] = w

    return matrix


# *** Benchmark function ***
def run_benchmark_with_bellman(
    generator_func,
    dijksta_algorithm,
    algo_new,
    n_nodes,
    bellman_ford_algorithm=None,
    output_file="project_discra.png",
):
    """
    Benchmark function to compare Dijkstra, BMSSP, and optionally Bellman-Ford.
    Plots a log-time graph of execution times vs sparsity.
    """
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
            print("Graph generation error")
            return

        t0 = timeit.default_timer()
        dijksta_algorithm(graph, start, end)
        times_dijkstra.append(timeit.default_timer() - t0)

        t0 = timeit.default_timer()
        algo_new(graph, start, end)
        times_new.append(timeit.default_timer() - t0)

        try:
            t0 = timeit.default_timer()
            bellman_ford_algorithm(graph, start, end)
            times_bellman.append(timeit.default_timer() - t0)
        except Exception as e:
            print(f"Bellman-Ford error: {e}")
            times_bellman.append(None)

    plt.figure(figsize=(10, 6))
    if times_bellman:
        plt.plot(sparsity_list, times_bellman, "g-^", label="Bellman-Ford (O(VE))")
    plt.plot(sparsity_list, times_dijkstra, "r-o", label="Dijkstra")
    plt.plot(sparsity_list, times_new, "b-s", label="New Algorithm")

    plt.yscale("log")
    plt.title(f"Performance Comparison - {n_nodes} Nodes")
    plt.xlabel("Sparsity")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(output_file)
    plt.show()


# *** Bellman-Ford algorithm ***
def bellman_ford_path(
    graph: Graph, source: Node, target: Node
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Bellman-Ford algorithm for shortest path in weighted directed graph.
    Returns path edges and total distance.
    Raises ValueError if a negative-weight cycle is detected.
    """
    n = len(graph)
    dist = [float("inf")] * n
    parent: List[Optional[int]] = [None] * n
    dist[source] = 0.0

    for _ in range(n - 1):
        for u in range(n):
            for v, w in enumerate(graph[u]):
                if not math.isfinite(w):
                    continue
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u

    for u in range(n):
        for v, w in enumerate(graph[u]):
            if not math.isfinite(w):
                continue
            if dist[u] + w < dist[v]:
                raise ValueError("Graph contains a negative-weight cycle")

    if not math.isfinite(dist[target]):
        return [], float("inf")

    path_nodes: List[int] = []
    u = target
    while u is not None:
        path_nodes.append(u)
        u = parent[u]
    path_nodes.reverse()
    path_edges = [
        (path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)
    ]
    return path_edges, dist[target]


# *** Dijkstra algorithm ***
def dijkstra_path(
    graph: Graph, source: Node, target: Node
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Standard Dijkstra algorithm to find shortest path in a weighted graph represented as a matrix.
    Returns path edges and total distance.
    """
    n = len(graph)
    dist = [float("inf")] * n
    parent: List[Optional[int]] = [None] * n
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        for v, w in enumerate(graph[u]):
            if not math.isfinite(w):
                continue
            newd = dist[u] + w
            if newd < dist[v]:
                dist[v] = newd
                parent[v] = u
                heapq.heappush(heap, (newd, v))

    if not math.isfinite(dist[target]):
        return [], float("inf")

    path_nodes: List[int] = []
    u = target
    while u is not None:
        path_nodes.append(u)
        u = parent[u]
    path_nodes.reverse()
    path_edges = [
        (path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)
    ]
    return path_edges, dist[target]


# *** BMSSP algorithm ***
def find_pivots(
    graph: Graph, dist: List[Weight], S: Set[Node], k_steps: int, p_limit: int
) -> Tuple[Set[Node], Set[Node]]:
    """
    Select pivot nodes for BMSSP algorithm and compute reachable nodes within k_steps.
    Returns tuple of pivot nodes (P) and all reached nodes (W).
    """
    if not S:
        return set(), set()
    S_sorted = sorted(S, key=lambda v: dist[v])
    P = set(S_sorted[: min(p_limit, len(S_sorted))])
    W = set(P)
    frontier = set(P)
    for _ in range(k_steps):
        next_front = set()
        for u in frontier:
            du = dist[u]
            if not math.isfinite(du):
                continue
            row = graph[u]
            for v, w in enumerate(row):
                if v in W or not math.isfinite(w):
                    continue
                W.add(v)
                next_front.add(v)
        frontier = next_front
        if not frontier:
            break
    return P, W

def bmssp_path(
    graph: Graph, source: Node, target: Node, l: int = 2, k_param: int = 50
) -> Tuple[List[Tuple[int, int]], float]:
    """
    BMSSP shortest path algorithm (multi-step recursive exploration).
    Returns path as list of edges and total distance.
    """
    n = len(graph)
    dist = [float("inf")] * n
    parent: List[Optional[int]] = [None] * n
    dist[source] = 0.0

    def bmssp_recur(S: Set[Node], l: int):
        if not S:
            return set()
        if l <= 0:
            x = min(S, key=lambda v: dist[v])
            heap = [(dist[x], x)]
            Uo = set()
            while heap and len(Uo) < k_param + 1:
                d_u, u = heapq.heappop(heap)
                if d_u > dist[u]:
                    continue
                Uo.add(u)
                row = graph[u]
                for v, w in enumerate(row):
                    if not math.isfinite(w):
                        continue
                    newd = dist[u] + w
                    if newd < dist[v]:
                        dist[v] = newd
                        parent[v] = u
                        heapq.heappush(heap, (newd, v))
            return Uo

        p_limit = min(5, len(S))
        P, W = find_pivots(graph, dist, S, k_steps=2, p_limit=p_limit)
        U = set()
        for x in P:
            U_sub = bmssp_recur({x}, l - 1)
            U |= U_sub
            for u in list(U_sub):
                row = graph[u]
                for v, w in enumerate(row):
                    if not math.isfinite(w):
                        continue
                    newd = dist[u] + w
                    if newd < dist[v]:
                        dist[v] = newd
                        parent[v] = u
                        U.add(v)
        U |= W
        return U

    bmssp_recur({source}, l)

    if not math.isfinite(dist[target]):
        return [], float("inf")

    path_nodes: List[int] = []
    u = target
    while u is not None:
        path_nodes.append(u)
        u = parent[u]
    path_nodes.reverse()
    path_edges = [
        (path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)
    ]
    return path_edges, dist[target]


# *** Graph generation ***
def generate_sparse_directed_graph(n: int, density: float, max_w: int = 10) -> Graph:
    """
    Generate a random sparse directed graph as adjacency matrix.
    n: number of nodes
    density: probability of edge existence
    max_w: maximum edge weight
    """
    graph: Graph = [[math.inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        graph[i][i] = 0
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < density:
                graph[i][j] = random.randint(1, max_w)
    return graph


# *** Function to print adjacency matrix ***
def print_adjacency_matrix(graph: Graph):
    """
    Print adjacency matrix with 'inf' representing no edge.
    """
    print("Adjacency matrix (math.inf = no edge):")
    for i, row in enumerate(graph):
        row_str = []
        for w in row:
            if math.isinf(w):
                row_str.append("inf")
            else:
                row_str.append(f"{w:.1f}")
        print(f"{i}: {row_str}")


# *** Main function ***
def main():
    """
    Parse arguments, load or generate graph, run Dijkstra, BMSSP, Bellman-Ford,
    print paths and lengths, optionally run benchmark.
    """
    parser = argparse.ArgumentParser(
        description="Compare Dijkstra, BMSSP, and Bellman-Ford on a graph"
    )
    parser.add_argument(
        "-n", "--nodes", type=int, default=500, help="Number of nodes (default: 50)"
    )
    parser.add_argument(
        "-d",
        "--density",
        type=float,
        default=0.3,
        help="Edge probability / density (default: 0.3)",
    )
    parser.add_argument(
        "-w",
        "--max-weight",
        type=int,
        default=10,
        help="Maximum random edge weight (default: 10)",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start node index (default: 0)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target node index (default: last node)",
    )
    parser.add_argument(
        "-l",
        "--recursion-depth",
        type=int,
        default=2,
        help="BMSSP recursion depth (default: 2)",
    )
    parser.add_argument(
        "--print-matrix", action="store_true", help="Print adjacency matrix"
    )
    parser.add_argument(
        "--source", type=str, default=None, help="Path to CSV file with edges"
    )
    parser.add_argument(
        "--force-benchmark",
        action="store_true",
        help="Run benchmark even if CSV provided",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark.png",
        help="Benchmark output filename",
    )

    args = parser.parse_args()

    if args.source:
        print(f"Loading graph from CSV: {args.source}")
        try:
            graph = load_csv_to_matrix(args.source)
        except FileNotFoundError:
            print(f"CSV file not found: {args.source}")
            return
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return
        n = len(graph)

        if args.start < 0 or args.start >= n:
            print(f"Start {args.start} out of bounds (0..{n-1}). Using 0.")
            source = 0
        else:
            source = args.start
        if args.target is not None:
            if args.target < 0 or args.target >= n:
                print(
                    f"Target {args.target} out of bounds (0..{n-1}). Using last node."
                )
                target = n - 1
            else:
                target = args.target
        else:
            target = n - 1
    else:
        print("Generating random graph...")
        n = args.nodes
        graph = generate_sparse_directed_graph(n, args.density, args.max_weight)
        source = args.start
        target = args.target if args.target is not None else n - 1
    
    example_graph = generate_sparse_directed_graph(5, 0.3, 10)
    l = args.recursion_depth

    print_adjacency_matrix(example_graph)

    # Run Dijkstra
    print("\nRunning Dijkstra...")
    path_dij, length_dij = dijkstra_path(graph, source, target)
    print(f"Dijkstra shortest path from {source} to {target}:")
    print(f"  edges: {path_dij}")
    print(f"  length: {length_dij}")

    # Run BMSSP
    print(f"\nRunning BMSSP (l={l})...")
    path_bm, length_bm = bmssp_path(graph, source, target, l=l)
    print(f"BMSSP (l={l}) path from {source} to {target}:")
    print(f"  edges: {path_bm}")
    print(f"  length: {length_bm}")

    # Run Bellman-Ford
    print("\nRunning Bellman-Ford...")
    try:
        path_bf, length_bf = bellman_ford_path(graph, source, target)
        print(f"Bellman-Ford path from {source} to {target}:")
        print(f"  edges: {path_bf}")
        print(f"  length: {length_bf}")
    except ValueError as e:
        print(f"Bellman-Ford error: {e}")
        length_bf = None

    if (
        math.isfinite(length_dij)
        and math.isfinite(length_bm)
        and (length_bf is not None)
    ):
        print(f"\nLength differences:")
        print(f"  |Dijkstra - BMSSP| = {abs(length_dij - length_bm)}")
        print(f"  |Dijkstra - Bellman-Ford| = {abs(length_dij - length_bf)}")
        print(f"  |BMSSP - Bellman-Ford| = {abs(length_bm - length_bf)}")

    do_benchmark = (not args.source) or args.force_benchmark

    if do_benchmark:
        print("\nRunning benchmark for different graph densities...")
        bench_n = args.nodes if args.nodes is not None else max(50, n)
        run_benchmark_with_bellman(
            generator_func=generate_sparse_directed_graph,
            dijksta_algorithm=dijkstra_path,
            algo_new=lambda g, s, t: bmssp_path(g, s, t, l=l),
            n_nodes=bench_n,
            bellman_ford_algorithm=bellman_ford_path,
            output_file=args.benchmark_output,
        )
        print(f"Benchmark saved to {args.benchmark_output}")
    else:
        print("\nBenchmark skipped (use --force-benchmark to run it when using --source).")


if __name__ == "__main__":
    main()
