import networkx as nx
from node import NodeInfo, ChannelInfo
from typing import List, Dict, Callable
import random


def init_graph(n: int, d: int, seed=None) -> nx.MultiGraph:
    """
    Initialize graph with n nodes, and d degree for Albert-Barabasi.
    :param n:
    :param d:
    :return:
    """

    # init barabasi-albert graph
    g = nx.generators.random_graphs.barabasi_albert_graph(n, d, seed)

    # add node data to each node
    for _, data in g.nodes(data=True):
        data['data'] = NodeInfo.init_random()

    # add edge data to each edge
    for a, b, data in g.edges(data=True):
        data['{:d}to{:d}'.format(a, b)] = ChannelInfo.init_random()
        data['{:d}to{:d}'.format(b, a)] = ChannelInfo.init_random()

    return g


def add_edge(g: nx.MultiGraph, a: int, b: int):
    g.add_edge(a, b)
    g.edges[a, b]['{:d}to{:d}'.format(a, b)] = ChannelInfo.init_random()
    g.edges[a, b]['{:d}to{:d}'.format(b, a)] = ChannelInfo.init_random()


def update_fee(g: nx.MultiGraph, a: int, b: int, fee: int):
    g[a][b]['{:d}to{:d}'.format(a, b)].base_fee = fee


def find_route(g: nx.MultiGraph, src: int, tgt: int, amt: int) -> List:

    # set starting weight to 0
    best_weight = {}
    best_weight[tgt] = 0

    # make weight function for use in dijkstra
    def _weight_func(a: int, b: int, data: dict):
        current_weight = best_weight[a]

        edge_data = data['{:d}to{:d}'.format(b, a)]

        if current_weight > edge_data.capacity:
            edge_weight = float('inf')
        else:
            edge_weight = edge_data.calc_weight(current_weight)
            new_weight = current_weight + edge_weight
            if b not in best_weight or best_weight[b] > new_weight:
                best_weight[b] = new_weight

        return edge_weight

    # path find from transaction target to source
    # so that fees can be accumulated
    path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
        g, tgt, src, _weight_func)

    return reversed(path)


def edge_betweenness(g: nx.MultiGraph, amt: int, normalized=True, exclude=None) -> int:
    """ Adapted from networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/betweenness.html#edge_betweenness_centrality
    """
    def weight_function(u: int, v: int, current_weight: int) -> int:
        edge_data = g[u][v]['{:d}to{:d}'.format(u, v)]
        if current_weight > edge_data.capacity:
            return float('inf')
        else:
            return edge_data.calc_weight(amt)

    # b[e]=0 for e in G.edges()
    betweenness = dict.fromkeys(g.edges, 0.0)
    betweenness.update(dict.fromkeys(
        map(lambda e: (e[1], e[0]), g.edges()), 0.0))

    # exclude paths that end at the new node
    for node in filter(lambda n: n != exclude, g.nodes().keys()):
        S, P, sigma = _single_target_dijkstra_path(g, node, weight_function)

        betweenness = _accumulate_edge(betweenness, S, P, sigma, exclude)

    # rescale value
    if normalized:
        n = len(g)
        if n <= 1:
            scale = None
        else:
            scale = 1 / (n * (n - 1))
    else:
        # rescale by 2 for undirected graphs
        scale = 1

    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness


def _single_target_dijkstra_path(g: nx.MultiGraph, s: int, weight: Callable[[int, int], int]):
    import heapq
    from itertools import count
    S = []
    P = {}
    for v in g:
        P[v] = []
    sigma = dict.fromkeys(g, 0.0)    # sigma[v]=0 for v in G
    visited = set()
    sigma[s] = 1.0
    push = heapq.heappush
    pop = heapq.heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in visited:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        visited.add(v)
        for w, _ in g[v].items():

            vw_dist = dist + weight(w, v, dist)
            if w not in visited and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma


def _accumulate_edge(betweenness: Dict, S: List, P: Dict, sigma: Dict, exclude_from: int):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()

        # exclude paths that start from the new node
        if w == exclude_from:
            continue

        # print_paths(w, P)

        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            betweenness[(w, v)] += c
            delta[v] += c

    return betweenness


def print_paths(src: int, P: Dict):
    paths = find_paths(src, P)
    for p in paths:
        print(p)


def find_paths(src: int, P: Dict) -> List[str]:
    multiple_paths = []
    if len(P[src]) != 0:
        for next in P[src]:
            paths = find_paths(next, P)
            multiple_paths.extend(map(lambda p: str(src)+' '+p, paths))
    else:
        multiple_paths.append(str(src))
    return multiple_paths


def ebc_manual_test(g: nx.MultiGraph = None, amt: int = 1000, exclude: int = None):
    if g is None:
        g = nx.read_gpickle('./graphstore')

    # for node in g.nodes:
    #     print('node {:d}\t:: neighbours {:s}'.format(
    #         node, ', '.join(map(lambda e: str(e), g[node]))))

    betweenness = edge_betweenness(g, amt, exclude)

    for v in betweenness:
        print('ebc ({:d}, {:d}) = {:f}'.format(v[0], v[1], betweenness[v]))

    return betweenness
