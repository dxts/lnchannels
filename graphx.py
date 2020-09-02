import networkx as nx
from node import NodeInfo, ChannelPolicies
from typing import List, Dict, Tuple, Callable
import random


def init_graph(n: int, d: int, seed=None) -> nx.Graph:
    """
    Initialize graph with n nodes, and d degree for Albert-Barabasi.
    :param n:
    :param d:
    :return:
    """
    import types

    # init barabasi-albert graph
    g: nx.Graph = nx.generators.random_graphs.barabasi_albert_graph(
        n, d, seed)

    # add node data to each node
    for _, data in g.nodes(data=True):
        data['data'] = NodeInfo.init_random()

    # add edge data to each edge
    for a, b, data in g.edges(data=True):
        _init_edge(g, a, b)

    # add custom add_edge method to graph
    # g.add_edge_with_init = types.MethodType(_add_edge, g)
    # g.update_fee = types.MethodType(_update_fee, g)
    # g.get_policy = types.MethodType(_get_policy, g)

    nx.Graph.add_edge_with_init = _add_edge
    nx.Graph.update_fee = _update_fee
    nx.Graph.get_policy = _get_policy

    return g


def _init_edge(g: nx.Graph,  a: int, b: int, default: bool = False):
    a_to_b = '{:d}to{:d}'.format(a, b)
    b_to_a = '{:d}to{:d}'.format(b, a)

    if default:
        g[a][b][a_to_b] = ChannelPolicies.init_default()
    else:
        g[a][b][a_to_b] = ChannelPolicies.init_random()

    g.edges[a, b][b_to_a] = ChannelPolicies.init_random()

    capacity = int(5000000 * random.random())
    g[a][b]['capacity'] = capacity
    g[a][b][a_to_b].balance = int(capacity / 2)
    g[a][b][b_to_a].balance = int(capacity / 2)


def _add_edge(self, a: int, b: int, default: bool = False):
    self.add_edge(a, b)
    _init_edge(self, a, b, default)


def _update_fee(self, a: int, b: int, fee: int, type: str):
    if type == 'base_fee':
        self[a][b]['{:d}to{:d}'.format(a, b)].base_fee = fee
    elif type == 'prop_fee':
        self[a][b]['{:d}to{:d}'.format(a, b)].prop_fee = fee


def _get_policy(self, a: int, b: int) -> float:
    return self[a][b]['{:d}to{:d}'.format(a, b)]


def find_route(g: nx.Graph, src: int, tgt: int, amt: int) -> Tuple[List, Dict, Dict]:

    # set starting weight to 0
    best_weight = {}
    best_weight[tgt] = amt

    node_profit = {}
    node_profit[tgt] = 0

    # make weight function for use in dijkstra
    def _weight_func(a: int, b: int, data: dict) -> float:
        current_weight = best_weight[a]

        edge_data = data['{:d}to{:d}'.format(b, a)]

        if current_weight > data['capacity']:
            best_weight[b] = float('inf')
            return float('inf')
        else:
            fee = edge_data.calc_fee(current_weight)
            risk_factor = edge_data.calc_risk(current_weight)

            new_weight = current_weight + fee + risk_factor
            if b not in best_weight or best_weight[b] > new_weight:
                best_weight[b] = new_weight
                node_profit[b] = fee

            return fee + risk_factor

    # path find from transaction target to source
    # so that fees can be accumulated
    try:
        path = nx.dijkstra_path(g, tgt, src, _weight_func)

        # if path distance is inf, no path found
        if best_weight[src] == float('inf'):
            return None, None, None

    except nx.NetworkXNoPath as err:
        print(err)
        return None, None, None

    return list(reversed(path)), best_weight, node_profit


def edge_betweenness(g: nx.Graph, amt: int, normalized=True, count=None) -> int:
    """ Adapted from networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/betweenness.html#edge_betweenness_centrality
    """
    def weight_function(u: int, v: int, current_weight: int) -> int:
        edge_data = g[u][v]['{:d}to{:d}'.format(u, v)]
        if current_weight > g[u][v]['capacity']:
            return float('inf')
        else:
            return edge_data.calc_fee(amt) + edge_data.calc_risk(amt)

    # b[e]=0 for e in G.edges()
    betweenness = dict.fromkeys(g.edges, 0.0)
    # add reverse edges, since the graph is undirected
    betweenness.update(dict.fromkeys(
        map(lambda e: (e[1], e[0]), g.edges()), 0.0))

    for node in g.nodes():
        # all shortest paths to one node
        S, P, sigma = _single_target_dijkstra_path(
            g, node, weight_function, amt)

        # debugging
        if count is not None:
            for n in S:
                _print_paths(n, P, count)

        # count and sum up betweeness for each edge
        betweenness = _accumulate_edge(betweenness, S, P, sigma)

    # debugging
    if count is not None:
        print('{:d} out of {:d} edges'.format(count[2], count[3]))

    # rescale value
    if normalized:
        n = len(g)
        if n > 1:
            scale = 1 / (n * (n - 1))
            for v in betweenness:
                betweenness[v] *= scale

    return betweenness


def _single_target_dijkstra_path(g: nx.Graph, s: int, weight: Callable[[int, int], int], amt: int):
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
    push(Q, (amt, next(c), s, s))
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


def _accumulate_edge(betweenness: Dict, S: List, P: Dict, sigma: Dict):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()

        # debugging
        # _print_paths(w, P)

        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            betweenness[(w, v)] += c
            delta[v] += c

    return betweenness


def _print_paths(src: int, P: Dict, edge: List = None):
    paths = _find_paths(src, P)
    for p in paths:
        if edge is not None:
            edge[3] += 1
            if str(edge[0]) not in p:
                continue
            else:
                edge[2] += 1
        print(p)


def _find_paths(src: int, P: Dict) -> List[str]:
    multiple_paths = []
    if len(P[src]) != 0:
        for next in P[src]:
            paths = _find_paths(next, P)
            multiple_paths.extend(map(lambda p: str(src)+' '+p, paths))
    else:
        multiple_paths.append(str(src))
    return multiple_paths
