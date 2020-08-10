import networkx as nx
from node import NodeInfo, ChannelInfo
from typing import List


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
    g.add_edge(a, b,
               {'{:d}to{:d}'.format(a, b): ChannelInfo.init_random(),
                '{:d}to{:d}'.format(b, a): ChannelInfo.init_random()})


def _curried_weight_func(g: nx.MultiGraph, amt: int) -> int:
    def _weight_func(a: int, b: int, data: dict):
        current_weight = g.nodes[a]['best_weight']

        new_weight = current_weight
        + data['{:d}to{:d}'.format(b, a)].calc_weight(amt)

        if 'best_weight' not in g.nodes[b] or g.nodes[b]['best_weight'] > new_weight:
            g.nodes[b]['best_weight'] = new_weight

        return new_weight

    return _weight_func


def find_route(g: nx.MultiGraph, src: int, tgt: int, amt: int) -> List:

    # set starting weight to 0
    g.nodes[tgt]['best_weight'] = 0

    # path find from transaction target to source
    # so that fees can be accumulated
    path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
        g, tgt, src, _curried_weight_func(g, amt))

    return reversed(path)


def edge_betweenness(g: nx.MultiGraph):
    print(g.edges())
