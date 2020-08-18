import networkx as nx
from graphx import init_graph, find_route, edge_betweenness
from channel_selection import select_channels
import random


def check_profit(node_count: int, node_degree: int, new_channel_count: int, amount: int):
    """
    :param node_count: Number of nodes in graph
    :param node_degree: Albert-Barabasi node degree
    :param new_channel_count: Number of new channels to build for adversary node
    :param transac_amount: Const amount for transactions, in millisatoshi
    """
    total_transac = 1000000

    graph, new_node = build_node(
        node_count, node_degree, new_channel_count, amount)

    # debugging
    for node in graph.nodes:
        print('node {:d}\t:: neighbours {:s}'.format(
            node, ', '.join(map(lambda e: str(e), graph[node]))))

    # filter out new_node, as it will not be a source or target in a transaction
    nodes = list(filter(lambda n: n != new_node, graph.nodes().keys()))

    transactions = []
    for _ in range(total_transac):
        src, tgt = random.sample(nodes, 2)
        transactions.append((src, tgt))

    _run_transactions(transactions, graph, new_node, amount)

    # clear edges
    num_edges = len(graph[new_node])
    graph.remove_edges_from(
        list(map(lambda v: (new_node, v), graph[new_node].keys())))
    # and make new edges
    rand_nodes = random.sample(nodes, num_edges)
    for target in rand_nodes:
        graph.add_edge_with_init(new_node, target)

    _run_transactions(transactions, graph, new_node, amount)

    return


def _run_transactions(transactions: list, graph: nx.MultiGraph, new_node: int, amount: int):
    profit = 0
    count = 0
    for source, target in transactions:
        path, weights = find_route(graph, source, target, amount)
        if new_node in path:
            count += 1
            profit += weights[new_node] - \
                weights[path[path.index(new_node) + 1]]

    print('profit by node \t{:f}, used in {:f}% transactions'.format(
        profit, count * 100 / len(transactions)))


def build_node(n: int, d: int, channel_count: int, amount: int):
    graph = init_graph(n, d)

    node, channels = select_channels(graph, channel_count, amount)

    # debugging
    print('selected channels \t:: {:s}'.format(
        ' ,'.join(map(lambda c: '({:d}, {:d}) - fee {:f})'.format(c[0], c[1], c[2]), channels))))

    return graph, node


def ebc():
    import time
    graph = init_graph(1000, 2)

    start = time.time()
    edge_betweenness(graph, 1000000)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    check_profit(50, 2, 2, 1000000)
