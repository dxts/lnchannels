import networkx as nx
from graphx import init_graph, find_route, edge_betweenness
from channel_selection import build_node
import random


def check_profit(node_count: int, node_degree: int, new_channel_count: int, amount: int):
    """
    :param node_count: Number of nodes in graph
    :param node_degree: Albert-Barabasi node degree
    :param new_channel_count: Number of new channels to build for adversary node
    :param transac_amount: Const amount for transactions, in millisatoshi
    """
    total_transac = 1000000

    graph = init_graph(node_count, node_degree)

    # debugging
    for node in graph.nodes:
        print('node {:d}\t:: neighbours {:s}'.format(
            node, ', '.join(map(lambda e: str(e), graph[node]))))

    transactions = []
    for _ in range(total_transac):
        src, tgt = random.sample(graph.nodes(), 2)
        transactions.append((src, tgt))

    # select channels with greedy algo
    new_node, channels = build_node(graph, new_channel_count, amount)

    # debugging
    print('greedy selected channels \t:: {:s}'.format(
        ' ,'.join(map(lambda c: '({:d}, {:d}) - fee {:f}'.format(c[0], c[1], c[2]), channels))))

    _run_transactions(transactions, graph, new_node, amount)

    # select channels randomly
    # clear edges
    graph.remove_edges_from(
        list(map(lambda v: (new_node, v), graph[new_node].keys())))
    # and make new edges, to other nodes
    rand_nodes = random.sample(list(filter(lambda n: n != new_node, graph.nodes().keys())),
                               len(channels))
    channels.clear()
    for target in rand_nodes:
        graph.add_edge_with_init(new_node, target)
        channels.append(
            (new_node, target, graph.get_fee(new_node, target, 'prop_fee')))

    # debugging
    print('randomly selected channels \t:: {:s}'.format(
        ' ,'.join(map(lambda c: '({:d}, {:d}) - fee {:f}'.format(c[0], c[1], c[2]), channels))))

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


def ebc():
    import time
    graph = init_graph(1000, 2)

    start = time.time()
    edge_betweenness(graph, 1000000)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    check_profit(20, 2, 2, 1000000)
