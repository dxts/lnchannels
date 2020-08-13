import networkx as nx
from graphx import init_graph, find_route, edge_betweenness
from channel_selection import select_channels
import random


def get_transac_amt():
    # random amt between 0.5 eur and 500 eur, in satoshis
    # return random.randint(5000, 5000000)
    return 1000


def check_profit(graph: nx.MultiGraph, a: int):
    total_transac = 10000
    count = 0

    for _ in range(total_transac):
        source, target = random.randint(
            0, len(graph) - 1), random.randint(0, len(graph) - 1)
        amount = get_transac_amt()

        path = find_route(graph, source, target, amount)
        # print('src: {:d} - tgt: {:d} \t:: {:s}'.format(
        #     source, target, ' --> '.join(map(lambda a: str(a), path))))
        if a in path:
            count += 1

    return count, total_transac


def build_node():
    graph = init_graph(20, 2)

    # for node in graph.nodes:
    #     print('node {:d}\t:: neighbours {:s}'.format(
    #         node, ', '.join(map(lambda e: str(e), graph[node]))))

    node, channels = select_channels(graph, 2, get_transac_amt())

    print('selected channels \t:: {:s}'.format(
        ' ,'.join(map(lambda c: '({:d} - {:d} - {:f})'.format(c[0], c[1], c[2]), channels))))

    count, total = check_profit(graph, node)

    print('edge used {:d} / {:d}'.format(count, total))


if __name__ == '__main__':
    build_node()
