from graphx import init_graph, find_route, edge_betweenness
import random


# for node in graph.nodes:
#     print('node {:d} with {:d} incoming edges and {:d} outgoing edges'.format(
#         node.id, len(node.incoming), len(node.outgoing)))

def get_transac_amt():
    # random amt between 0.5 eur and 500 eur, in satoshis
    return random.randint(5000, 5000000)


def transact():
    graph = init_graph(100, 2)

    source, target = random.sample(graph.nodes, 2)
    amount = get_transac_amt()

    path = find_route(graph, source, target, amount)

    # for node in graph.nodes:
    #     print('{:d} : {:s}'.format(node, ', '.join(
    #         map(lambda e: str(e), graph.edges(node)))))

    print('src: {:d} - tgt: {:d} \t:: {:s}'.format(
        source, target, ' --> '.join(map(lambda a: str(a), path))))


# for i in range(100):
#     transact()

graph = init_graph(100, 2)
edge_betweenness(graph)
