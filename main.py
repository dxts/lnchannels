import io
import random
from functools import reduce
from collections import defaultdict
from shutil import copyfileobj
from datetime import datetime

import networkx as nx
from graphx import init_graph, find_route, edge_betweenness
from channel_selection import build_node
from node import NodeInfo

# setup data logging
log = io.StringIO()


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var('first_call', True)
def setup_graph(m=0, n=0):
    """ Sets up graph with last node as adversary with two existing channels.
    Caches graph for repeated use.
    """
    if not setup_graph.first_call:
        return nx.read_gpickle('graphstore.temp')
    else:
        setup_graph.first_call = False

        graph = init_graph(m, n)
        new_node = len(graph)
        graph.add_node(new_node, data=NodeInfo.init_random())

        # make initial channels for new node
        highest_degrees = sorted(
            graph.nodes, key=lambda node: len(graph[node]), reverse=True)
        e1, e2 = highest_degrees[0:2]
        graph.add_edge_with_init(new_node, e1, default=True)
        graph.add_edge_with_init(new_node, e2, default=True)

        # save graph for different runs
        nx.write_gpickle(graph, 'graphstore.temp')

        # debugging
        log.write('''## Graph used
|node |neighbours|
| --- | --- |
''')
        log.write('\n'.join(f'|{node:>4d} |{list(neighbours.keys())} |'
                            for node, neighbours in graph.adjacency()) + '\n --- \n')

        return graph


def check_profit(node_count: int, node_degree: int, new_channel_count: int):
    """
    :param node_count: Number of nodes in graph
    :param node_degree: Albert-Barabasi node degree
    :param new_channel_count: Number of new channels to build for adversary node
    :param transac_amount: Const amount for transactions, in millisatoshi
    """

    graph = setup_graph(node_count, node_degree)
    new_node = len(graph) - 1

    # avg 200 trans as source per node
    total_transac = node_count * 250
    # try medium and macro payments
    trans_amounts = [10000, 100000]

    # generate random transactions
    transactions = []
    for _ in range(total_transac):
        src, tgt = random.sample(graph.nodes(), 2)
        transactions.append((src, tgt))

    for amount in trans_amounts:
        log.write(f'## Running transactions of {amount:,d} sat\n')

        # select channels with greedy algo (maximise base_fee)
        # -------------------------------------------------------
        graph = setup_graph()

        # debugging
        log.write('---\n### Greedily selected channels (maximised base fee)\n')

        build_node(graph, new_node, new_channel_count, amount, 'base_fee')

        _run_transactions(transactions, graph, new_node, amount)

        # select channels with greedy algo (maximise prop_fee)
        # -------------------------------------------------------

        # reset graph
        graph = setup_graph()

        # debugging
        log.write('---\n### Greedily selected channels (maximised prop fee)\n')

        build_node(graph, new_node, new_channel_count, amount, 'prop_fee')

        _run_transactions(transactions, graph, new_node, amount)

        # select channels randomly (run twice)
        # -------------------------------------------------------
        for _ in range(2):
            # reset graph
            graph = setup_graph()

            # debugging
            log.write('---\n### Randomly selected channels (default fees)\n')

            # don't make edges with existing neighbours
            potential_edges = set(range(0, new_node))
            for edge in graph[new_node]:
                potential_edges.remove(edge)

            # and make new edges, to other nodes
            rand_neighbours = random.sample(potential_edges, new_channel_count)
            for target in rand_neighbours:
                graph.add_edge_with_init(new_node, target, default=True)

            _run_transactions(transactions, graph, new_node, amount)

    # write info to file
    fd = open(f'logs/{datetime.now()}.lnsim.md', 'w+')
    log.seek(0)
    copyfileobj(log, fd)
    fd.close()

    return


def _run_transactions(transactions: list, graph: nx.MultiGraph, new_node: int, amount: int):

    adversary_channels = list(
        map(lambda x: (new_node, x), graph[new_node].keys()))
    peer_channels = list(map(lambda x: (x, new_node), graph[new_node].keys()))

    info = {'success': defaultdict(lambda: 0), 'failure': defaultdict(lambda: 0), 'failure_rev': defaultdict(lambda: 0),
            'profit': defaultdict(lambda: 0),
            'total_trans': len(transactions),
            'total_success': 0, 'no_path': 0, 'channel_imbalance': 0}

    log.write('\n'.join(
        f'* ({new_node}, {neighbour}) - {graph.get_policy(new_node, neighbour)}' for neighbour in graph[new_node]))

    # find route and run trans
    for source, target in transactions:
        # try transaction
        try:
            _transact(graph, source, target, amount, new_node, info)
        # if failed due to channel imbalance, try transaction in rev direction
        except Exception as ex1:
            if str(ex1) == 'Adversary channel imbalance':
                try:
                    _transact(graph, target, source, amount, new_node, info)
                    info['total_trans'] += 1
                    # if passed, record info
                    info['failure_rev'][source, target] += 1
                except Exception as ex2:
                    if str(ex2) == 'Adversary channel imbalance':
                        continue
                    else:
                        print('Error in _transact')
                        exit(0)
            else:
                print('Error in _transact')
                exit(0)

    # total profit earned by all our channels
    node_profit = reduce(lambda sum, k: sum +
                         info['profit'][k], adversary_channels, 0)

    # print total number of failed transactions
    # print fee collection info for each channel
    log.write(f'''\n
**Transactions**
* total: {info['total_trans']:,d}
* failed: {info['total_trans'] - info['total_success']:,d}
    * no path - {info['no_path']:,d}
    * channel imbalance - {info['channel_imbalance']:,d}

**Profit** for node: {node_profit:,.0f} sat

| channel | profit (sat) | used in _ trans | failed _ trans | _ failed trans that <br/> passed in reverse dir. | policy |
| --- | --- | --- | --- | --- | --- |
''')

    info_str = ''
    for i in range(0, len(peer_channels)):
        key1 = adversary_channels[i]
        key2 = peer_channels[i]
        info_str += f"|**{key1}**| **{info['profit'][key1]:,.0f}**| **{info['success'][key1]}**| **{info['failure'][key1]}** |**{info['failure_rev'][key1]}** |**{graph.get_policy(*key1)}**|\n"
        info_str += f"|{key2} |{info['profit'][key2]:,.0f} |{info['success'][key2]} |{info['failure'][key2]} |{info['failure_rev'][key2]} |{graph.get_policy(*key2)} |\n"
    log.write(info_str)

    return


def _transact(graph: nx.MultiGraph, src: int, tgt: int, amount: int, new_node: int, info: dict):
    from more_itertools import pairwise

    path, weights, fees = find_route(graph, src, tgt, amount)

    # no path found
    if path is None:
        info['no_path'] += 1
        return

    for (node, neighbour) in pairwise(path):
        channel_trans_amt = weights[neighbour]
        channel_policy = graph.get_policy(node, neighbour)

        # channel cannot handle transaction
        if channel_policy.balance < channel_trans_amt:
            info['channel_imbalance'] += 1
            if node == new_node or neighbour == new_node:
                info['failure'][node, neighbour] += 1
                raise Exception('Adversary channel imbalance')
            return

    # record successful transaction
    info['total_success'] += 1

    total_amt = 0
    # move amount between nodes in path
    for (node, neighbour) in pairwise(path):
        if neighbour is None:
            # remove total amount from src
            graph.nodes[path[0]]['data'].capacity -= total_amt
            # add amount to tgt
            graph.nodes[node]['data'].capacity += weights[node]
            break

        channel_trans_amt = weights[neighbour]
        channel_policy = graph.get_policy(node, neighbour)

        # pay channel fees to node
        graph.nodes[node]['data'].capacity += fees[node]
        total_amt += fees[node]

        # shift channel balance
        channel_policy.balance -= channel_trans_amt
        graph.get_policy(neighbour, node).balance += channel_trans_amt

        # record profit for adversary
        if node == new_node or neighbour == new_node:
            info['success'][node, neighbour] += 1
            info['profit'][node, neighbour] += fees[node]


if __name__ == '__main__':
    check_profit(100, 3, 3)
