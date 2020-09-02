import io
import networkx as nx
from graphx import init_graph, find_route, edge_betweenness
from channel_selection import build_node
from node import NodeInfo
import random


def check_profit(node_count: int, node_degree: int, new_channel_count: int):
    """
    :param node_count: Number of nodes in graph
    :param node_degree: Albert-Barabasi node degree
    :param new_channel_count: Number of new channels to build for adversary node
    :param transac_amount: Const amount for transactions, in millisatoshi
    """
    total_transac = 100000

    # try medium and macro payments
    trans_amounts = [50000]  # , 1000000]

    graph = init_graph(node_count, node_degree)
    new_node = len(graph)
    graph.add_node(new_node, data=NodeInfo.init_random())

    # make initial channels for new node
    highest_degrees = sorted(
        graph.nodes, key=lambda node: len(graph[node]), reverse=True)
    e1, e2 = highest_degrees[0:2]
    graph.add_edge_with_init(new_node, e1, default=True)
    graph.add_edge_with_init(new_node, e2, default=True)

    # save graph for different runs
    initial_graph = 'graphstore.temp'
    nx.write_gpickle(graph, initial_graph)

    # debugging
    for node, neighbours in graph.adjacency():
        print(f'node {node:>3d} :: neighbours {list(neighbours.keys())}')

    # generate random transactions
    transactions = []
    for _ in range(total_transac):
        src, tgt = random.sample(graph.nodes(), 2)
        transactions.append((src, tgt))

    for amount in trans_amounts:
        print(f'\nRunning transactions with {amount:,d} sat.')

        # -------------------------------------------------------
        # select channels with greedy algo (maximise base_fee)
        graph = nx.read_gpickle(initial_graph)

        # debugging
        print('\ngreedy selected channels (maximised base fee)')

        build_node(graph, new_node, new_channel_count, amount, 'base_fee')

        _run_transactions(transactions, graph, new_node, amount)

        # -------------------------------------------------------
        # select channels with greedy algo (maximise prop_fee)

        # reset graph
        graph = nx.read_gpickle(initial_graph)

        # debugging
        print('\ngreedy selected channels (maximised prop fee)')

        build_node(graph, new_node, new_channel_count, amount, 'prop_fee')

        _run_transactions(transactions, graph, new_node, amount)

        # -------------------------------------------------------
        # select channels randomly

        # reset graph
        graph = nx.read_gpickle(initial_graph)

        # debugging
        print('\nrandomly selected channels(default fees)')

        potential_edges = set(range(0, new_node))
        for edge in graph[new_node]:
            potential_edges.remove(edge)

        # and make new edges, to other nodes
        rand_neighbours = random.sample(potential_edges, new_channel_count)
        for target in rand_neighbours:
            graph.add_edge_with_init(new_node, target, default=True)

        _run_transactions(transactions, graph, new_node, amount)

    return


def _run_transactions(transactions: list, graph: nx.MultiGraph, new_node: int, amount: int):

    info = {'profit': {}, 'success': {}, 'failure': {}, 'failure_rev': {},
            'total_count': len(transactions), 'total_success': 0, 'no_path': 0}

    for neighbour in graph[new_node]:
        key1 = (new_node, neighbour)
        key2 = (neighbour, new_node)

        info['profit'][key1] = 0
        info['profit'][key2] = 0
        info['success'][key1] = 0
        info['success'][key2] = 0
        info['failure'][key1] = 0
        info['failure'][key2] = 0
        info['failure_rev'][key1] = 0
        info['failure_rev'][key2] = 0

        policy = graph.get_policy(new_node, neighbour)
        print(
            f'\t({new_node}, {neighbour}) - policy {policy.base_fee} msat, {policy.prop_fee}')

    # find route and run trans
    for source, target in transactions:
        # try transaction
        try:
            _transact(graph, source, target, amount, new_node, info)
        # if failed due to channel imbalance, try transaction in rev direction
        except Exception:
            try:
                _transact(graph, target, source, amount, new_node, info)
                info['total_count'] += 1
                # if passed, record info
                info['failure_rev'][source, target] += 1
            except Exception:
                continue

    # print total number of failed transactions
    print('\n\tfailed transactions - {:.2f}% \tno path - {:.2f}%'.format(
        (info['total_count'] - info['total_success']) *
        100 / info['total_count'],
        info['no_path'] * 100 / info['total_count']))

    node_profit = 0
    # print fee collection info for each channel
    print('\tchannel \tprofit sat \tused in %% of total \tfailed %% of incoming \t %% of failed that pass in reverse dir. \tpolicy ')
    for key in info['profit'].keys():
        succeded = info['success'][key]
        failed = info['failure'][key]
        print('\t{:s} \t{:14,.0f} \t\t{:5.2f} \t\t\t{:5.2f} \t\t\t{:5.2f} \t\t\t{}'.format(
            str(key),
            info['profit'][key],
            succeded * 100 / info['total_count'],
            0 if failed == 0 else failed * 100 / (succeded + failed),
            0 if failed == 0 else info['failure_rev'][key] * 100 / failed,
            graph.get_policy(key[0], key[1])
        ))

        if key[0] == new_node:
            node_profit += info['profit'][key]

    print(f'\ttotal profit for node - {node_profit:,.0f} sat')

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
            if node == new_node or neighbour == new_node:
                info['failure'][node, neighbour] += 1
                raise Exception('channel imbalance')
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

        # record profit for adversary
        if node == new_node or neighbour == new_node:
            info['success'][node, neighbour] += 1
            info['profit'][node, neighbour] += fees[node]


def ebc():
    import time
    graph = init_graph(1000, 2)

    start = time.time()
    edge_betweenness(graph, 1000000)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    check_profit(20, 3, 2)
