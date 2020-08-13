from typing import List, Tuple
import networkx as nx
from graphx import add_edge, update_fee, edge_betweenness
from node import NodeInfo, ChannelInfo
import random


def select_channels(g: nx.MultiGraph, n: int, const_amt: int) -> Tuple[int, List]:
    new_node = len(g)
    g.add_node(new_node, data=NodeInfo.init_random())

    # make initial channels for new node
    highest_degrees = sorted(
        g.nodes, key=lambda node: len(g[node]), reverse=True)
    add_edge(g, new_node, highest_degrees[0])
    add_edge(g, new_node, highest_degrees[1])

    for node in g.nodes:
        print('node {:d}\t:: neighbours {:s}'.format(
            node, ', '.join(map(lambda e: str(e), g[node]))))

    selected_edges = []

    while len(selected_edges) < n:
        max_reward = 0
        selected_node = None
        maximised_fee: ChannelInfo = None

        # try all channels
        for node in g.nodes:
            if node == new_node:
                continue

            # create new channel
            add_edge(g, new_node, node)
            # calculate max reward
            reward, fee = maximise_fee(
                g, (new_node, node), const_amt)

            # print('({:d}, {:d}) fee {:f} \treward {:f} \t'.format(
            #     new_node, node, fee, reward))

            # select channel if highest reward
            if max_reward <= reward:
                max_reward = reward
                selected_node = node
                maximised_fee = fee

            # reset graph for next channel
            g.remove_edge(new_node, node)

        print('added channel ({:d}, {:d}, {:f})'.format(
            new_node, selected_node, maximised_fee))
        # add selected channel and redo for next channel
        selected_edges.append((new_node, selected_node, maximised_fee))
        add_edge(g, new_node, selected_node)
        update_fee(g, new_node, selected_node, maximised_fee)

    # return node with selected channels
    return new_node, selected_edges


def maximise_fee(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int,
                 fee_low=0.00001, fee_high=75) -> Tuple[int, int]:

    max_reward = 0
    max_fee = fee_low

    # how many intervals to partition into
    divison_parameter = 20

    # anchor step
    if fee_high - fee_low <= divison_parameter:
        # try all fee values in current range
        for fee in range(fee_low, fee_high+1):
            # set channel fee to $fee$
            # TODO
            update_fee(g, edge[0], edge[1], fee)

            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, edge, fee, const_amt)
            reward = edge_reward + remaining_reward
            if reward >= max_reward:
                max_reward = reward
                max_fee = fee

        return max_reward, max_fee
    # recursive step
    else:
        fees = [None]*divison_parameter
        rewards = [None]*divison_parameter

        for i in range(divison_parameter):
            fee = i * (fee_high - fee_low) / divison_parameter + fee_low

            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, edge, fee, const_amt)
            reward = edge_reward + remaining_reward
            if reward >= max_reward:
                max_reward = reward
                max_fee = fee

            fees[i] = fee
            rewards[i] = (edge_reward, remaining_reward)

            # print('({:d}, {:d}) fee {:f} \treward {:f} \t'.format(
            #     from_node, to_node, fee, reward))

        for i in range(divison_parameter-1):
            # max possible reward
            possible_reward = rewards[i][0] * \
                fees[i+1] / fees[i] + rewards[i+1][1]

            # print('({:d}, {:d}) possible reward ({:f}, {:f}) {:f}'.format(
            #     from_node, to_node, fees[i], fees[i+1], possible_reward))

            if possible_reward > max_reward:
                # print('--------- recursin into interval ---------')
                fee_low = int(fees[i])
                fee_high = int(fees[i+1])
                return maximise_fee(g, edge, const_amt, fee_low, fee_high)

        return max_reward, max_fee


def total_reward(g: nx.MultiGraph, edge: Tuple[int, int], fee: int, const_amt: int) -> Tuple[int, int]:
    edge_reward = reward(g, edge, const_amt, fee)

    other_edges = filter(lambda v: v != edge[1], g[edge[0]].keys())
    remaining_reward = 0
    for v in other_edges:
        remaining_reward += reward(g, (edge[0], v), const_amt)

    return (edge_reward, remaining_reward)


def reward(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int, fee: int = None) -> int:
    u, v = edge
    flag = False
    if fee is not None:
        flag = True
        update_fee(g, u, v, fee)
    else:
        fee = g[u][v]['{:d}to{:d}'.format(u, v)].base_fee

    ebc = edge_betweenness(g, const_amt, u)

    if flag:
        print('({:d}, {:d}) fee {:f} \tebc {:f}'.format(
            u, v, fee, ebc[edge]))

    return fee * ebc[edge]
