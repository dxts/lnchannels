from typing import List, Tuple
import networkx as nx
from graphx import edge_betweenness
from node import NodeInfo, ChannelInfo
import random


def build_node(g: nx.MultiGraph, n: int, const_amt: int) -> Tuple[int, List]:
    new_node = len(g)
    g.add_node(new_node, data=NodeInfo.init_random())

    # make initial channels for new node
    highest_degrees = sorted(
        g.nodes, key=lambda node: len(g[node]), reverse=True)
    e1, e2 = highest_degrees[0:2]
    g.add_edge_with_init(new_node, e1)
    g.add_edge_with_init(new_node, e2)

    selected_edges = []
    selected_edges.append((new_node, e1, g.get_fee(new_node, e1, 'prop_fee')))
    selected_edges.append((new_node, e2, g.get_fee(new_node, e2, 'prop_fee')))

    # debugging
    for node in g.nodes:
        print('node {:d}\t:: neighbours {:s}'.format(
            node, ', '.join(map(lambda e: str(e), g[node]))))

    # 2 initial edges so routes can pass through new node
    while len(selected_edges) < n + 2:
        max_reward = 0
        selected_channel = None

        # try all channels
        for node in g.nodes():
            if node == new_node or node in g[new_node]:
                continue

            # create new channel
            g.add_edge_with_init(new_node, node)
            # calculate max reward
            reward, fee = maximise_fee(
                g, (new_node, node), const_amt)

            # select channel if highest reward
            if max_reward <= reward:
                max_reward = reward
                selected_channel = (node, fee)

            # reset graph for next channel
            g.remove_edge(new_node, node)

        # add selected channel and redo for next channel
        selected_edges.append(
            (new_node, selected_channel[0], selected_channel[1]))

        g.add_edge_with_init(new_node, selected_channel[0])
        g.update_fee(
            new_node, selected_channel[0], selected_channel[1], 'prop_fee')

    # return node with selected channels
    return new_node, selected_edges


def maximise_fee(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int,
                 fee_low: float = 1, fee_high: float = 10000) -> Tuple[float, float]:

    max_reward = 0
    max_fee = fee_low

    # how many intervals to partition into
    divison_parameter = 10

    # anchor step
    if fee_high - fee_low <= 10:
        # try all fee values in current range
        step = (fee_high - fee_low) / divison_parameter
        for fee in [i * step + fee_low for i in range(0, 10)]:
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
        fees = [None]*(divison_parameter+1)
        rewards = [None]*(divison_parameter+1)

        # include fee low and fee high
        for i in range(divison_parameter+1):
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

        for i in range(divison_parameter):
            # max possible reward
            possible_reward = rewards[i][0] * \
                (fees[i+1] / fees[i]) + rewards[i+1][1]

            if possible_reward > max_reward:
                return maximise_fee(g, edge, const_amt, fees[i], fees[i+1])

        return max_reward, max_fee


def total_reward(g: nx.MultiGraph, edge: Tuple[int, int], fee: int, const_amt: int) -> Tuple[int, int]:
    edge_reward = reward(g, edge, const_amt, fee)

    other_edges = filter(lambda v: v != edge[1], g[edge[0]].keys())
    remaining_reward = 0
    for v in other_edges:
        remaining_reward += reward(g, (edge[0], v), const_amt)

    # debugging
    print('rew {:f} \trew\' {:f}'.format(
        edge_reward, remaining_reward))

    return edge_reward, remaining_reward


def reward(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int, fee: int = None) -> int:
    # debugging
    print_flag = fee is not None

    u, v = edge
    if fee is not None:
        g.update_fee(u, v, fee, 'prop_fee')
    else:
        fee = g.get_fee(u, v, 'prop_fee')

    ebc = edge_betweenness(g, const_amt)

    # debugging
    if print_flag:
        print('({:d}, {:d}) \tfee {:f} \tebc {:f} \t'.format(
            u, v, fee, ebc[edge]), end='')

    return fee * ebc[edge]
