from typing import List, Tuple
import networkx as nx
from graphx import edge_betweenness
from node import NodeInfo, ChannelPolicies
import random


def build_node(g: nx.MultiGraph, new_node: int, n: int, const_amt: int, fee_type: str) -> List:

    selected = []

    # 2 initial edges so routes can pass through new node
    while len(selected) < n:
        max_reward = 0
        best_channel = None

        # try all channels
        for node in g.nodes():
            if node == new_node or node in g[new_node]:
                continue

            # create new channel
            g.add_edge_with_init(new_node, node)
            # calculate max reward
            reward, fee = maximise_fee(
                g, (new_node, node), const_amt, fee_type)

            # select channel if highest reward
            if max_reward <= reward:
                max_reward = reward
                best_channel = (node, fee)

            # reset graph for next channel
            g.remove_edge(new_node, node)

        if best_channel is None:
            print('-- No channel produced any reward -- (channel selection exiting..)')
            return selected

        # add selected channel and redo for next channel
        selected.append((new_node, best_channel[0], best_channel[1]))

        g.add_edge_with_init(new_node, best_channel[0], default=True)
        g.update_fee(new_node, best_channel[0], best_channel[1], fee_type)

    # return node with selected channels
    return selected


def maximise_fee(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int, fee_type: str,
                 fee_low: int = None, fee_high: int = None) -> Tuple[float, int]:

    if fee_low is None:
        if fee_type == 'base_fee':
            # fee is in millisatoshi
            fee_low, fee_high = 1, 1000000
        elif fee_type == 'prop_fee':
            fee_low, fee_high = 1, 1000

    max_reward = 0
    max_fee = fee_low

    # how many intervals to partition into
    divison_parameter = 10

    # anchor step
    if fee_high - fee_low <= 10:
        # try all fee values in current range
        for i in range(0, fee_high - fee_low + 1):
            fee = fee_low + i
            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, edge, const_amt, fee_type, fee)
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
        step = int((fee_high - fee_low) / divison_parameter)
        for i in range(divison_parameter+1):
            fee = i * step + fee_low

            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, edge, const_amt, fee_type, fee)
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
                return maximise_fee(g, edge, const_amt, fee_type, fees[i], fees[i+1])

        return max_reward, max_fee


def total_reward(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int, fee_type: str, fee: int) -> Tuple[int, int]:
    edge_reward = reward(g, edge, const_amt, fee_type, fee)

    other_edges = filter(lambda v: v != edge[1], g[edge[0]].keys())
    remaining_reward = 0
    for v in other_edges:
        remaining_reward += reward(g, (edge[0], v), const_amt, fee_type)

    # debugging
    # print('rew {:>9.5f}  rew\' {:>9.5f}'.format(
    #     edge_reward, remaining_reward))

    return edge_reward, remaining_reward


def reward(g: nx.MultiGraph, edge: Tuple[int, int], const_amt: int, fee_type: str, fee: int = None) -> int:
    # debugging
    # print_flag = fee is not None

    u, v = edge
    if fee is not None:
        g.update_fee(u, v, fee, fee_type)
    else:
        fee = getattr(g.get_policy(u, v), fee_type)

    ebc = edge_betweenness(g, const_amt)[edge]

    # debugging
    # if print_flag:
    #     print('({:>3d}, {:>3d})  fee {:>9d}  ebc {:>9f}  '.format(
    #         u, v, fee, ebc), end='')

    return fee * ebc
