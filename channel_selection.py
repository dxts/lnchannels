import functools
import networkx as nx
from graphx import add_edge
from node import NodeInfo, ChannelInfo


def channel_selection(g: nx.MultiGraph, n: int, const_amt: int):
    new_node = g.add_node(g.order(), data=NodeInfo.init_random())

    selected_edges = []

    while len(selected_edges) < n:
        max_reward = 0
        selected_node = None
        maximised_fee: ChannelInfo = None

        # try all channels
        for node in g.nodes:
            # create new channel
            add_edge(g, new_node, node)
            edge = g.add_edge_from(new_node, node)
            # calculate max reward
            reward = maximise_fee(
                g, new_node, edge, const_amt)

            # select channel if highest reward
            if max_reward <= reward:
                max_reward = reward
                selected_node = node
                maximised_fee = edge.from_to_props

            # reset graph for next channel
            g.remove_edge_from(new_node, node)

        # add selected channel and redo for next channel
        g.add_edge_from(new_node, selected_node, maximised_fee)

    # return node with selected channels
    return new_node


def maximise_fee(g: Graph, node: Node, channel: Edge, const_amt: int,
                 fee_low=1, fee_high=ChannelInfo.default_channel_cost) -> int:

    max_reward = 0
    max_fee = fee_low

    divison_parameter = 1000

    # anchor step
    if fee_high - fee_low <= divison_parameter:
        # try all fee values in current range
        for fee in range(fee_low, fee_high+1):
            # set channel fee to $fee$
            # TODO
            channel.from_to_props.base_fee = fee

            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, node, channel, const_amt)
            reward = edge_reward + remaining_reward
            if reward >= max_reward:
                max_reward = reward
                max_fee = fee

        # set channel fee to $max_fee$
        # TODO
        channel.from_to_props.base_fee = max_fee
        return max_reward
    # recursive step
    else:
        fees = []
        rewards = []

        for i in range(1, divison_parameter + 1):
            fee = i * (fee_high - fee_low) / divison_parameter + fee_low
            # set channel fee to $fee$
            # TODO
            channel.from_to_props.base_fee = fee

            # calculate reward and select max
            edge_reward, remaining_reward = total_reward(
                g, node, channel, const_amt)
            reward = edge_reward + remaining_reward
            if reward >= max_reward:
                max_reward = reward
                max_fee = fee

            fees.insert(i, fee)
            rewards.insert(i, (edge_reward, remaining_reward))

        # set channel fee to $max_fee$
        # TODO
        channel.from_to_props.base_fee = max_fee

        for i in range(1, divison_parameter + 1):
            # max possible reward
            max_possible_reward = rewards[i][0] * \
                fees[i+1] / fees[i] + rewards[i+1][1]

            if max_possible_reward > max_reward:
                fee_low = fees[i]
                fee_high = fees[i+1]
                return maximise_fee(g, node, channel, const_amt, fee_low, fee_high)

        return max_reward


def total_reward(g: Graph, node: Node, channel: Edge, const_amt: int):
    edge_reward = reward(g, node, channel, const_amt)

    other_edges = filter(lambda edge: edge != channel, node.edges)
    remaining_reward = functools.reduce(
        lambda acc, channel: acc + reward(g, node, channel, const_amt), other_edges, 0)

    return (edge_reward, remaining_reward)


def reward(g: Graph, node: Node, channel: Edge, const_amt: int):
    return 0
