''' Built over the networkx implementation of random graph generators.
`https://networkx.github.io/documentation/networkx-1.9.1/_modules/networkx/generators/random_graphs.html#barabasi_albert_graph`
'''

from typing import List, Tuple
from node import NodeInfo, ChannelInfo
import random


class Node:
    '''Represents node in graph
    '''

    def __init__(self, id: int, node_info: NodeInfo):
        self.id = id
        self.props = NodeInfo.init_random() if node_info is None else node_info
        self.edges: List[Edge] = []


class Edge:
    '''
    Represents edge in graph, contains channel info.
    '''

    def __init__(self,  from_node: Node, to_node: Node, from_to_props: ChannelInfo, to_from_props: ChannelInfo):
        self.from_node = from_node
        self.to_node = to_node

        self.from_to_props = ChannelInfo.init_random(
        ) if from_to_props is None else from_to_props

        self.to_from_props = ChannelInfo.init_random(
        ) if to_from_props is None else to_from_props

    def get_incoming_channelinfo(self, node: int):
        ''' Return channel props for transaction incoming to node.
        '''
        return self.from_to_props if self.to_node == node else self.to_from_props

    def get_neighbour(self, node: int):
        return self.from_node.id if self.from_node.id != node else self.to_node.id


class Graph:

    def __init__(self, n: int = 100):
        ''' Initialize graph with n (default = 100) nodes.
        '''
        self.nodes: List[Node] = []
        for i in range(n):
            self.add_node(i)

    def add_node(self, id: int,
                 node_info: NodeInfo = None):
        ''' Add node to graph, with given or random properties.
        '''
        self.nodes.insert(id, Node(id, node_info))

    def add_edge_from(self, start: int, end: int,
                      outgoing_props: ChannelInfo = None,
                      incoming_props: ChannelInfo = None):
        ''' Add edge from start node to end node, with given or random properties.
        Creates nodes if they don't yet exist.
        Returns Edge from
        '''
        # create nodes if necessary
        if start >= len(self.nodes):
            self.add_node(start)
        if end >= len(self.nodes):
            self.add_node(end)

        # create edges for both directions with respective properties.
        edge = Edge(self.nodes[start],
                    self.nodes[end], outgoing_props, incoming_props)

        # add edges to resp. nodes
        self.nodes[start].edges.append(edge)
        self.nodes[end].edges.append(edge)

        return edge

    def remove_edge_from(self, start: int, end: int):
        ''' Remove edge between two nodes. Removes both directions of edges.
        '''
        self.nodes[start].edges = filter(
            lambda edge: edge.to_node != end, self.nodes[start].edges)
        self.nodes[end].edges = filter(
            lambda edge: edge.from_node != start, self.nodes[end].edges)

    def get_edge_from(self, start: int, end: int) -> Edge:
        ''' Get edge from start node to end node.
        '''
        edge = filter(lambda edge:
                      edge.to_node == end
                      or edge.from_node == end,
                      self.nodes[start].edges)
        return list(edge)[0]

    @staticmethod
    def _random_subset(seq, m, rng):
        """ Return m unique elements from seq.
        This differs from random.sample which can return repeated
        elements if seq holds repeated elements.
        Note: rng is a random.Random or numpy.random.RandomState instance.
        """
        targets = set()
        while len(targets) < m:
            x = rng.choice(seq)
            targets.add(x)
        return targets

    @staticmethod
    def barabasi_albert_graph(n, m, seed=None):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.
        A graph of $n$ nodes is grown by attaching new nodes each with $m$
        edges that are preferentially attached to existing nodes with high degree.
        Parameters
        ----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        Returns
        -------
        G : Graph
        Raises
        ------
        NetworkXError
            If `m` does not satisfy ``1 <= m < n``.
        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
           random networks", Science 286, pp 509-512, 1999.
        """

        if m < 1 or m >= n:
            raise IOError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )

        if seed is None:
            seed = random._inst

        # Add m initial nodes (m0 in barabasi-speak)
        G = Graph(m)
        # Target nodes for new edges
        targets = list(range(m))
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = []
        # Start adding the other n-m nodes. The first node is m.
        source = m
        while source < n:
            # Add edges to m nodes from the source.
            for target in targets:
                G.add_edge_from(source, target)

            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * m)
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = Graph._random_subset(repeated_nodes, m, seed)
            source += 1
        return G

    @staticmethod
    def path_find(g, source: int, target: int, amt: int) -> List[Tuple[int, int]]:
        ''' Uses djikstra to find the path to target.
        Returns seq of edges in reverse order if path found, else None.
        '''
        import heapq

        # path find from transaction target to source
        # so that fees can be accumulated
        temp = source
        source = target
        target = temp
        del temp

        # stores current best distance for a node
        best_weight = {}

        # stores best edge for a given node
        # used to construct the path later
        best_edge = {}

        # priority queue to get next node to explore
        to_visit = []

        # initialize path-finding
        best_weight[source] = 0
        heapq.heappush(to_visit, (0, source))

        target_reached = False
        while not target_reached and to_visit:
            _, current = heapq.heappop(to_visit)

            if current == target:
                target_reached = True
            else:
                current_weight = best_weight[current]

                for edge in g.nodes[current].edges:
                    neighbour = edge.get_neighbour(current)

                    neighbour_to_curr = edge.get_incoming_channelinfo(current)
                    edge_weight = neighbour_to_curr.calc_weight(current_weight)

                    neighbour_weight = current_weight + edge_weight

                    if neighbour not in best_weight or best_weight[neighbour] > neighbour_weight:
                        best_weight[neighbour] = neighbour_weight
                        best_edge[neighbour] = edge
                        heapq.heappush(to_visit, (neighbour_weight, neighbour))

        if not target_reached:
            return None
        else:
            path = []
            current_node = target
            while current_node != source:
                # stores current partial path head
                next_node = best_edge[current_node].get_neighbour(current_node)
                path.append((current_node, next_node))
                current_node = next_node
            return path
