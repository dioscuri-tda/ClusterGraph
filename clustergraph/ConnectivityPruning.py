import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


class ConnectivityPruning:
    """
    A class to prune edges from a graph using different algorithms while preserving connectivity.

    This class implements edge pruning strategies based on connectivity preservation. The algorithms aim
    to prune edges in a manner that minimally impacts the overall connectivity of the graph.

    Reference:
    ----------
    Zhou, F., Mahler, S., Toivonen, H.: Simplification of Networks by Edge Pruning.
    In: Berthold, M.R. (ed.) Bisociative Knowledge Discovery: An Introduction to
    Concept, Algorithms, Tools, And Applications, pp. 179–198. Springer, Berlin, Heidelberg (2012).
    https://doi.org/10.1007/978-3-642-31830-6_13

    Parameters
    ----------
    algo : str, optional
        The algorithm used to prune edges. Possible values are:
        - "bf" for Brute Force (slower but better)
        - "ps" for Path Simplification (faster but less accurate)
        Default is "bf".
    weight : str, optional
        The key under which the weight of edges is stored in the graph. Default is "weight".
    """

    def __init__(self, algo="bf", weight="weight"):
        """
        Initializes the ConnectivityPruning object with the specified algorithm and edge weight key.

        Parameters
        ----------
        algo : str, optional
            The algorithm used to prune edges. Options are "bf" or "ps". Default is "bf".
        weight : str, optional
            The key under which the weight/size of edges is stored in the graph. Default is "weight".
        """
        self.weight = weight
        if algo == "bf":
            self.prune = self.BF_edge_choice
        else:
            self.prune = self.PS_edge_choice

    def connectivity_graph(self, graph):
        """
        Computes the global connectivity of a given graph.

        The global connectivity is calculated by summing the inverse shortest paths between all pairs of nodes.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph for which the global connectivity is computed.

        Returns
        -------
        float
            The global connectivity score of the graph.
        """
        nodes = list(graph.nodes)
        short_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=self.weight))
        nb_nodes = len(nodes)
        C_V_E = 0
        nb_not_existing_path = 0
        for i in range(nb_nodes):
            for j in range(i, nb_nodes):
                if i != j:
                    try:
                        C_V_E += 1 / short_paths[nodes[i]][nodes[j]]
                    except:
                        nb_not_existing_path += 1

        if nb_not_existing_path == 0:
            C_V_E = C_V_E * 2 / (nb_nodes * (nb_nodes - 1))
        else:
            C_V_E = C_V_E * (2 / (nb_nodes * (nb_nodes - 1)) - 1 / nb_not_existing_path)

        return C_V_E

    def BF_edge_choice(self, g, nb_edges=-1, score=False):
        """
        Prunes edges from the graph using the Brute Force algorithm based on connectivity.

        Parameters
        ----------
        g : networkx.Graph
            The input graph from which edges are to be pruned.
        nb_edges : int, optional
            The number of edges to prune. If -1, all edges will be considered for pruning.
            Default is -1.
        score : bool, optional
            If True, returns the connectivity score evolution after each pruning step.
            Default is False.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list
            A list of removed edges in the form of tuples (u, v).
        list, optional
            If `score=True`, a list of float values representing the connectivity score evolution.
        """
        graph = g.copy()
        f = list(graph.edges)
        removed_edges = []
        M = []
        conn_prune = [1]

        if nb_edges == -1:
            nb_edges = len(f)

        for i in range(nb_edges):
            rk_largest = float("-inf")
            e_largest = False

            # Get F \ M (remaining edges after pruning)
            f_minus_M = deepcopy(f)
            if len(f_minus_M) != len(f):
                raise Exception("Error: length mismatch")

            for e in M:
                for i in range(len(f_minus_M)):
                    if f_minus_M[i][0] == e[0] and f_minus_M[i][1] == e[1]:
                        f_minus_M.pop(i)
                        break

            c_fix_loop = self.connectivity_graph(graph)

            for edge in f_minus_M:
                edge_data = deepcopy(graph.get_edge_data(edge[0], edge[1]))
                graph.remove_edge(edge[0], edge[1])
                nb_compo = nx.number_connected_components(graph)

                if nb_compo == 1:
                    rk = self.connectivity_graph(graph) / c_fix_loop

                    if rk > rk_largest:
                        rk_largest = rk
                        e_largest = edge
                else:
                    M.append(edge)

                graph.add_edge(edge[0], edge[1], **edge_data)

            if not isinstance(e_largest, bool):
                conn_prune.append(rk_largest)
                f = [e for e in f if e != e_largest]
                removed_edges.append((e_largest[0], e_largest[1]))
                graph.remove_edge(e_largest[0], e_largest[1])

        if not score:
            return graph, removed_edges
        else:
            return graph, removed_edges, conn_prune

    def PS_edge_choice(self, g, nb_edges, score=False):
        """
        Prunes edges from the graph using the Path Simplification algorithm.

        Parameters
        ----------
        g : networkx.Graph
            The input graph from which edges are to be pruned.
        nb_edges : int
            The number of edges to prune.
        score : bool, optional
            If True, returns the evolution of the evaluation criteria after each pruning step.
            Default is False.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list
            A list of removed edges in the form of tuples (u, v).
        list, optional
            If `score=True`, a list of float values representing the evaluation criteria after each pruning step.
        """
        graph = g.copy()
        f = list(graph.edges)
        M = []
        removed_edges = []
        lost_prune = []

        for i in range(nb_edges):
            k_largest = float("-inf")
            e_largest = False

            # Get F \ M (remaining edges after pruning)
            f_minus_M = deepcopy(f)
            if len(f_minus_M) != len(f):
                raise Exception("Error: length mismatch")

            for e in M:
                for i in range(len(f_minus_M)):
                    if f_minus_M[i][0] == e[0] and f_minus_M[i][1] == e[1]:
                        f_minus_M.pop(i)
                        break

            for edge in f_minus_M:
                edge_data = deepcopy(graph.get_edge_data(edge[0], edge[1]))
                edge_err = deepcopy(edge_data[self.weight])
                graph.remove_edge(edge[0], edge[1])

                try:
                    min_path_error = 1 / nx.dijkstra_path_length(
                        graph, edge[0], edge[1], weight=self.weight
                    )
                except nx.NetworkXNoPath:
                    min_path_error = -1

                graph.add_edge(edge[0], edge[1], **edge_data)

                if min_path_error >= 1 / edge_err:
                    k = 1
                    # Delete the edge
                    f = [e for e in f if e != edge]
                    graph.remove_edge(edge[0], edge[1])
                    e_largest = False
                    break
                elif 0 < min_path_error < 1 / edge_err:
                    k = min_path_error / (1 / edge_err)
                else:
                    k = float("-inf")
                    M.append([edge[0], edge[1]])

                if k > k_largest:
                    k_largest = k
                    e_largest = deepcopy(edge)

            if not isinstance(e_largest, bool):
                f = [e for e in f if e != e_largest]
                lost_prune.append(k_largest)
                removed_edges.append((e_largest[0], e_largest[1]))
                graph.remove_edge(e_largest[0], e_largest[1])

            if len(f) != len(graph.edges):
                print("EMERGENCY: Edge list mismatch")
                raise Exception("Emergency: Edge list mismatch")

        if not score:
            return graph, removed_edges
        else:
            plt.scatter(range(len(lost_prune)), lost_prune)
            return graph, removed_edges, lost_prune
