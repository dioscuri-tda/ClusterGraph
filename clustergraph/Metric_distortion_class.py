from scipy.spatial.distance import euclidean
from networkx import add_path
import networkx as nx
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random


class Metric_distortion:

    def __init__(
        self,
        graph,
        knn_g,
        weight_knn_g="weight",
        k_compo=2,
        dist_weight=True,
        algo="bf",
    ):
        """
        Initializes the Metric_distortion object for pruning a graph based on metric distortion.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to prune.
        knn_g : networkx.Graph
            The k-nearest neighbors graph from which the intrinsic distance between points of the dataset is retrieved.
            The dataset should be the same as the one used for computing the "graph".
        weight_knn_g : str, optional
            Key/label under which the weight of edges is stored in the “knn_g” graph. The weight corresponds to the distance
            between two nodes (default is 'weight').
        k_compo : int, optional
            Number of edges that will be added to each disconnected component to merge them after the metric distortion
            pruning process. The edges added are those that connect disconnected components and are the shortest (default is 2).
        dist_weight : bool, optional
            If True, the distortion is computed with weight on edges. If False, the distortion is computed without weights (default is True).
        algo : {'bf', 'ps'}, optional
            Choice of the algorithm used to prune edges. 'bf' refers to the brute force algorithm (slowest) and 'ps' to the
            quickest algorithm. (default is 'bf').
        """
        self.graph = graph
        self.knn_g = knn_g
        self.weight_knn_g = weight_knn_g
        self.label_points_covered_intr = "points_covered"
        self.nb_points_disco = 0
        self.nb_points_should_be_evaluated = 0
        self.k_nn_compo = k_compo
        self.dijkstra_length_dict = dict(
            nx.all_pairs_dijkstra_path_length(self.knn_g, weight=weight_knn_g)
        )

        if algo == "bf":
            self.prune = self.prune_edges_BF
        elif algo == "ps":
            self.prune = self.prune_edges_PS

        if dist_weight:
            self.distortion_graph = self.distortion_graph_weight
        else:
            self.distortion_graph = self.distortion_graph_no_weight

        # Creation of the intrinsic cluster graph
        self.intri_cg = graph.copy()
        nodes = list(self.intri_cg.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.intri_cg.add_edge(nodes[i], nodes[j])

        self.intrin_dist_cg()

    def intrin_dist_cg(self):
        """
        Computes the intrinsic distance between clusters and adds them to the graph.

        This method creates the intrinsic graph in which the distance between nodes is the average shortest path between
        points. It removes edges between disconnected components and updates the graph with the intrinsic distances between
        connected clusters.

        Returns
        -------
        None
        """
        edges_between_compos, self.intri_cg = self.remove_edges(self.intri_cg)
        connected_components = [
            self.intri_cg.subgraph(c) for c in nx.connected_components(self.intri_cg)
        ]

        for cc in connected_components:
            for n1 in cc.nodes:
                for n2 in cc.nodes:
                    if n1 < n2:
                        intr_dist_c1_c2 = self.intr_two_clusters(
                            cc.nodes[n1][self.label_points_covered_intr],
                            cc.nodes[n2][self.label_points_covered_intr],
                        )
                        self.intri_cg.edges[(n1, n2)]["intr_dist"] = intr_dist_c1_c2

        if self.nb_points_disco > 0:
            print(
                "Be careful, ",
                self.nb_points_disco,
                " intrinsic distance between points have not been evaluated in the metric distortion process. It represents ",
                self.nb_points_disco / self.nb_points_should_be_evaluated,
                " of the distances.",
            )

    def intr_two_clusters(self, c1, c2):
        """
        Computes the average intrinsic distance between two clusters, ignoring disconnected points.

        Parameters
        ----------
        c1 : list of int
            List of indices representing the first cluster.
        c2 : list of int
            List of indices representing the second cluster.

        Returns
        -------
        float
            The intrinsic distance between the two clusters.
        """
        l_c1 = len(c1)
        l_c2 = len(c2)
        intr_dist = 0
        nb_not_connected = 0
        for i in c1:
            for j in c2:
                dist = self.intr_two_points(i, j)
                if dist >= 0:
                    intr_dist += dist
                else:
                    nb_not_connected += 1
        self.nb_points_disco += nb_not_connected
        self.nb_points_should_be_evaluated += l_c1 * l_c2
        return intr_dist / (l_c1 * l_c2 - nb_not_connected)

    def intr_two_points(self, i, j):
        """
        Computes the intrinsic distance between two data points, represented by their indices, using the shortest path.

        Parameters
        ----------
        i : int
            Index of the first data point.
        j : int
            Index of the second data point.

        Returns
        -------
        float
            The shortest path (intrinsic distance) between the two points, or -1 if the points are not in the same connected component.
        """
        try:
            return self.dijkstra_length_dict[i][j]
        except:
            return -1

    def distortion_graph_no_weight(self, graph, intrinsic_graph):
        """
        Computes the distortion between a graph and its intrinsic version without taking cluster sizes into account.

        Parameters
        ----------
        graph : networkx.Graph
            The graph for which the distortion is computed.
        intrinsic_graph : networkx.Graph
            The intrinsic graph containing the edges' intrinsic distances between clusters.

        Returns
        -------
        float
            The distortion between the graph and its intrinsic counterpart. Returns 0 if no edges could be evaluated.
        """
        connected_components = [
            graph.subgraph(c).copy() for c in nx.connected_components(graph)
        ]
        short_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
        distortion_g = 0
        nb_pair = 0

        for cc in connected_components:
            dist_cc = 0
            nodes = list(cc.nodes)
            nodes.sort()
            for n1 in nodes:
                for n2 in nodes:
                    if n1 < n2:
                        nb_pair += 1
                        distortion_g += abs(
                            np.log10(
                                (short_paths[n1][n2])
                                / intrinsic_graph.edges[(n1, n2)]["intr_dist"]
                            )
                        )
                        dist_cc += abs(
                            np.log10(
                                (short_paths[n1][n2])
                                / intrinsic_graph.edges[(n1, n2)]["intr_dist"]
                            )
                        )

        if nb_pair > 0:
            return distortion_g / nb_pair
        else:
            print("No edge to evaluate")
            return 0

    def distortion_graph_weight(self, graph, intrinsic_graph):
        """
        Computes the distortion between a graph and its intrinsic version while considering the sizes of clusters.

        Parameters
        ----------
        graph : networkx.Graph
            The graph for which the distortion is computed.
        intrinsic_graph : networkx.Graph
            The intrinsic graph containing the edges' intrinsic distances between clusters.

        Returns
        -------
        float
            The weighted distortion between the graph and its intrinsic counterpart. Returns 0 if no edges could be evaluated.
        """
        connected_components = [
            graph.subgraph(c).copy() for c in nx.connected_components(graph)
        ]
        short_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
        dist_global = 0
        nb_pair_global = 0

        for cc in connected_components:
            nodes = list(cc.nodes)
            weight_total = 0
            dist_compo = 0
            nb_pair_compo = 0
            for n1 in nodes:
                for n2 in nodes:
                    if n1 < n2:
                        weight_n1_n2 = (
                            self.nb_compo_cluster[n1] + self.nb_compo_cluster[n2]
                        )
                        dist_pair = abs(
                            np.log10(
                                (short_paths[n1][n2])
                                / intrinsic_graph.edges[(n1, n2)]["intr_dist"]
                            )
                        )
                        dist_compo += weight_n1_n2 * dist_pair
                        weight_total += weight_n1_n2
                        nb_pair_compo += 1
            if nb_pair_compo > 0:
                dist_compo = dist_compo / (nb_pair_compo * weight_total)
                nb_pair_global += nb_pair_compo
                dist_global += dist_compo * nb_pair_compo

        if nb_pair_global > 0:
            dist_global = dist_global / nb_pair_global
            return dist_global
        else:
            print("No edge to evaluate")
            return 0

    def associate_cluster_one_compo(self, cluster):
        """
        For a given cluster, finds the dominating connected component in the k-nearest neighbors graph.

        Parameters
        ----------
        cluster : list of int
            List of indices representing the cluster.

        Returns
        -------
        int, int
            The index of the dominating connected component in the k-nearest neighbors graph, and the number of points
            that belong to this component.
        """
        connected_components = [
            self.knn_g.subgraph(c).copy() for c in nx.connected_components(self.knn_g)
        ]
        number_per_compo = []
        for cc in connected_components:
            nodes = list(cc.nodes)
            count = 0
            for pt in cluster:
                if pt in nodes:
                    count += 1
            number_per_compo.append(count)

        return np.argmax(number_per_compo), max(number_per_compo)

    def associate_clusters_compo(self):
        """
        Associates each node with a connected component in the k-nearest neighbors graph based on the most represented
        component within the cluster.

        Returns
        -------
        dict
            A dictionary with connected component indices as keys and a list of clusters that belong to each component.
        """
        nb_compo = nx.number_connected_components(self.knn_g)
        compo_clusters = {}
        self.nb_compo_cluster = {}
        for i in range(nb_compo):
            compo_clusters[i] = []
        for n in self.graph.nodes:
            compo, nb = self.associate_cluster_one_compo(
                self.graph.nodes[n][self.label_points_covered_intr]
            )
            compo_clusters[compo].append(n)
            self.nb_compo_cluster[n] = nb

        return compo_clusters

    def remove_edges(self, graph):
        """
        Removes edges connecting nodes that belong to different disconnected components in the k-nearest neighbors graph.

        Parameters
        ----------
        graph : networkx.Graph
            The graph from which edges between disconnected components should be removed.

        Returns
        -------
        list, networkx.Graph
            A list of removed edges with their data, and the updated graph.
        """
        compo_clusters = self.associate_clusters_compo()
        keys = list(compo_clusters)
        edges_in_between = []
        for e in self.graph.edges:
            found = False
            for k in keys:
                if (e[0] in compo_clusters[k]) and (e[1] in compo_clusters[k]):
                    found = True
                    break
            if not found:
                data = deepcopy(self.graph.edges[e])
                edges_in_between.append([e[0], e[1], data])
                graph.remove_edge(e[0], e[1])

        return edges_in_between, graph

    def prune_edges_BF(self, graph, nb_edges_pruned=-1, md_plot=True):
        """
        Iteratively prunes edges by selecting the edge that minimizes metric distortion at each iteration.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to prune.
        nb_edges_pruned : int, optional
            The maximum number of edges to prune. If negative, all edges will be pruned, by default -1.
        md_plot : bool, optional
            If True, the method will return a list with the metric distortion at each iteration, by default True.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list
            The list of pruned edges, or the list of metric distortions per iteration if `md_plot` is True.
        """
        # Method implementation...

    def prune_edges_PS(self, g, nb_edges_pruned=None, md_plot=True):
        """
        Prunes edges iteratively by selecting the edge with the highest metric distortion at each iteration.

        Parameters
        ----------
        g : networkx.Graph
            The graph to prune.
        nb_edges_pruned : int, optional
            The maximum number of edges to prune. If None, all edges are considered, by default None.
        md_plot : bool, optional
            If True, the method will return a list with the metric distortion at each iteration, by default True.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list
            The list of pruned edges, or the list of metric distortions per iteration if `md_plot` is True.
        """
        # Method implementation...

    def greedy_pruning(self, alpha=0.5, nb_edges=-1, weight="distortion"):
        """
        Prunes edges with distortion lower than a given threshold `alpha`, prioritizing edges with higher distortion.

        Parameters
        ----------
        alpha : float, optional
            The threshold for edge distortion. Edges with distortion smaller than `alpha` will be pruned, by default 0.5.
        nb_edges : int, optional
            The number of edges to prune. If negative, all edges are considered, by default -1.
        weight : str, optional
            The attribute name used to store distortion on edges, by default "distortion".

        Returns
        -------
        networkx.Graph
            The pruned graph.
        """
        # Method implementation...

    def set_distortion_edges(self, graph, weight="distortion"):
        """
        Computes the distortion for each edge in the graph as the ratio of edge weight to intrinsic distance.

        Parameters
        ----------
        graph : networkx.Graph
            The graph on which distortion is computed.
        weight : str, optional
            The label under which distortion is stored in the graph, by default 'distortion'.

        Returns
        -------
        networkx.Graph
            The graph with distortion values set on each edge.
        """
        # Method implementation...

    def plt_md_prune_computed(self, save=None):
        """
        Plots the evolution of metric distortion depending on the number of edges pruned.

        Parameters
        ----------
        save : str, optional
            If provided, the plot will be saved as a PDF with the specified filename, by default None.
        """
        # Method implementation...

    def get_distance_matrix_ccompo(self, pruned_graph):
        """
        Returns a distance matrix between all nodes in the graph, where the distance is the real distance for nodes in
        different components and the maximum distance for nodes in the same component.

        Parameters
        ----------
        pruned_graph : networkx.Graph
            The pruned graph to compute the distance matrix for.

        Returns
        -------
        numpy.ndarray
            A 2D distance matrix between all nodes in the graph.
        """
        # Method implementation...

    def create_knn_graph_merge_compo_CG(self, distance_matrix, k):
        """
        Creates a k-nearest neighbors graph from the provided distance matrix.

        Parameters
        ----------
        distance_matrix : numpy.ndarray
            The distance matrix for which the k-nearest neighbors graph is to be created.
        k : int
            The number of neighbors in the k-nearest neighbors graph.

        Returns
        -------
        networkx.Graph
            The k-nearest neighbors graph created from the distance matrix.
        """
        # Method implementation...

    def connectivity_graph(self, graph):
        """
        Computes the global connectivity of a graph based on the inverse shortest path distances between all pairs of nodes.

        Parameters
        ----------
        graph : networkx.Graph
            The graph for which connectivity is computed.

        Returns
        -------
        float
            The global connectivity value of the graph.
        """
        # Method implementation...

    def plt_conn_prune_computed(self):
        """
        Plots the evolution of connectivity depending on the number of edges pruned.

        Returns
        -------
        None
            Displays the plot of connectivity vs. number of pruned edges.
        """
        # Method implementation...

    def merge_components(self, pruned_gg):
        """
        Merges disconnected components in the graph by adding edges between the nearest neighbors from different components.

        Parameters
        ----------
        pruned_gg : networkx.Graph
            The pruned graph with disconnected components.

        Returns
        -------
        networkx.Graph
            The graph with added edges between components.
        list
            The list of edges added to the graph to connect components.
        """
        # Method implementation...

    def conn_prune_merged_graph(self, pruned_gg, nb_edges_pruned=None, k_compo=None):
        """
        After merging components, prunes a specified number of edges to obtain a less noisy graph.

        Parameters
        ----------
        pruned_gg : networkx.Graph
            The graph after merging components.
        nb_edges_pruned : int, optional
            The maximum number of edges to prune, by default None (prunes as many edges as possible).
        k_compo : int, optional
            The number of nearest neighbors between components, by default None.

        Returns
        -------
        networkx.Graph
            The pruned and merged graph.
        list
            The list of removed edges.
        list
            The list of connectivity values after each pruning iteration.
        """
        # Method implementation...
