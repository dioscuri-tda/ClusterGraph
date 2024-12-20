from scipy.spatial.distance import euclidean
from networkx import add_path
import networkx as nx
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random

from .ConnectivityPruning import ConnectivityPruning
from .Metric_distortion_class import Metric_distortion


class GraphPruning:
    """
    A class to perform pruning of a graph using different strategies like connectivity pruning or metric distortion pruning.

    The class allows for the pruning of edges based on connectivity preservation or metric distortion.
    It also supports merging disconnected components and pruning edges in the merged graph to reduce noise.
    """

    def __init__(
        self,
        graph=None,
        type_pruning="conn",
        algo="bf",
        weight="weight",
        knn_g=None,
        weight_knn_g="weight",
        k_compo=2,
        dist_weight=True,
    ):
        """
        Initializes the GraphPruning object with the provided graph and pruning strategy.

        Parameters
        ----------
        graph : networkx.Graph, optional
            The graph to prune. Default is None.
        type_pruning : str, optional
            The type of pruning to apply. Options are:
            "conn" for connectivity pruning (default),
            "md" for metric distortion pruning.
        algo : str, optional
            The algorithm to use for pruning edges. Options are:
            "bf" for brute-force (best but slowest), "ps" for path simplification (faster).
            Default is "bf".
        weight : str, optional
            The key in the graph to use as the edge weight. Default is "weight".
        knn_g : networkx.Graph, optional
            A k-nearest neighbors graph used for metric distortion pruning. Default is None.
        weight_knn_g : str, optional
            The key for the edge weight in the knn graph. Default is "weight".
        k_compo : int, optional
            The number of edges to add to merge disconnected components after metric distortion pruning.
            Default is 2.
        dist_weight : bool, optional
            If True, edge weights are used when calculating metric distortion. Default is True.
        """
        if graph is not None:
            self.original_graph = graph

        self.pruned_graph = None
        self.merged_graph = None
        self.is_pruned = "not_pruned"
        self.prunedEdgesHistory = {
            "md_bf": {"all_pruned": False, "edges": [], "score": [], "knn_g": -1},
            "md_ps": {"all_pruned": False, "edges": [], "score": [], "knn_g": -1},
            "conn_bf": {"all_pruned": False, "edges": [], "score": []},
            "conn_ps": {"all_pruned": False, "edges": [], "score": []},
            "in_between_compo": {"edges": []},
            "conn_merged": {
                "all_pruned": False,
                "edges": [],
                "score": [],
                "k_compo": -1,
                "other_edges_remove": [],
            },
        }

        if graph is not None:
            if type_pruning == "conn":
                self.prunedStrategy = ConnectivityPruning(algo=algo, weight=weight)
            elif type_pruning == "md":
                self.prunedStrategy = Metric_distortion(
                    graph=self.original_graph,
                    knn_g=knn_g,
                    weight_knn_g=weight_knn_g,
                    k_compo=k_compo,
                    dist_weight=dist_weight,
                    algo=algo,
                )

    def prune(self, graph=None, nb_edge_pruned=-1, score=False):
        """
        Prunes edges from the graph using the selected pruning strategy.

        Parameters
        ----------
        graph : networkx.Graph, optional
            The graph to prune. If None, the graph provided at initialization is used.
            Default is None.
        nb_edge_pruned : int, optional
            The maximum number of edges to prune. If -1, all possible edges will be pruned.
            Default is -1.
        score : bool, optional
            If True, the method returns the score evolution (connectivity or metric distortion).
            Default is False.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list of float, optional
            A list of float values representing the evolution of the score if `score` is True.
        """
        if graph is None:
            graph = self.original_graph

        if score:
            self.pruned_graph, evolScore = self.prunedStrategy.prune(
                graph, nb_edge_pruned, score
            )
            return self.pruned_graph, evolScore
        else:
            self.pruned_graph = self.prunedStrategy.prune(graph, nb_edge_pruned, score)
            return self.pruned_graph

    def merge_graph_draft(self, pruned_gg=None, nb_edges=-1):
        """
        Merges disconnected components in the graph and prunes edges among the merged components.

        Parameters
        ----------
        pruned_gg : networkx.Graph, optional
            The graph to merge and prune. If None, the pruned graph is used.
            Default is None.
        nb_edges : int, optional
            The maximum number of edges to prune from the merged graph. If -1, all edges will be pruned.
            Default is -1.

        Returns
        -------
        networkx.Graph
            The merged and pruned graph.
        """
        if pruned_gg is None:
            pruned_gg = self.pruned_graph

        self.merged_graph = self.prunedStrategy.conn_prune_merged_graph(
            pruned_gg, nb_edges
        ).copy()
        return self.merged_graph

    def prune_distortion_pr(
        self,
        knn_g,
        nb_edge_pruned=-1,
        score=False,
        algo="bf",
        weight_knn_g="weight",
        k_compo=2,
        dist_weight=True,
        is_knn_computed=-1,
    ):
        """
        Performs metric distortion pruning using the k-nearest neighbors graph.

        Parameters
        ----------
        knn_g : networkx.Graph
            The k-nearest neighbors graph to use for metric distortion pruning.
        nb_edge_pruned : int, optional
            The number of edges to prune. If -1, all possible edges will be pruned. Default is -1.
        score : bool, optional
            If True, the method returns the score evolution (metric distortion). Default is False.
        algo : str, optional
            The pruning algorithm to use. Options are "bf" for brute-force and "ps" for path simplification.
            Default is "bf".
        weight_knn_g : str, optional
            The key for edge weight in the k-nearest neighbors graph. Default is "weight".
        k_compo : int, optional
            The number of edges to add to merge disconnected components after pruning. Default is 2.
        dist_weight : bool, optional
            If True, edge weights are used when calculating metric distortion. Default is True.
        is_knn_computed : int, optional
            The identifier for the computed k-nearest neighbors graph.

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list of float, optional
            A list of float values representing the evolution of the score if `score` is True.
        """
        if algo not in ["bf", "ps"]:
            raise ValueError("The algorithm can only be 'bf' or 'ps'.")

        self.is_pruned = "md_" + algo

        if (
            self.prunedEdgesHistory[self.is_pruned]["all_pruned"]
            and self.prunedEdgesHistory[self.is_pruned]["knn_g"] == is_knn_computed
        ) or (
            nb_edge_pruned > 0
            and len(self.prunedEdgesHistory[self.is_pruned]["edges"]) >= nb_edge_pruned
        ):
            pruned_graph = self.original_graph.copy()
            if nb_edge_pruned == -1:
                nb_edge_pruned = len(self.prunedEdgesHistory[self.is_pruned]["edges"])
            pruned_graph.remove_edges_from(
                self.prunedEdgesHistory[self.is_pruned]["edges"][:nb_edge_pruned]
            )
            pruned_graph.remove_edges_from(
                self.prunedEdgesHistory["in_between_compo"]["edges"]
            )

            if score:
                return (
                    pruned_graph,
                    self.prunedEdgesHistory[self.is_pruned]["score"][:nb_edge_pruned],
                )
            else:
                return pruned_graph

        self.prunedEdgesHistory[self.is_pruned]["knn_g"] = is_knn_computed
        self.prunedMetricDistortionStrategy = Metric_distortion(
            graph=self.original_graph,
            knn_g=knn_g,
            weight_knn_g=weight_knn_g,
            k_compo=k_compo,
            dist_weight=dist_weight,
            algo=algo,
        )

        if nb_edge_pruned == -1:
            self.prunedEdgesHistory[self.is_pruned]["all_pruned"] = True

        pruned_graph, removed_edges, evolScore = (
            self.prunedMetricDistortionStrategy.prune(
                self.original_graph, nb_edge_pruned, True
            )
        )
        self.prunedEdgesHistory[self.is_pruned]["edges"] = deepcopy(removed_edges)
        self.prunedEdgesHistory[self.is_pruned]["score"] = evolScore
        self.prunedEdgesHistory["in_between_compo"]["edges"] = [
            (e[0], e[1])
            for e in self.prunedMetricDistortionStrategy.edges_between_compo
        ]

        if score:
            return pruned_graph, evolScore
        else:
            return pruned_graph

    def prune_conn(self, nb_edge_pruned=-1, score=False, algo="bf", weight="weight"):
        """
        Performs connectivity pruning to retain the most important edges in the graph.

        Parameters
        ----------
        nb_edge_pruned : int, optional
            The number of edges to prune. If -1, all possible edges will be pruned. Default is -1.
        score : bool, optional
            If True, the method returns the score evolution (connectivity). Default is False.
        algo : str, optional
            The pruning algorithm to use. Options are "bf" for brute-force and "ps" for path simplification.
            Default is "bf".
        weight : str, optional
            The key for edge weight in the graph. Default is "weight".

        Returns
        -------
        networkx.Graph
            The pruned graph.
        list of float, optional
            A list of float values representing the evolution of the score if `score` is True.
        """
        if algo not in ["bf", "ps"]:
            raise ValueError("The algorithm can only be 'bf' or 'ps'.")

        self.is_pruned = "conn_" + algo

        if self.prunedEdgesHistory[self.is_pruned]["all_pruned"] or (
            nb_edge_pruned > 0
            and len(self.prunedEdgesHistory[self.is_pruned]["edges"]) >= nb_edge_pruned
        ):
            pruned_graph = self.original_graph.copy()
            pruned_graph.remove_edges_from(
                self.prunedEdgesHistory[self.is_pruned]["edges"][:nb_edge_pruned]
            )
            if score:
                return (
                    pruned_graph,
                    self.prunedEdgesHistory[self.is_pruned]["score"][:nb_edge_pruned],
                )
            else:
                return pruned_graph

        self.prunedConnectivityStrategy = ConnectivityPruning(algo=algo, weight=weight)

        if nb_edge_pruned == -1:
            self.prunedEdgesHistory[self.is_pruned]["all_pruned"] = True

        pruned_graph, removed_edges, evolScore = self.prunedConnectivityStrategy.prune(
            self.original_graph, nb_edge_pruned, True
        )
        self.prunedEdgesHistory[self.is_pruned]["edges"] = removed_edges
        self.prunedEdgesHistory[self.is_pruned]["score"] = evolScore

        if score:
            return pruned_graph, evolScore
        else:
            return pruned_graph

    def merge_graph(self, nb_edges=-1, k_compo=2, score=False):
        """
        Merges disconnected components in the graph and prunes edges among the merged components.

        Parameters
        ----------
        nb_edges : int, optional
            The number of edges to prune from the merged graph. If -1, all edges will be pruned.
            Default is -1.
        k_compo : int, optional
            The number of edges to add to merge disconnected components after pruning.
            Default is 2.
        score : bool, optional
            If True, the method returns the score evolution (connectivity).
            Default is False.

        Returns
        -------
        networkx.Graph
            The merged and pruned graph.
        list of float, optional
            A list of float values representing the evolution of the score if `score` is True.
        """
        merged_graph = self.original_graph.copy()
        pruning = None
        if self.prunedEdgesHistory["md_bf"]["all_pruned"]:
            pruning = "md_bf"
        elif self.prunedEdgesHistory["md_ps"]["all_pruned"]:
            pruning = "md_ps"
        else:
            raise ValueError("The metric distortion pruning has to be launched first.")

        if (
            self.prunedEdgesHistory["conn_merged"]["all_pruned"]
            and k_compo == self.prunedEdgesHistory["conn_merged"]["k_compo"]
        ):
            merged_graph = self.get_merged_graph(pruning, nb_edges)
            conn_score = self.prunedEdgesHistory["conn_merged"]["score"]
        else:
            merged_graph.remove_edges_from(self.prunedEdgesHistory[pruning]["edges"])
            merged_graph.remove_edges_from(
                self.prunedEdgesHistory["in_between_compo"]["edges"]
            )
            g, removed_edges, conn_score = (
                self.prunedMetricDistortionStrategy.conn_prune_merged_graph(
                    merged_graph, None, k_compo
                )
            )
            self.prunedEdgesHistory["conn_merged"]["all_pruned"] = True
            self.prunedEdgesHistory["conn_merged"]["edges"] = deepcopy(removed_edges)
            between_not_current = [
                e
                for e in self.prunedEdgesHistory["in_between_compo"]["edges"]
                if e not in removed_edges and (e[1], e[0]) not in removed_edges
            ]
            self.prunedEdgesHistory["conn_merged"]["other_edges_remove"] = deepcopy(
                between_not_current
            )
            self.prunedEdgesHistory["conn_merged"]["score"] = conn_score
            self.prunedEdgesHistory["conn_merged"]["k_compo"] = k_compo
            merged_graph = self.get_merged_graph(pruning, nb_edges)

        if score:
            return merged_graph, conn_score
        else:
            return merged_graph

    def get_merged_graph(self, key, nb_edges):
        """
        Retrieves the merged graph after pruning edges.

        Parameters
        ----------
        key : str
            The key indicating which pruning strategy was used ("md_bf", "md_ps", or "conn_merged").
        nb_edges : int
            The number of edges to prune. If -1, all edges will be pruned.

        Returns
        -------
        networkx.Graph
            The merged and pruned graph.
        """
        merged_graph = self.original_graph.copy()
        edges_to_prune = deepcopy(self.prunedEdgesHistory[key]["edges"])

        if nb_edges == -1:
            nb_edges = len(self.prunedEdgesHistory["conn_merged"]["edges"])

        edges_to_prune.extend(
            deepcopy(self.prunedEdgesHistory["conn_merged"]["edges"][:nb_edges])
        )
        edges_to_prune.extend(
            deepcopy(self.prunedEdgesHistory["conn_merged"]["other_edges_remove"])
        )

        merged_graph.remove_edges_from(edges_to_prune)
        return merged_graph
