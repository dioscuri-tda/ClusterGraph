import scipy.spatial.distance as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from . import distances
from .c_GraphPreprocess import GraphPreprocess
from .GraphPruning import GraphPruning
from sklearn.neighbors import NearestNeighbors


class ClusterGraph(GraphPreprocess, GraphPruning):
    """
    ClusterGraph Class
    ==================

    A class representing a graph of clusters. The graph nodes represent clusters,
    and the edges represent the distances between clusters, which are computed based on the distance
    between points in the clusters or centroids.

    Inherits from:
    --------------
    GraphPreprocess
    GraphPruning

    Attributes
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input data points.
    clusters : list of arrays
        A list where each element is an array representing the points in a cluster.
    metric_clusters : str, optional
        The method used to calculate the distance between clusters. Default is "centroids".
    metric_points : callable, optional
        The distance metric used to calculate distances between points. Default is Euclidean distance.
    parameters_metric_points : dict, optional
        Additional parameters for the point distance metric.
    type_pruning : str, optional
        The type of pruning method to apply. Default is "conn".
    algo : str, optional
        The algorithm used for graph pruning. Default is "bf".
    weight : str, optional
        The edge weight attribute name in the graph. Default is "weight".
    knn_g : networkx.Graph or None, optional
        The precomputed k-NN graph. Default is None, meaning the k-NN graph will be computed.
    weight_knn_g : str, optional
        The edge weight attribute for the k-NN graph. Default is "weight".
    k_compo : int, optional
        A parameter related to the graph pruning. Default is 2.
    dist_weight : bool, optional
        Whether to apply distance-based weighting for pruning. Default is True.
    Graph : networkx.Graph
        The graph representing the clusters and their distances.
    is_knn_computed : int
        Flag indicating whether the k-NN graph has been computed.
    original_graph : networkx.Graph
        The original, unpruned graph.
    """

    def __init__(
        self,
        X,
        clusters,
        metric_clusters="centroids",
        metric_points=sp.euclidean,
        parameters_metric_points={},
        type_pruning="conn",
        algo="bf",
        weight="weight",
        knn_g=None,
        weight_knn_g="weight",
        k_compo=2,
        dist_weight=True,
    ):
        """
        Initializes the ClusterGraph with the given parameters.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input data points.
        clusters : list of arrays
            A list where each element is an array representing the points in a cluster.
        metric_clusters : str, optional
            The method used to calculate the distance between clusters. Default is "centroids".
        metric_points : callable, optional
            The distance metric used to calculate distances between points. Default is Euclidean distance.
        parameters_metric_points : dict, optional
            Additional parameters for the point distance metric.
        type_pruning : str, optional
            The type of pruning method to apply. Default is "conn".
        algo : str, optional
            The algorithm used for graph pruning. Default is "bf".
        weight : str, optional
            The edge weight attribute name in the graph. Default is "weight".
        knn_g : networkx.Graph or None, optional
            The precomputed k-NN graph. Default is None, meaning the k-NN graph will be computed.
        weight_knn_g : str, optional
            The edge weight attribute for the k-NN graph. Default is "weight".
        k_compo : int, optional
            A parameter related to the graph pruning. Default is 2.
        dist_weight : bool, optional
            Whether to apply distance-based weighting for pruning. Default is True.
        """
        self.clusters = clusters
        self.is_knn_computed = -1
        self.X = X

        if knn_g is None or isinstance(
            knn_g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
        ):
            self.knn_g = knn_g
            self.is_knn_computed = 0
        elif isinstance(knn_g, int):
            neigh = NearestNeighbors(n_neighbors=knn_g, radius=1)
            neigh.fit(X=X)
            nn_adjacency = neigh.kneighbors_graph(
                X=X, n_neighbors=knn_g, mode="distance"
            )
            nn_Graph = nx.from_scipy_sparse_array(
                nn_adjacency, edge_attribute=weight_knn_g
            )

            for node in nn_Graph.nodes:
                nn_Graph.remove_edge(node, node)
            self.knn_g = nn_Graph
            self.is_knn_computed = knn_g
        else:
            raise TypeError(
                "The variable 'knn_g' must be None, an integer or a networkx Graph."
            )

        # distance between ids of datapoints
        if metric_points == "precomputed":
            self.distance_points = lambda i, j: X[i, j]
        elif metric_clusters == "centroids":
            self.distance_points = lambda c_i, c_j: metric_points(
                np.mean(X[c_i], axis=0),
                np.mean(X[c_j], axis=0),
                **parameters_metric_points
            )
        else:
            self.distance_points = lambda i, j: metric_points(
                X[i], X[j], **parameters_metric_points
            )

        # distance between clusters
        if metric_clusters == "centroids":
            self.distance_clusters = distances.centroid_dist
        elif metric_clusters == "average":
            self.distance_clusters = distances.average_dist
        elif metric_clusters == "min":
            self.distance_clusters = distances.min_dist
        elif metric_clusters == "max":
            self.distance_clusters = distances.max_dist
        elif metric_clusters == "emd":
            self.distance_clusters = distances.EMD_for_two_clusters
        else:
            raise ValueError(
                "The value {} is not a valid distance. Options are 'min', 'max', 'average', 'emd'".format(
                    metric_clusters
                )
            )

        # Creation of the ClusterGraph
        self.Graph = nx.Graph()

        # one node for each cluster
        self.Graph.add_nodes_from(
            [(i, dict(size=len(c), points_covered=c)) for i, c in enumerate(clusters)]
        )

        # compute all distances and add all edges
        self.Graph.add_weighted_edges_from(
            [
                (i, j, self.distance_clusters(C_i, C_j, self.distance_points))
                for i, C_i in enumerate(clusters[:-1])
                for j, C_j in enumerate(clusters[i + 1 :], start=(i + 1))
            ],
            weight="weight",
        )

        GraphPreprocess.__init__(self)
        self.graph_prepro = self.Graph

        GraphPruning.__init__(
            self,
            graph=self.Graph,
            type_pruning=type_pruning,
            algo=algo,
            weight=weight,
            knn_g=self.knn_g,
            weight_knn_g=weight_knn_g,
            k_compo=k_compo,
            dist_weight=dist_weight,
        )
        self.original_graph = self.Graph

    def get_graph(self):
        """
        Retrieves the pruned or original graph.

        Returns
        -------
        networkx.Graph
            The ClusterGraph, either pruned or unpruned depending on the state of the graph.

        Notes
        -----
        If the graph is pruned, the pruned edges will be removed and the pruned graph will be returned.
        """
        if self.is_pruned != "not_pruned":
            return self.Graph
        else:
            pruned_graph = self.Graph.copy()
            pruned_graph.remove_edges_from(
                self.prunedEdgesHistory[self.is_pruned]["edges"]
            )
            return pruned_graph

    def prune_distortion(
        self,
        knn_g=10,
        nb_edge_pruned=-1,
        score=False,
        algo="bf",
        weight_knn_g="weight",
        k_compo=2,
        dist_weight=True,
    ):
        """
        Performs graph pruning based on distortion score.

        Parameters
        ----------
        knn_g : int or networkx.Graph, optional
            The number of nearest neighbors or a precomputed k-NN graph. Default is 10.
        nb_edge_pruned : int, optional
            The number of edges to prune. Default is -1 (no pruning limit).
        score : bool, optional
            Whether to compute and return the pruning score. Default is False.
        algo : str, optional
            The pruning algorithm. Default is "bf".
        weight_knn_g : str, optional
            The edge weight attribute for the k-NN graph. Default is "weight".
        k_compo : int, optional
            A parameter related to pruning. Default is 2.
        dist_weight : bool, optional
            Whether to apply distance-based weighting for pruning. Default is True.

        Returns
        -------
        networkx.Graph
            The pruned graph after distortion pruning.
        """

        if isinstance(knn_g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            self.knn_g = knn_g
            self.is_knn_computed = -1

        elif isinstance(knn_g, int):
            if self.is_knn_computed != knn_g:
                neigh = NearestNeighbors(n_neighbors=knn_g, radius=1)
                neigh.fit(X=self.X)
                nn_adjacency = neigh.kneighbors_graph(
                    X=self.X, n_neighbors=knn_g, mode="distance"
                )
                nn_Graph = nx.from_scipy_sparse_array(
                    nn_adjacency, edge_attribute=weight_knn_g
                )

                for node in nn_Graph.nodes:
                    nn_Graph.remove_edge(node, node)
                self.knn_g = nn_Graph
                self.is_knn_computed = knn_g

        return self.prune_distortion_pr(
            knn_g=self.knn_g,
            nb_edge_pruned=nb_edge_pruned,
            score=score,
            algo=algo,
            weight_knn_g=weight_knn_g,
            k_compo=k_compo,
            dist_weight=dist_weight,
            is_knn_computed=self.is_knn_computed,
        )

    def add_coloring(
        self,
        coloring_df,
        custom_function=np.mean,
    ):
        """Takes pandas dataframe and compute the average \
        of each column for the subset of points covered by each node.
        Add such values as attributes to each node in the Graph

        Parameters
        ----------
        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function)
        custom_function : callable, optional
            a function to compute on the `coloring_df` columns, by default numpy.mean
        custom_name : string, optional
            sets the attributes naming scheme, by default None, the attribute names will be the column names
        add_std: bool, default=False
            Wheter to compute also the standard deviation on each ball
        """
        # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
        for node in self.Graph.nodes:
            for col_name, avg in (
                coloring_df.loc[self.Graph.nodes[node]["points_covered"]]
                .apply(custom_function, axis=0)
                .items()
            ):
                self.Graph.nodes[node][col_name] = avg
