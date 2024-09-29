import scipy.spatial.distance as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from . import distances
from .c_GraphPreprocess import GraphPreprocess
from .GraphPruning import GraphPruning
from pyballmapper import BallMapper


class ClusterGraph(GraphPreprocess, GraphPruning):

    def __init__(
        self,
        X,
        clusters,
        metric_clusters="average",
        # Parameters connected with Distance_between_points
        metric_points=sp.euclidean,
        parameters_metric_points={},
        type_pruning = "conn", 
        algo ="bf",
        weight = "label",
        knn_g = None,  
        weight_knn_g = 'label', 
        k_compo = 2,
        dist_weight = True 
    ):

        self.clusters = clusters

        # distance between ids of datapoints
        if metric_points == "precomputed":
            self.distance_points = lambda i, j: X[i, j]
        else:
            self.distance_points = lambda i, j: metric_points(
                X[i], X[j], **parameters_metric_points
            )

        # distance between clusters
        if metric_clusters == "average":
            self.distance_clusters = distances.average_dist
        elif metric_clusters == "min":
            self.distance_clusters = distances.min_dist
        elif metric_clusters == "max":
            self.distance_clusters = distances.max_dist
        elif metric_clusters == "emd":
            self.distance_clusters = distances.EMD_for_two_clusters
        else:
            raise ValueError(
                "the value {} is not a valid distance. Options are 'min', 'max', 'average', 'emd'".format(
                    metric_clusters
                )
            )

        # Creation of the ClusterGraph
        self.graph = nx.Graph()

        # one node for each cluster
        self.graph.add_nodes_from(
            [(i, dict(size=len(c), points_covered=c)) for i, c in enumerate(clusters)]
        )

        # compute all distances and add all edges
        self.graph.add_weighted_edges_from(
            [
                (i, j, self.distance_clusters(C_i, C_j, self.distance_points))
                for i, C_i in enumerate(clusters[:-1])
                for j, C_j in enumerate(clusters[i + 1 :], start=(i + 1))
            ],
            weight="label",
        )

        GraphPreprocess.__init__(self)
        self.graph_prepro = self.graph

        GraphPruning.__init__(self, 
            graph=self.graph,
            type_pruning = type_pruning, 
            algo =algo,
            weight = weight,
            knn_g = knn_g,  
            weight_knn_g = weight_knn_g, 
            k_compo = k_compo,
            dist_weight = dist_weight
        )
        self.original_graph=self.graph

    def get_graph(self):
        """
        Returns
        -------
        networkx.Graph
            Returns the ClusterGraph
        """
        return self.graph
    
