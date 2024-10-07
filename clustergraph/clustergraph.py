import scipy.spatial.distance as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from . import distances
from .c_GraphPreprocess import GraphPreprocess
from .GraphPruning import GraphPruning
from sklearn.neighbors import NearestNeighbors
from pyballmapper import BallMapper


class ClusterGraph(GraphPreprocess, GraphPruning):

    def __init__(
        self,
        X,
        clusters,
        metric_clusters="centroids",
        # Parameters connected with Distance_between_points
        metric_points=sp.euclidean,
        parameters_metric_points={},
        type_pruning = "conn", 
        algo ="bf",
        weight = "weight",
        knn_g = None,  
        weight_knn_g = 'weight', 
        k_compo = 2,
        dist_weight = True 
    ):

        self.clusters = clusters
        self.is_knn_computed = -1
        self.X = X
        if knn_g is None or isinstance(knn_g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)) :
            self.knn_g = knn_g
            self.is_knn_computed = 0
            
        elif isinstance(knn_g, int) :
            neigh = NearestNeighbors(n_neighbors=knn_g, radius=1)
            neigh.fit(X=X)
            nn_adjacency = neigh.kneighbors_graph(X=X, n_neighbors=knn_g, mode="distance")
            nn_Graph = nx.from_scipy_sparse_array(nn_adjacency, edge_attribute=weight_knn_g)

            for node in nn_Graph.nodes:
                nn_Graph.remove_edge(node, node)
            self.knn_g = nn_Graph
            self.is_knn_computed = knn_g
        else :
            raise TypeError("The variable 'knn_g' must be None, an integer or a networkx Graph.")

        # distance between ids of datapoints
        if metric_points == "precomputed":
            self.distance_points = lambda i, j: X[i, j]

        elif metric_clusters == "centroids" :
            self.distance_points = lambda c_i, c_j: metric_points( np.mean(X[c_i], axis=0), np.mean(X[c_j], axis=0), 
                                                                  **parameters_metric_points )

        else:
            self.distance_points = lambda i, j: metric_points(
                X[i], X[j], **parameters_metric_points
            )

        # distance between clusters
        if metric_clusters == "centroids" :
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
            weight="weight",
        )

        GraphPreprocess.__init__(self)
        self.graph_prepro = self.graph

        GraphPruning.__init__(self, 
            graph=self.graph,
            type_pruning = type_pruning, 
            algo =algo,
            weight = weight,
            knn_g = self.knn_g,  
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
        if self.is_pruned != "not_pruned" :
            return self.graph
        else :
            pruned_graph = self.graph.copy()
            pruned_graph.remove_edges_from(self.prunedEdgesHistory[self.is_pruned]["edges"])
            return pruned_graph
        

    def prune_distortion(self,
                knn_g = 10 ,
                nb_edge_pruned = -1, 
                score = False,
                algo="bf",
                weight_knn_g = 'weight', 
                k_compo = 2, 
                dist_weight = True) :
        
        if isinstance(knn_g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)) :
            self.knn_g = knn_g
            self.is_knn_computed = -1
            
        elif isinstance(knn_g, int) :
            if self.is_knn_computed != knn_g :
                neigh = NearestNeighbors(n_neighbors=knn_g, radius=1)
                neigh.fit(X=self.X)
                nn_adjacency = neigh.kneighbors_graph(X=self.X, n_neighbors=knn_g, mode="distance")
                nn_Graph = nx.from_scipy_sparse_array(nn_adjacency, edge_attribute=weight_knn_g)

                for node in nn_Graph.nodes:
                    nn_Graph.remove_edge(node, node)
                self.knn_g = nn_Graph
                self.is_knn_computed = knn_g
        
        return self.prune_distortion_pr(knn_g=self.knn_g,
                nb_edge_pruned=nb_edge_pruned,
                score=score,
                algo=algo,
                weight_knn_g=weight_knn_g, 
                k_compo=k_compo,
                dist_weight=dist_weight,
                is_knn_computed=self.is_knn_computed )

        
    
