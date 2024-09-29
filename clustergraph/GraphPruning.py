from scipy.spatial.distance import euclidean
from networkx import add_path
import networkx as nx
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

from  .ConnectivityPruning import ConnectivityPruning
from .Metric_distortion_class import Metric_distortion



class GraphPruning :
    
    def __init__(self,  graph=None,  type_pruning = "conn" ,  algo ="bf", weight = "label" ,
                knn_g = None ,  weight_knn_g = 'label', k_compo = 2, dist_weight = True ) :
        """_summary_

        Parameters
        ----------
        graph : networkx.Graph, optional
            Graph to prune
        type_pruning : str in {"conn", "md"}, optional
            The type of pruning chosen. It can be "md" for the metric distortion pruning or "conn" for the connectivity pruning. The connectivity pruning returns a summary of the graph meanwhile the metric distortion pruning returns a graph which tends to be close to the shape of data, by default "conn"
        algo : str in {"bf","ps"}, optional
            Choice of the algorithm used to prune edges in the graph. “bf” correspond to the best and also the slowest algorithm (the brut force algorithm).
            “ps” is the quickest but does not ensure the best pruning, by default "bf"
        weight : str, optional
            The key underwhich the weight/size of edges is stored in the graph, by default "label"
        knn_g : networkx.Graph, optional
            The k-nearest neighbors graph from which the intrinsic distance between points of the dataset is retrieved. 
            The dataset should be the same than the one on which the “graph” was computed. It is mandatory when the "type_pruning" is "md", by default None
        weight_knn_g : str, optional
            Key/label underwhich the weight of edges is store in the “graph”. The weight corresponds to the distance between two nodes, by default 'label'
        k_compo : int, optional
            Number of edges that will be added to each disconnected component to merge them after the metric distortion pruning process.
            The edges added are edges which are connecting disconnected components and the shortest are picked, by default 2
        dist_weight : bool, optional
            If “dist_weight” is set to True, the distortion will be computed with weight on edges and it will not be the case if it is set to False, by default True
        """
        if graph != None :
            self.original_graph = graph

        self.pruned_graph = None
        self.merged_graph = None

        if (type_pruning == "conn") :
            self.prunedStrategy =  ConnectivityPruning(algo=algo, weight=weight )
        
        elif( type_pruning == "md" ) :
            self.prunedStrategy =  Metric_distortion( graph=self.original_graph, knn_g = knn_g ,  weight_knn_g = weight_knn_g , k_compo = k_compo , dist_weight = dist_weight, algo =algo)
        
    
    
    def prune(self, graph = None, nb_edge_pruned = -1, score = False ) :
        """_summary_
        Method which launch the pruning of the graph. It returns the pruned graph and the list of the evolution of the score if “score” is set to “True”. The score is the connectivity or the metric distortion depending on the type of pruning chosen.
        
        Parameters
        ----------
        graph : networkx.Graph, optional
            Graph to prune. If no graph is given, the one given at the initialization will be taken, by default None
        nb_edge_pruned : int, optional
            Maximum number of edges to prune. If "-1" is chosen, the algorithm will prune as many edges as possible, by default -1
        score : bool, optional
            The method will return the score if it is set to "True". The score is the connectivity or the metric distortion depending on the type of pruning chosen, by default False

        Returns
        -------
        networkx.Graph or networkx.Graph, list of float
            Returns the pruned graph and the list of score if chosen.
        """
        if(graph is None) :
            graph = self.original_graph
       
        
        if( score ) :
            self.pruned_graph, evolScore = self.prunedStrategy.prune(  graph , nb_edge_pruned , score   )
            return self.pruned_graph, evolScore
        else : 
            self.pruned_graph = self.prunedStrategy.prune( graph, nb_edge_pruned, score )
            return self.pruned_graph

    
    def merge_graph(self, pruned_gg = None,  nb_edges = -1 ) :
        """_summary_
        Method which after merging the disconnected components in the graph, prune a given number of edges (among the ones added by the merge) in order to get a less noisy graph.

        Parameters
        ----------
        pruned_gg : networkx Graph
            The graph which should be merged in order to get one connected component.
        nb_edges_pruned : int, optional
            The maximum number of edges which should be pruned after the merge. If the value is None, all possible edges will be pruned, by default None

        Returns
        -------
        networkx Graph
            Returns the merged and pruned graph.
        """
        if(pruned_gg is None) :
            pruned_gg = self.pruned_graph

        self.merged_graph = self.prunedStrategy.conn_prune_merged_graph(pruned_gg, nb_edges ).copy()
        return self.merged_graph



        
            

                
 
    
    
    

        

