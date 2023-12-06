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
    
    def __init__(self,  graph,  type_pruning = "conn" ,  algo ="bf", weight = "label" ,
                knn_g = None , X = None, sample = 1,  k_n = 2 ,   weight_knn_g = 'label', k_compo = 2, merged_knn = False, dist_weight = True ) :

        self.original_graph = graph
        self.pruned_graph = None

        if (type_pruning == "conn") :
            self.prunedStrategy =  ConnectivityPruning(algo = algo, weight=weight )
        
        elif( type_pruning == "md" ) :
            self.prunedStrategy =  Metric_distortion( graph =  graph, knn_g = knn_g , X = X, k_n = 2 ,   weight_knn_g = weight_knn_g , k_compo = k_compo , dist_weight = dist_weight)
        
    
    
    def prune(self, graph = None, nb_edge_pruned = -1, score = False ) :
        if(graph is None) :
            graph = self.original_graph

        return self.prunedStrategy.prune(  graph , nb_edge_pruned, score   )



        
            

                
 
    
    
    

        

