from scipy.spatial.distance import euclidean
from networkx import add_path
import networkx as nx
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random

from  .ConnectivityPruning import ConnectivityPruning
from .Metric_distortion_class import Metric_distortion



class GraphPruning :
    
    def __init__(self,  graph=None,  type_pruning = None,  algo ="bf", weight = "weight" ,
                knn_g = None,  weight_knn_g = 'weight', k_compo = 2, dist_weight = True ) :
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
            The key underwhich the weight/size of edges is stored in the graph, by default "weight"
        knn_g : networkx.Graph, optional
            The k-nearest neighbors graph from which the intrinsic distance between points of the dataset is retrieved. 
            The dataset should be the same than the one on which the “graph” was computed. It is mandatory when the "type_pruning" is "md", by default None
        weight_knn_g : str, optional
            Key/Weight underwhich the weight of edges is store in the “graph”. The weight corresponds to the distance between two nodes, by default 'weight'
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
        self.is_pruned = "not_pruned"

        self.prunedEdgesHistory={"md_bf" :{"all_pruned":False, "edges":[], "score":[],"knn_g":-1},
                                 "md_ps":{"all_pruned":False, "edges":[], "score":[], "knn_g":-1},
                                 "conn_bf":{"all_pruned":False, "edges":[], "score":[]},
                                 "conn_ps":{"all_pruned":False, "edges":[], "score":[]},
                                 "in_between_compo":{"edges":[]},
                                 "conn_merged":{"all_pruned":False, "edges":[], "score":[], 'k_compo': -1, "other_edges_remove":[]}
                                 }
        if not(graph is None) :
            if (type_pruning == "conn"):
                self.prunedStrategy =  ConnectivityPruning(algo=algo, weight=weight)
            
            elif(type_pruning == "md"):
                self.prunedStrategy =  Metric_distortion( graph=self.original_graph,
                                                        knn_g = knn_g,  
                                                        weight_knn_g = weight_knn_g, 
                                                        k_compo = k_compo, 
                                                        dist_weight = dist_weight, 
                                                        algo =algo)
        
    def prune(self, graph = None, nb_edge_pruned = -1, score = False) :
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
        

    
    def merge_graph_draft(self, pruned_gg = None,  nb_edges = -1 ) :
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
    
    def prune_distortion_pr(self,
                knn_g,
                nb_edge_pruned = -1, 
                score = False,
                algo="bf",
                weight_knn_g = 'weight', 
                k_compo = 2, 
                dist_weight = True,
                is_knn_computed = -1) :
        if (algo!="bf" and algo!="ps") :
            raise ValueError("The algorithm can only be 'bf' or 'ps'.")
        
        self.is_pruned = "md_"+algo
        
        if( 
            (self.prunedEdgesHistory[self.is_pruned]["all_pruned"] and 
             self.prunedEdgesHistory[self.is_pruned]["knn_g"] == is_knn_computed  ) 
           or 
           ( nb_edge_pruned > 0 and len(self.prunedEdgesHistory[self.is_pruned]["edges"]) >= nb_edge_pruned
             ) ) :
            pruned_graph = self.original_graph.copy()
            if nb_edge_pruned == -1 :
                nb_edge_pruned = len(self.prunedEdgesHistory[self.is_pruned]["edges"])
            pruned_graph.remove_edges_from(self.prunedEdgesHistory[self.is_pruned]["edges"][:nb_edge_pruned])
            pruned_graph.remove_edges_from(self.prunedEdgesHistory["in_between_compo"]["edges"])

            if score :
                return pruned_graph, self.prunedEdgesHistory[self.is_pruned]["score"][ :nb_edge_pruned]
            else :
                return pruned_graph
            
        self.prunedEdgesHistory[self.is_pruned]["knn_g"] = is_knn_computed   
        self.prunedMetricDistortionStrategy=Metric_distortion( 
                                            graph=self.original_graph,
                                            knn_g=knn_g,  
                                            weight_knn_g=weight_knn_g, 
                                            k_compo=k_compo, 
                                            dist_weight=dist_weight, 
                                            algo=algo)
        
        if nb_edge_pruned == -1 :
            self.prunedEdgesHistory[self.is_pruned]["all_pruned"]=True

        pruned_graph, removed_edges, evolScore = self.prunedMetricDistortionStrategy.prune(self.original_graph, nb_edge_pruned, True)
        self.prunedEdgesHistory[self.is_pruned]["edges"]=deepcopy(removed_edges)
        self.prunedEdgesHistory[self.is_pruned]["score"]=evolScore
        self.prunedEdgesHistory["in_between_compo"]["edges"]=[ (e[0],e[1]) for e in self.prunedMetricDistortionStrategy.edges_between_compo]
        if score :
            return pruned_graph, evolScore
        else : 
            return pruned_graph
    



    def prune_conn(self,
                nb_edge_pruned = -1, 
                score=False,
                algo="bf",
                weight="weight"
                ):
        if (algo!="bf" and algo!="ps") :
            raise ValueError("The algorithm can only be 'bf' or 'ps'.")
        
        self.is_pruned = "conn_"+algo
        
        if( self.prunedEdgesHistory[self.is_pruned]["all_pruned"] or 
           ( nb_edge_pruned > 0 and len(self.prunedEdgesHistory[self.is_pruned]["edges"]) >= nb_edge_pruned
             ) ) :
            pruned_graph = self.original_graph.copy()
            pruned_graph.remove_edges_from(self.prunedEdgesHistory[self.is_pruned]["edges"][:nb_edge_pruned])
            if score :
                return pruned_graph, self.prunedEdgesHistory[self.is_pruned]["score"][ :nb_edge_pruned]
            else :
                return pruned_graph
            
        self.prunedConnectivityStrategy=ConnectivityPruning(algo=algo, weight=weight)

        if nb_edge_pruned == -1 :
            self.prunedEdgesHistory[self.is_pruned]["all_pruned"]=True

        if score :
            self.pruned_graph, removed_edges, evolScore = self.prunedConnectivityStrategy.prune(self.original_graph, nb_edge_pruned, True)
            self.prunedEdgesHistory[self.is_pruned]["edges"]=removed_edges
            self.prunedEdgesHistory[self.is_pruned]["score"]=evolScore
            return self.pruned_graph, evolScore
        else : 
            pruned_graph, removed_edges, evolScore = self.prunedConnectivityStrategy.prune(self.original_graph, nb_edge_pruned, True)
            self.prunedEdgesHistory[self.is_pruned]["edges"]=removed_edges
            self.prunedEdgesHistory[self.is_pruned]["score"]=evolScore
            return pruned_graph
        



    def merge_graph(self, nb_edges = -1, k_compo = 2, score=False) :
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
        merged_graph = self.original_graph.copy()
        pruning = None
        if self.prunedEdgesHistory["md_bf"]["all_pruned"] :
            pruning = "md_bf"
        elif self.prunedEdgesHistory["md_ps"]["all_pruned"] :
            pruning = "md_ps"
        else :
            raise ValueError("The metric distortion pruning has to be launched first.")
        
        if self.prunedEdgesHistory["conn_merged"]['all_pruned'] and k_compo == self.prunedEdgesHistory["conn_merged"]['k_compo'] :
            merged_graph = self.get_merged_graph(pruning, nb_edges)
            conn_score = self.prunedEdgesHistory["conn_merged"]['score']
        
        else :
            merged_graph.remove_edges_from(self.prunedEdgesHistory[pruning]["edges"])
            merged_graph.remove_edges_from(self.prunedEdgesHistory["in_between_compo"]["edges"])
            g, removed_edges, conn_score = self.prunedMetricDistortionStrategy.conn_prune_merged_graph(merged_graph, None, k_compo)
            self.prunedEdgesHistory["conn_merged"]['all_pruned'] = True
            self.prunedEdgesHistory["conn_merged"]['edges'] = deepcopy(removed_edges)
            between_not_current = []
            for e in self.prunedEdgesHistory["in_between_compo"]["edges"] :
                if not (e in removed_edges or (e[1], e[0]) in removed_edges) :
                    between_not_current.append(e)

            self.prunedEdgesHistory["conn_merged"]["other_edges_remove"] = deepcopy(between_not_current)
            self.prunedEdgesHistory["conn_merged"]['score'] = conn_score
            self.prunedEdgesHistory["conn_merged"]['k_compo'] = k_compo
            merged_graph = self.get_merged_graph(pruning, nb_edges)
        if score :
            return merged_graph, conn_score
        else :
            return merged_graph
    
    def get_merged_graph(self, key, nb_edges) :
        merged_graph = self.original_graph.copy()
        edges_to_prune = deepcopy(self.prunedEdgesHistory[key]["edges"])
        #edges_to_prune.extend(deepcopy(self.prunedEdgesHistory["in_between_compo"]["edges"]))
        if nb_edges == -1 :
            nb_edges = len(self.prunedEdgesHistory["conn_merged"]['edges'])
        
        edges_to_prune.extend( deepcopy(self.prunedEdgesHistory["conn_merged"]['edges'][:nb_edges])  )
        edges_to_prune.extend( deepcopy(self.prunedEdgesHistory["conn_merged"]["other_edges_remove"])  ) 

        merged_graph.remove_edges_from(edges_to_prune)
        return merged_graph



        
            

                
 
    
    
    

        

