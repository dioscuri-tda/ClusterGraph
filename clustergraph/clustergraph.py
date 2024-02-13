import scipy.spatial.distance as sp
import numpy as np
import networkx as nx
from .utils import get_corresponding_edges
import matplotlib.pyplot as plt
from .distances import CreationDistances
from pyballmapper import BallMapper








class ClusterGraph:
    """_summary_
    """

    def __init__(self, clusters=None, X=None , sample = 1,
        metric_clusters="average",
        parameters_metric_clusters={},
        distance_points=None,
        data_preparation="usual",
        # Parameters connected with Distance_between_points
        metric_points=sp.euclidean,
        parameters_metric_points={},
        seed = None
        ) :
        """_summary_

        Parameters
        ----------
        clusters : _type_, optional
            _description_, by default None
        X : _type_, optional
            _description_, by default None
        sample : int, optional
            _description_, by default 1
        metric_clusters : str, optional
            _description_, by default "average"
        parameters_metric_clusters : dict, optional
            _description_, by default {}
        distance_points : _type_, optional
            _description_, by default None
        data_preparation : str, optional
            _description_, by default "usual"
        metric_points : _type_, optional
            _description_, by default sp.euclidean
        parameters_metric_points : dict, optional
            _description_, by default {}
        seed : _type_, optional
            _description_, by default None
        """

        if(clusters is not None) :
            if(type(clusters) is BallMapper ) :
                self.get_clusters_from = self.get_clusters_from_BM

            # working for every dictionary ?
            elif ( type(clusters) is dict   ) :
                self.get_clusters_from = self.get_clusters_from_Mapper
            #should be a list or np array how to make difference between a prediction and a good format (already given) clusters
            else :
                self.get_clusters_from = self.get_clusters_from_scikit

            clusters = self.get_clusters_from(clusters)



        # Get a DistancesBetweenClusters object
        self.distance_clusters = CreationDistances(  metric_clusters= metric_clusters,
        parameters_metric_clusters= parameters_metric_clusters ,  clusters= clusters , distance_points= distance_points , 
        data_preparation=data_preparation ,
        # Parameters connected with Distance_between_points
        metric_points= metric_points, parameters_metric_points= parameters_metric_points ,  X=X  ).get_distance_cluster()

        # Creation of the ClusterGraph 
        self.graph = nx.Graph()
        # List [ [node_1,node_2, length_edge_1] , ...]
        self.edges = []
        # List of integers which correspond to the numbered clusters
        self.vertices = []
        self.central_vertice = False
        self.farthest_vertice = False
        self.list_diameters = []
        self.my_graph = nx.Graph()

        if(sample < 1) :
            self.distance_clusters.subsample( sample, seed )
        


   
    def distances_clusters(self):
        """_summary_
        Method which compute the distances between clusters and creates ClusterGraph.

        Parameters
        ----------
        normalize : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        (
            self.vertices,
            self.edges,
            nb_points_by_clusters,
        ) = self.distance_clusters.compute_all_distances()

        self.update_graph(nb_points_by_clusters)
        self.ends_avg_cluster()

        print("Central Vertice", self.central_vertice)
        print("Farthest away Vertice", self.farthest_vertice)
        print("Vertices and size", nx.get_node_attributes(self.graph, "size"))

        return self.graph


    
    def update_graph(self, list_node_attributes=[], nb_edges=-1):
        self.graph.clear()
        self.graph_add_nodes_attributes(list_node_attributes)
        self.graph_add_edges(nb_edges=nb_edges)
        return self.graph

    
    def graph_add_edges(self, nb_edges=-1):
        if nb_edges < 0:
            nb_edges = len(self.edges)
        for i in range(nb_edges):
            self.graph.add_edge(
                self.edges[i][0], self.edges[i][1], label=self.edges[i][2]
            )
        return self.graph

  
    def graph_add_nodes_attributes(self, list_node_attributes):
        for i in list_node_attributes:
            self.graph.add_node(i[0], size=i[1], points_covered=i[2])

        return self.graph

    

    
    def ends_avg_cluster(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if len(self.edges) == 0:
            print("No edges saved")
            return -1, -1
        if len(self.vertices) == 0:
            print("No vertices saved")
            return -1, -1

        # distance min and max between two clusters because distances are ordered from shortest to longest
        mini = self.edges[-1][2]
        maxi = self.edges[0][2]
        node_mini = self.vertices[0]
        node_maxi = self.vertices[0]

        for i in self.vertices:
            compt = 0
            avg = 0
            for edge in self.edges:
                if edge[0] == i or edge[1] == i:
                    avg += edge[2]
                    compt += 1

            if compt != 0:
                test = avg / compt

            if compt > 0 and mini > test:
                mini = test
                node_mini = i
            if compt > 0 and test > maxi:
                maxi = test
                node_maxi = i
            self.central_vertice = node_mini
            self.farthest_vertice = node_maxi

        return node_mini, node_maxi



    # GET CLUSTERS IS THE GOOD FORMAT

    def get_clusters_from_scikit(self, prediction):
        """_summary_
        From a list of prediction returns a list of clusters with each cluster being a list of indices
        Parameters
        ----------
        prediction : _type_ list or numpy array
            _description_ List of clusters. At each index there is a label corresponding to the cluster of the data point. 

        Returns
        -------
        _type_ list
            _description_ Returns a list of clusters. Each element of the list is also a list in which all indices of the points coverd by this cluster are stored.
        """
        clusters = np.unique(prediction)
        print(clusters)
        dictionary = {}
        list_clusters = []
        for i in range(len(clusters)):
            dictionary[int(clusters[i])] = i 
            list_clusters.append([])

        count = 0
        for i in prediction:
            list_clusters[dictionary[int(i)] ].append(count)
            count += 1
        return list_clusters



    def get_clusters_from_BM(self, bm):
        """_summary_
        From a BallMapper object returns a list of clusters with each cluster being a list of indices corresponding to the points covered

        Parameters
        ----------
        bm : _type_ BallMapper
            _description_

        Returns
        -------
        _type_ list
            _description_ Returns a list of clusters. Each element of the list is also a list in which all indices of the points coverd by this cluster are stored.
        """
        clusters = list(bm.points_covered_by_landmarks)
        nb_clusters = len(clusters)
        list_clusters = []
        nb_nodes = 0
        list_clusters = []
        # Creation of the list for keys to be ordered
        for i in clusters:
            list_clusters.append([])

        for i in clusters:
            list_clusters[nb_nodes] = bm.points_covered_by_landmarks[i]
            nb_nodes += 1
        return list_clusters



    def get_clusters_from_Mapper(self, graph):
        """_summary_
        From a Mapper object returns a list of clusters with each cluster being a list of indices corresponding to the points covered 

        Parameters
        ----------
        graph : _type_
            _description_

        Returns
        -------
        _type_ list
            _description_ Returns a list of clusters. Each element of the list is also a list in which all indices of the points coverd by this cluster are stored.
        """
        clusters = list(graph["nodes"])
        nb_clusters = len(clusters)
        list_clusters = []
        nb_nodes = 0
        list_clusters = []
        # Creation of the list for keys to be ordered
        for i in clusters:
            list_clusters.append([])

        for i in graph["nodes"]:
            list_clusters[nb_nodes] = graph["nodes"][i]
            nb_nodes += 1
        return list_clusters
