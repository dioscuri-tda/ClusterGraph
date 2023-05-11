import scipy.spatial.distance as sp
import numpy as np
import networkx as nx
from .utils import get_corresponding_edges
import matplotlib.pyplot as plt


def get_clusters_from_scikit(prediction):
    """_summary_

    Parameters
    ----------
    prediction : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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



def get_clusters_from_BM(bm):
    """_summary_

    Parameters
    ----------
    bm : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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



def get_clusters_from_Mapper(graph):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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




class ClusterGraph:
    """_summary_
    """

    def __init__(self, distance_clusters):
        """_summary_

        Parameters
        ----------
        distance_clusters : _type_
            _description_
        """
        self.distance_clusters = distance_clusters

        self.graph = nx.Graph()
        # List [ [node_1,node_2, length_edge_1] , ...]
        self.edges = []
        # List of integers which correspond to the numbered clusters
        self.vertices = []
        self.central_vertice = False
        self.farthest_vertice = False
        self.list_diameters = []
        self.my_graph = nx.Graph()


   
    def distances_clusters(self, normalize=True, verbose=False):
        """_summary_

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
        if normalize == True:
            self.normalize_max()
        self.ends_avg_cluster()

        if verbose:
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

    

    def normalize_max(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if self.edges == []:
            print("No edges computed")
            return []

        maxi = self.edges[-1][2]
        for i in self.edges:
            i[2] = i[2] / maxi

        self.graph_add_edges()
        return self.edges

  
    def normalize_edges_diameter(
        self,
        metric_clusters=None,
        distance_points=None,
        data_preparation=None,
        parameters_metric_clusters=None,
    ):
        """_summary_

        Parameters
        ----------
        metric_clusters : _type_, optional
            _description_, by default None
        distance_points : _type_, optional
            _description_, by default None
        data_preparation : _type_, optional
            _description_, by default None
        parameters_metric_clusters : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        self.list_diameters = self.distance_clusters.diameter_clusters(
            metric_clusters,
            distance_points,
            data_preparation,
            parameters_metric_clusters,
        )

        maxi_diameter = 0
        maxi_edge = self.edges[-1][2]

        for i in range(len(self.list_diameters)):
            if self.list_diameters[i][1] > maxi_diameter:
                maxi_diameter = self.list_diameters[i][1]

        for j in range(len(self.edges)):
            self.edges[j][2] = self.edges[j][2] / maxi_diameter

        self.graph.remove_edges_from(list(self.graph.edges()))
        self.normalize_max()
        return self.edges


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


    def draw_edges(self):
        """_summary_
        """
        nb = list(range(1, len(self.edges) + 1))

        lengths = []
        plt.close("all")
        for i in range(len(self.edges)):
            lengths.append(self.edges[i][2])

        plt.plot(nb, lengths)
        plt.show()



    def personnalize_graph(self, vertices=False, nb_edges=-1, edges=None):
        """_summary_

        Parameters
        ----------
        vertices : bool, optional
            _description_, by default False
        nb_edges : int, optional
            _description_, by default -1
        edges : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(vertices, np.ndarray) or isinstance(vertices, list):
            vertices = np.sort(vertices)

        self.my_graph = nx.Graph()

        # ADDING VERTICES
        # IF no vertices given we add them all
        if not (isinstance(vertices, np.ndarray) or isinstance(vertices, list)):
            self.my_graph.add_nodes_from(self.graph.nodes(data=True))
            vertices = self.graph.nodes

        else:
            # We add the wanted vertices
            for i in vertices:
                self.my_graph.add_node(i, **self.graph.nodes(data=True)[i])

        # ADDING EDGES
        # If no edges given we add
        if not (edges):
            edges = self.edges
            # Get ordered edges corresponding
            corres_edges = get_corresponding_edges(vertices, edges)

        # IF given edges we add them all if demanded
        else:
            # maybe here add a test to add only edges which connect, wanted vertices
            corres_edges = edges

        # IF EDGES NEGATIVE WE ADD ALL EDGES CORRESPONDING
        if nb_edges < 0:
            nb_edges = len(corres_edges)

        for i in range(nb_edges):
            self.my_graph.add_edge(
                corres_edges[i][0], corres_edges[i][1], label=corres_edges[i][2]
            )

        return self.my_graph





def filtration(graph, edges, label="label", precision=0.5):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    edges : _type_
        _description_
    label : str, optional
        _description_, by default "label"
    precision : float, optional
        _description_, by default 0.5

    Returns
    -------
    _type_
        _description_
    """
    g = nx.Graph()
    g.add_nodes_from(graph.nodes(data=True))

    for edge in edges:
        # We test if there exist a path
        try:
            shortest_path = nx.dijkstra_path_length(g, edge[0], edge[1], weight=label)
            error = precision * edge[2]

            # If there is a big difference between the two then we add
            if abs(shortest_path - edge[2]) > error:
                g.add_edge(edge[0], edge[1], label=edge[2])

        # If there is no existing path between the two nodes, we add the edge
        except nx.NetworkXNoPath:
            g.add_edge(edge[0], edge[1], label=edge[2])

    return g
