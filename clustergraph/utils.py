#####################
# ANNEXES FUNCTIONS #
#####################

import numpy as np


def get_clusters_from_scikit(prediction):
    """_summary_
    From a list of prediction returns a list of clusters with each cluster being a list of indices
    Parameters
    ----------
    prediction :  list or numpy array
         Cluster labels. At each index there is a label corresponding to the cluster of the data point.
    Returns
    -------
     list
         Returns a list of clusters. Each element of the list is numpy array in which all indices of the points coverd by this cluster are stored.
    """

    return [np.where(prediction == clustNum)[0] for clustNum in np.unique(prediction)]


def get_clusters_from_BM(bm):
    """_summary_
    From a BallMapper object returns a list of clusters with each cluster being a list of indices corresponding to the points covered
    Parameters
    ----------
    bm :  BallMapper
        
    Returns
    -------
     list
         Returns a list of clusters. Each element of the list is also a list in which all indices of the points coverd by this cluster are stored.
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
    From a Mapper object returns a list of clusters with each cluster being a list of indices corresponding to the points covered
    Parameters
    ----------
    graph : 
        
    Returns
    -------
     list
         Returns a list of clusters. Each element of the list is also a list in which all indices of the points coverd by this cluster are stored.
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



def replace_in_array(list_1, list_2, arr, val):
    """Function which in a numpy darray replace the crossing positions values for the lines list_1 and columns list_2 and inverse (symetric change) , by the value wanted

    Parameters
    ----------
    list_1 : list or numpy.array
        the rows in which we want to change the value
    list_2 : list or numpy.array
        the columns in which we want to change the value
    arr : numpy.darray
        the darray that we want to modify
    val : float or int
        the value we want to be in those positions

    Returns
    -------
    numpy.darray
       The darray modified
    """
    for i in list_1:
        for j in list_2:
            arr[i, j] = val
            arr[j, i] = val
    return arr


def insert_sorted_list(liste, element_to_insert):
    """Function which inserts in a ordered list a new element. Each element has this form [keys_1, keys_2, value] and we order depending of on the 'value' element.
    Returns the ordered list with the new element

    Parameters
    ----------
    liste : list
        list of element each element is represented by a list [keys_1, keys_2, value], the list is already ordered based on the 'value'
    element_to_insert : list
       list as followed [keys_1, keys_2, value] that we want to insert in the list by keeping it ordered
    Returns
    -------
    list
        Returns the ordered list with the new element
    """
    if len(element_to_insert) < 3:
        raise ValueError("Element to insert has less than 3 elements")

    index = len(liste)
    if liste == []:
        return [element_to_insert]
    # Searching for the position
    for i in range(len(liste)):
        if liste[i][2] > element_to_insert[2]:
            index = i
            break
    if index == len(liste):
        liste = liste[:index] + [element_to_insert]
    else:
        liste = liste[:index] + [element_to_insert] + liste[index:]

    return liste



def get_values(list_key_value):
    """_summary_

    Parameters
    ----------
    list_key_value : 
        

    Returns
    -------
    
        

    Raises
    ------
    ValueError
        
    """
    if list_key_value == []:
        raise ValueError("List is empty")
    values = []
    for i in list_key_value:
        values.append(i[1])
    return values





def get_sorted_edges(graph, variable_length="label"):
    """_summary_

    Parameters
    ----------
    graph : 
        
    variable_length : str, optional
        , by default "label"

    Returns
    -------
    
        
    """
    edges = []
    for edge in graph.edges:
        edges = insert_sorted_list(
            edges, [edge[0], edge[1], graph.edges[edge][variable_length]]
        )

    return edges



def get_corresponding_edges(vertices, edges):
    """_summary_

    Parameters
    ----------
    vertices : 
        
    edges : 
        

    Returns
    -------
    
        
    """
    corres_edges = []
    for edge in edges:
        if edge[0] in vertices and edge[1] in vertices:
            corres_edges = insert_sorted_list(corres_edges, edge)
    return corres_edges




def max_size_node_graph(graph, variable, nodes=None):
    """_summary_

    Parameters
    ----------
    graph : 
        
    variable : 
        
    nodes : , optional
        , by default None

    Returns
    -------
    
        
    """
    if not (nodes):
        nodes = graph.nodes
    maxi = 0
    for node in nodes:
        # print("NODE", node)
        # print("TEST", graph.nodes[node][variable])
        size = len(graph.nodes[node][variable])
        if size > maxi:
            maxi = size
    return maxi
