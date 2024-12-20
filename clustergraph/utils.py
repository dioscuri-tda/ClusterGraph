import numpy as np

#####################
# ANNEXES FUNCTIONS #
#####################


def get_clusters_from_scikit(prediction, return_mapping=False):
    """
    From a list of predictions, returns a list of clusters with each cluster being a list of indices.

    Parameters
    ----------
    prediction : list or numpy.ndarray
        Cluster labels. At each index there is a label corresponding to the cluster of the data point.
    return_mapping : bool, optional
        If True, returns a dictionary mapping each cluster label to its index. Default is False.

    Returns
    -------
    list
        A list of clusters, where each element is a numpy array containing the indices of the data points in that cluster.
    dict, optional
        If `return_mapping` is True, a dictionary mapping each cluster label to an index.
    """
    unique_labels = np.unique(prediction)

    if return_mapping:
        return [np.where(prediction == clustNum)[0] for clustNum in unique_labels], {
            i: u for i, u in enumerate(unique_labels)
        }
    else:
        return [np.where(prediction == clustNum)[0] for clustNum in unique_labels]


def get_clusters_from_BM(bm):
    """
    From a BallMapper object, returns a list of clusters, where each cluster is a list of indices corresponding to the points covered.

    Parameters
    ----------
    bm : BallMapper
        A BallMapper object which contains information about the clusters.

    Returns
    -------
    list
        A list of clusters, where each element is a list of indices corresponding to the points covered by that cluster.
    """
    clusters = list(bm.points_covered_by_landmarks)
    nb_clusters = len(clusters)
    list_clusters = []
    nb_nodes = 0

    # Creation of the list for keys to be ordered
    for i in clusters:
        list_clusters.append([])

    for i in clusters:
        list_clusters[nb_nodes] = bm.points_covered_by_landmarks[i]
        nb_nodes += 1
    return list_clusters


def get_clusters_from_Mapper(graph):
    """
    From a Mapper object, returns a list of clusters, where each cluster is a list of indices corresponding to the points covered.

    Parameters
    ----------
    graph : dict
        A Mapper object which contains the cluster information.

    Returns
    -------
    list
        A list of clusters, where each element is a list of indices corresponding to the points covered by that cluster.
    """
    clusters = list(graph["nodes"])
    nb_clusters = len(clusters)
    list_clusters = []
    nb_nodes = 0

    # Creation of the list for keys to be ordered
    for i in clusters:
        list_clusters.append([])

    for i in graph["nodes"]:
        list_clusters[nb_nodes] = graph["nodes"][i]
        nb_nodes += 1
    return list_clusters


def replace_in_array(list_1, list_2, arr, val):
    """
    Replaces the values in a numpy array at the positions specified by `list_1` and `list_2` (and their symmetric positions) with the given value.

    Parameters
    ----------
    list_1 : list or numpy.ndarray
        The rows in which we want to change the value.
    list_2 : list or numpy.ndarray
        The columns in which we want to change the value.
    arr : numpy.ndarray
        The numpy array to modify.
    val : float or int
        The value to place at the specified positions.

    Returns
    -------
    numpy.ndarray
        The modified numpy array.
    """
    for i in list_1:
        for j in list_2:
            arr[i, j] = val
            arr[j, i] = val
    return arr


def insert_sorted_list(liste, element_to_insert):
    """
    Inserts an element into an already sorted list based on the 'value' element (the third element in the list).

    Parameters
    ----------
    liste : list
        A list of elements, each represented by a list `[key_1, key_2, value]`. The list is already sorted based on the 'value'.
    element_to_insert : list
        A list `[key_1, key_2, value]` that we want to insert in the list while maintaining the order based on 'value'.

    Returns
    -------
    list
        The ordered list with the new element inserted.

    Raises
    ------
    ValueError
        If `element_to_insert` contains fewer than 3 elements.
    """
    if len(element_to_insert) < 3:
        raise ValueError("Element to insert has less than 3 elements")

    index = len(liste)
    if liste == []:
        return [element_to_insert]

    # Searching for the position to insert
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
    """
    Extracts the values from a list of key-value pairs.

    Parameters
    ----------
    list_key_value : list
        A list of key-value pairs, where each element is a list `[key, value]`.

    Returns
    -------
    list
        A list of values extracted from the input list of key-value pairs.

    Raises
    ------
    ValueError
        If the input list is empty.
    """
    if list_key_value == []:
        raise ValueError("List is empty")

    values = [i[1] for i in list_key_value]
    return values


def get_sorted_edges(graph, variable_length="label"):
    """
    Returns the edges of the graph sorted by the specified edge attribute.

    Parameters
    ----------
    graph : networkx.Graph
        A NetworkX graph object.
    variable_length : str, optional
        The attribute used for sorting the edges. Default is "label".

    Returns
    -------
    list
        A list of edges sorted by the specified attribute.
    """
    edges = []
    for edge in graph.edges:
        edges = insert_sorted_list(
            edges, [edge[0], edge[1], graph.edges[edge][variable_length]]
        )

    return edges


def get_corresponding_edges(vertices, edges):
    """
    Returns the edges that correspond to a given set of vertices.

    Parameters
    ----------
    vertices : list
        A list of vertices.
    edges : list
        A list of edges, where each edge is represented as `[vertex_1, vertex_2, value]`.

    Returns
    -------
    list
        A list of edges where both vertices are in the given list of vertices.
    """
    corres_edges = []
    for edge in edges:
        if edge[0] in vertices and edge[1] in vertices:
            corres_edges = insert_sorted_list(corres_edges, edge)
    return corres_edges


def max_size_node_graph(graph, variable, nodes=None):
    """
    Returns the maximum size of a node based on a given attribute in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        A NetworkX graph object.
    variable : str
        The attribute of the node that is used to determine the size.
    nodes : list, optional
        A list of nodes to check. If None, all nodes in the graph are checked.

    Returns
    -------
    int
        The maximum size of the node based on the given attribute.
    """
    if nodes is None:
        nodes = graph.nodes

    maxi = 0
    for node in nodes:
        size = len(graph.nodes[node][variable])
        if size > maxi:
            maxi = size
    return maxi
