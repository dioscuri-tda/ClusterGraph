import scipy.spatial.distance as sp
from .utils import insert_sorted_list
import ot
import numpy as np
from .subsampling import Subsampling


def centroid_dist(X_1, X_2, distance_points) :
    return distance_points(X_1, X_2)

def average_dist(X_1, X_2, distance_points):
    return np.mean([distance_points(i, j) for i in X_1 for j in X_2])

def min_dist(X_1, X_2, distance_points):
    return np.min([distance_points(i, j) for i in X_1 for j in X_2])


def max_dist(X_1, X_2, distance_points):
    return np.max([distance_points(i, j) for i in X_1 for j in X_2])


def EMD_for_two_clusters(X_1, X_2, distance_points, normalize=True):
    """_summary_
    Method which computes the Earth Mover distance between X_1 and X_2 two clusters.

    Parameters
    ----------
    X_1 :  numpy darray
         Dataset restricted to the indices of the first cluster
    X_2 :  numpy darray
         Dataset restricted to the indices of the second cluster
    distance_points : , optional
        , by default None
    normalize :  bool, optional
     If True the distance will be normalized by the number of distances computed, by default True

    Returns
    -------
     float
         returns the Eart Moving distance between X_1 and X_2
    """

    EMD = ot.da.EMDTransport()
    weight_matrix = EMD.fit(Xs=X_1, Xt=X_2)
    # GET THE OPTIMIZE TRANSPORT OF DIRT FROM CLUSTER 1 TO CLUSTER 2
    weight_matrix = EMD.coupling_

    row = weight_matrix.shape[0]
    col = weight_matrix.shape[1]
    d = 0
    compt = 0
    # FOR EACH DIRT MOVEMENT, WE MULTIPLY IT BY THE DISTANCE BETWEEN THE TWO POINTS
    for i in range(row):
        for j in range(col):
            weight = weight_matrix[i, j]
            if weight != 0:
                d += weight * distance_points.compute_distance_points(X_1[i], X_2[j])
                compt += 1
    if not (normalize):
        compt = 1
    return d / compt