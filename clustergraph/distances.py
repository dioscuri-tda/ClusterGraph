import scipy.spatial.distance as sp
from .utils import insert_sorted_list
import ot
import numpy as np
from .subsampling import Subsampling


def centroid_dist(X_1, X_2, distance_points):
    """
    Computes the distance between the centroids of two point sets.

    Parameters
    ----------
    X_1 : numpy.ndarray
        A dataset (array of points) representing the first cluster.
    X_2 : numpy.ndarray
        A dataset (array of points) representing the second cluster.
    distance_points : callable
        A function that calculates the distance between two points.

    Returns
    -------
    float
        The distance between the centroids of the two clusters, calculated using the provided `distance_points` function.
    """
    return distance_points(X_1, X_2)


def average_dist(X_1, X_2, distance_points):
    """
    Computes the average distance between all pairs of points from two point sets.

    Parameters
    ----------
    X_1 : numpy.ndarray
        A dataset (array of points) representing the first cluster.
    X_2 : numpy.ndarray
        A dataset (array of points) representing the second cluster.
    distance_points : callable
        A function that calculates the distance between two points.

    Returns
    -------
    float
        The average distance between all pairs of points from `X_1` and `X_2`, computed using the provided `distance_points` function.
    """
    return np.mean([distance_points(i, j) for i in X_1 for j in X_2])


def min_dist(X_1, X_2, distance_points):
    """
    Computes the minimum distance between any pair of points from two point sets.

    Parameters
    ----------
    X_1 : numpy.ndarray
        A dataset (array of points) representing the first cluster.
    X_2 : numpy.ndarray
        A dataset (array of points) representing the second cluster.
    distance_points : callable
        A function that calculates the distance between two points.

    Returns
    -------
    float
        The minimum distance between any pair of points from `X_1` and `X_2`, computed using the provided `distance_points` function.
    """
    return np.min([distance_points(i, j) for i in X_1 for j in X_2])


def max_dist(X_1, X_2, distance_points):
    """
    Computes the maximum distance between any pair of points from two point sets.

    Parameters
    ----------
    X_1 : numpy.ndarray
        A dataset (array of points) representing the first cluster.
    X_2 : numpy.ndarray
        A dataset (array of points) representing the second cluster.
    distance_points : callable
        A function that calculates the distance between two points.

    Returns
    -------
    float
        The maximum distance between any pair of points from `X_1` and `X_2`, computed using the provided `distance_points` function.
    """
    return np.max([distance_points(i, j) for i in X_1 for j in X_2])


def EMD_for_two_clusters(X_1, X_2, distance_points, normalize=True):
    """
    Computes the Earth Mover's Distance (EMD) between two clusters of points.

    This function computes the optimal transport (Earth Mover's Distance) between two sets of points,
    X_1 and X_2, using the `distance_points` function for measuring the distance between points.
    Optionally, the distance can be normalized by the number of distances computed.

    Parameters
    ----------
    X_1 : numpy.ndarray
        A dataset (array of points) representing the first cluster.
    X_2 : numpy.ndarray
        A dataset (array of points) representing the second cluster.
    distance_points : callable
        A function or object that calculates the distance between two points.
    normalize : bool, optional
        If True, the computed distance will be normalized by the number of distances evaluated.
        If False, no normalization is performed. Default is True.

    Returns
    -------
    float
        The Earth Mover's Distance (EMD) between the two clusters `X_1` and `X_2`. The result is
        normalized if `normalize` is True, otherwise it returns the raw distance.
    """
    # Compute the Earth Mover's Distance (EMD) using Optimal Transport (OT)
    EMD = ot.da.EMDTransport()
    weight_matrix = EMD.fit(Xs=X_1, Xt=X_2)
    # GET THE OPTIMIZED TRANSPORT OF DIRT FROM CLUSTER 1 TO CLUSTER 2
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
    if not normalize:
        compt = 1
    return d / compt
