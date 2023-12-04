import scipy.spatial.distance as sp
from .utils import insert_sorted_list
import ot
import numpy as np



class Distances_between_points:
    """_summary_
    """
    def __init__(self, metric_points=sp.euclidean, X=None, parameters_metric_points={}):
        """_summary_

        Parameters
        ----------
        metric_points : _type_, optional
            _description_, by default sp.euclidean
        X : _type_, optional
            _description_, by default None
        parameters_metric_points : dict, optional
            _description_, by default {}
        """
        self.X = X
        if not (parameters_metric_points):
            self.parameters_metric_points = {}
        else:
            self.parameters_metric_points = parameters_metric_points

        # IF METRIC IS NOT CALLABLE THEN IT SHOULD BE A DISTANCE MATRIX
        if callable(metric_points):
            if X is None:
                print("Metric between points is callable and no dataset given")

            self.get_distance = metric_points
            self_distance_matrix_points = None

        else:
            self.get_distance = self.dist_matrix_points
            self.distance_matrix_points = metric_points

  

    def compute_distance_points(self, point_1, point_2):
        """_summary_

        Parameters
        ----------
        point_1 : _type_
            _description_
        point_2 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.get_distance(point_1, point_2, **self.parameters_metric_points)

 
    def dist_matrix_points(self, point_1, point_2):
        """_summary_

        Parameters
        ----------
        point_1 : _type_
            _description_
        point_2 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.distance_matrix_points[point_1][point_2]



class Distances_between_clusters:
    """_summary_
    """
    def __init__(
        self,
        metric_clusters="average",
        parameters_metric_clusters={},
        clusters=None,
        distance_points=None,
        data_preparation="usual",
    ):
        """_summary_

        Parameters
        ----------
        metric_clusters : str, optional
            _description_, by default "average"
        parameters_metric_clusters : dict, optional
            _description_, by default {}
        clusters : _type_, optional
            _description_, by default None
        distance_points : _type_, optional
            _description_, by default None
        data_preparation : str, optional
            _description_, by default "usual"

        Raises
        ------
        ValueError
            _description_
        TypeError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        

        self.get_clusters = self.get_clusters_identity

        self.distance_points = distance_points
        if not (parameters_metric_clusters):
            self.parameters_metric_clusters = {}
        else:
            self.parameters_metric_clusters = parameters_metric_clusters

            ########################################################
            # CHOICE OF THE DISTANCE FUNCTION BETWEEN TWO CLUSTERS #
            ########################################################

        # IF METRIC IS A FUNCTION WE STORE IT
        if callable(metric_clusters) or isinstance(metric_clusters, str):
            if clusters is None:
                raise ValueError(
                    "clusters can not be None if metric_clusters is not a distance matrix"
                )
            else:
                self.clusters = clusters

            if distance_points is None:
                raise TypeError(
                    "If metric_clusters is not a distance matrix, distance_points must be different from None"
                )

            self.distance_matrix_clusters = None

            # CHOICE OF GET_DISTANCE

            # If metric cluster is a string then we associate the good function
            if isinstance(metric_clusters, str):
                if metric_clusters == "min":
                    self.get_distance = self.min_dist
                elif metric_clusters == "max":
                    self.get_distance = self.max_dist

                elif metric_clusters == "emd":
                    self.get_distance = self.EMD_for_two_clusters

                elif metric_clusters == "avg_ind":
                    self.get_distance = self.mean_dist_indices

                else:
                    self.get_distance = self.mean_dist
                self.distance_matrix_clusters = None

            # If metric is a function we keep this function
            else:
                self.get_distance = metric_clusters

                # CHOICE OF THE DATA PREPARATION #

            # If the user has his own way to prepare data from two list of indices, we let the user use this one
            if callable(data_preparation):
                self.data_preparation = data_preparation

            # If the metric between clusters requires a distance matrix (between points) we store this method
            elif data_preparation == "dist_mat":
                self.data_preparation = self.get_submatrix

            elif data_preparation == "id":
                self.data_preparation = self.identity

            else:
                # If there is no dataset we will need to return positions of points to access distance matrix
                if distance_points.X is None:
                    self.data_preparation = self.identity

                else:
                    self.data_preparation = self.get_sets_points

        # If metric is a distance matrix we store it method to access the distances
        else:
            if metric_clusters.shape[0] != metric_clusters.shape[1]:
                raise ValueError(
                    "The distance matrix must have the same number of rows and columns"
                )
            self.get_distance = self.access_dist_matrix_clusters
            self.distance_matrix_clusters = metric_clusters
            self.data_preparation = self.identity
            self.parameters_metric_clusters = {}

            if not (clusters):
                nb_clusters = len(self.distance_matrix_clusters)
                self.clusters = np.array(list(range(nb_clusters))).reshape(
                    nb_clusters, 1
                )
            else:
                if len(clusters) != metric_clusters.shape[0]:
                    raise ValueError(
                        "The number of clusters must be the same than the number of columns and rows in the distance matrix"
                    )
                self.clusters = clusters



    def get_clusters_identity(self) :
        return self.clusters
    
    def get_subsampled_clusters(self) :
        return self.subsample_.subsampled_clusters
    


    def compute_distance_clusters(self, parameters):
        """_summary_

        Parameters
        ----------
        parameters : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.get_distance(*parameters, **self.parameters_metric_clusters)

        
    def mean_dist(self, X_1, X_2, distance_points=None):
        """_summary_

        Parameters
        ----------
        X_1 : _type_
            _description_
        X_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

        compt = 0
        avg = 0
        for i in range(len(X_1)):
            for j in range(len(X_2)):
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                avg += d
                compt += 1

        return avg / compt

   
    def min_dist(self, X_1, X_2, distance_points=None):
        """_summary_

        Parameters
        ----------
        X_1 : _type_
            _description_
        X_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

        mini = distance_points.compute_distance_points(X_1[0], X_2[0])
        compt = 0
        for i in range(len(X_1)):
            for j in range(len(X_2)):
                compt += 1
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                if mini > d:
                    mini = d
        return mini


    def max_dist(self, X_1, X_2, distance_points=None):
        """_summary_

        Parameters
        ----------
        X_1 : _type_
            _description_
        X_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

        maxi = distance_points.compute_distance_points(X_1[0], X_2[0])
        # compt = 0
        for i in range(len(X_1)):
            for j in range(len(X_2)):
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                if maxi < d:
                    maxi = d
        return maxi


    def EMD_for_two_clusters(self, X_1, X_2, distance_points=None, normalize=True):
        """_summary_

        Parameters
        ----------
        X_1 : _type_
            _description_
        X_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None
        normalize : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

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
                    d += weight * distance_points.compute_distance_points(
                        X_1[i], X_2[j]
                    )
                    compt += 1
        if not (normalize):
            compt = 1
        return d / compt


    def access_dist_matrix_clusters(self, cluster_1, cluster_2):
        """_summary_

        Parameters
        ----------
        cluster_1 : _type_
            _description_
        cluster_2 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.distance_matrix_clusters[cluster_1[0]][cluster_2[0]]

    def mean_dist_indices(self, X_1, X_2, distance_points=None):
        if not (distance_points):
            distance_points = self.distance_points

        compt = 0
        avg = 0
        for i in range(len(X_1)):
            for j in range(len(X_2)):
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                avg += d
                compt += 1

        return avg / compt
    

    
    

    def compute_all_distances(self):
        """_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if self.distance_matrix_clusters is None:
            clusters = self.get_clusters()
        else:
            nb_clusters = len(self.distance_matrix_clusters)
            clusters = np.array(list(range(nb_clusters))).reshape(nb_clusters, 1)

        keys = list(range(1, len(clusters) + 1))
        nb_clusters = keys[-1]
        edges = []
        nb_points_by_clusters = []

        if nb_clusters == 1:
            raise ValueError("Only one cluster given")

        for i in keys:
            C1 = clusters[i - 1]
            nb_points_by_clusters.append(
                [i, len(self.clusters[i - 1]), self.clusters[i - 1]]
            )

            for j in keys:
                if i < j:
                    C2 = clusters[j - 1]
                    distance_between_clusters = self.compute_distance_clusters(
                        self.data_preparation(C1, C2)
                    )
                    edges = insert_sorted_list(edges, [i, j, distance_between_clusters])

        return keys, edges, nb_points_by_clusters


    def get_submatrix(self, cluster_1, cluster_2, distance_points=None):
        """_summary_

        Parameters
        ----------
        cluster_1 : _type_
            _description_
        cluster_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

        return [distance_points.distance_matrix_points[cluster_1][:, cluster_2]]


    def get_sets_points(self, cluster_1, cluster_2, distance_points=None):
        """_summary_

        Parameters
        ----------
        cluster_1 : _type_
            _description_
        cluster_2 : _type_
            _description_
        distance_points : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if not (distance_points):
            distance_points = self.distance_points

        return [distance_points.X[cluster_1], distance_points.X[cluster_2]]

    def identity(self, cluster_1, cluster_2):
        return (cluster_1, cluster_2)

        
    def choose_distance(self, metric):
        """_summary_

        Parameters
        ----------
        metric : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # IF METRIC IS NOT CALLABLE THEN IT SHOULD BE A DISTANCE MATRIX
        if callable(metric):
            return metric

        # If metric cluster is a string then we associate the good function
        elif isinstance(metric_clusters, str):
            if metric_clusters == "min":
                return self.min_dist
            elif metric_clusters == "max":
                return self.max_dist

            elif metric_clusters == "emd":
                return self.EMD_for_two_clusters

            elif metric_clusters == "avg_ind":
                return self.mean_dist_indices

            else:
                return self.mean_dist

        # If metric is a distance matrix we store it method to access the distances
        else:
            return self.access_dist_matrix_clusters

    # -----------------------------------------------------------


    # Has sense only if we have list of cluster, not possible with distance matrix
    def diameter_clusters(
        self,
        metric_clusters=None,
        distance_points=None,
        data_preparation=None,
        parameters_metric_clusters={},
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
        parameters_metric_clusters : dict, optional
            _description_, by default {}

        Returns
        -------
        _type_
            _description_
        """
        # If not metric between clusters given then we use the initial one, otherwise we take the one fitting
        if not (metric_clusters):
            metric_clusters = self.get_distance
        else:
            metric_clusters = self.choose_distance(metric_clusters)

        # If no data_preparation for metric between object given then we use the initial one
        if not (data_preparation):
            data_preparation = self.data_preparation

        # If no distance_points object given then we use the initial one
        if not (distance_points):
            distance_points = self.distance_points

        if not (parameters_metric_clusters):
            parameters_metric_clusters = self.parameters_metric_clusters

        list_diameters = []

        for i in range(len(self.clusters)):
            d = metric_clusters(
                *data_preparation(self.clusters[i], self.clusters[i], distance_points),
                **parameters_metric_clusters
            )
            list_diameters.append([i + 1, d])

        return list_diameters





class Creation_distances:
    """_summary_
    """
    def __init__(
        self,
        # Parameters connected with Distance_between_clusters
        metric_clusters="average",
        parameters_metric_clusters={},
        clusters=None,
        distance_points=None,
        data_preparation="usual",
        # Parameters connected with Distance_between_points
        metric_points=sp.euclidean,
        parameters_metric_points={},
        X=None,
    ):
        """_summary_

        Parameters
        ----------
        metric_clusters : str, optional
            _description_, by default "average"
        parameters_metric_clusters : dict, optional
            _description_, by default {}
        clusters : _type_, optional
            _description_, by default None
        distance_points : _type_, optional
            _description_, by default None
        data_preparation : str, optional
            _description_, by default "usual"
        metric_points : _type_, optional
            _description_, by default sp.euclidean
        parameters_metric_points : dict, optional
            _description_, by default {}
        X : _type_, optional
            _description_, by default None
        """
        # if metric is a function then we need to create the object distance between points
        if callable(metric_clusters) or isinstance(metric_clusters, str):
            self.distance_points_ = Distances_between_points(
                metric_points=metric_points,
                parameters_metric_points=parameters_metric_points,
                X=X,
            )

        else:
            distance_points_ = None

        self.distance_clusters_ = Distances_between_clusters(
            metric_clusters=metric_clusters,
            parameters_metric_clusters=parameters_metric_clusters,
            clusters=clusters,
            distance_points=self.distance_points_,
            data_preparation=data_preparation,
        )

    def get_distance_cluster(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self.distance_clusters_
    
    
