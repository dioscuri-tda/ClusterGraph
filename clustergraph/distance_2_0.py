import scipy.spatial.distance as sp
from annexe_functions import insert_sorted_list
import ot
import numpy as np

                                                        ###########################
                                                        # DISTANCE BETWEEN POINTS #
                                                        ###########################


"""
Class which let the user get the distance between two points in a flexible way from distance matrix or
a dataset X.


metric_points : is a function or distance matrix
X : array if we need a dataset
parameters_metric_points : dictionary with the extra parameters if wished for the given metric
"""
class Distances_between_points() :
    
    def __init__(self, metric_points =  sp.euclidean, X = None, parameters_metric_points = {} ) :
        
        self.X = X
        if(not(parameters_metric_points)) :
            self.parameters_metric_points = {}
        else :
            self.parameters_metric_points = parameters_metric_points

        
        # IF METRIC IS NOT CALLABLE THEN IT SHOULD BE A DISTANCE MATRIX
        if( callable(metric_points) ) :
            if(X is None) :
                print("Metric between points is callable and no dataset given")
                
            self.get_distance  = metric_points
            self_distance_matrix_points = None
  
        else :
            self.get_distance = self.dist_matrix_points
            self.distance_matrix_points = metric_points
            
       
    """
    Compute the distance based on two points and add the given parameters 
    Returns a float which is the distance between two points
    
    """
    def compute_distance_points(self, point_1, point_2) : 
        return self.get_distance(point_1, point_2, **self.parameters_metric_points)
        
            

    """
    Function which returns the distance between two points in a distance matrix
    """
    def dist_matrix_points(self, point_1, point_2) :
        return self.distance_matrix_points[point_1][point_2]
    
    
    """
    Function which returns the distance between two points in a distance matrix
    def dist_points_(self, point_1, point_2) :
        return self.distance_matrix_points[point_1][point_2]
    """
    
        
        
        
        
        
        
        
#-------------------------------------------------------------------------------------------------------------------------------------------        
        
    
    
    
                                                #############################
                                                # DISTANCE BETWEEN CLUSTERS #
                                                #############################
                
                
                

        
""" 
Class which lets the user compute the distances between the given clusters


metric_clusters: is a function callable, a string or an array for a distance matrix

clusters : list of clusters, each cluster represent a list of indices of the points covered in the dataset

data_preparation : string or function , information to choose which function will prepare the data to compute distances between two clusters
those functions must accept two list of indices describing two clusters and must return the first parameters expected for the distance chosen to compute

parameters_metric_clusters : dictionary with the wished parameters for the given metric

distance_points : Distance_between_points object which lets Distance_between_clusters access to the distance between two points if needed.


"""
class Distances_between_clusters() :
    
    
    def __init__(self, metric_clusters = 'average', parameters_metric_clusters = {}, clusters = None
                , distance_points= None, data_preparation = 'usual' ) :
        

        """
        attributes :
        - parameters_metric_clusters
        - get_distance_clusters
        - distance_matrix_clusters
        - data_preparation
        -clusters
        - distance_points
        """
        
        self.distance_points = distance_points
        if(not(parameters_metric_clusters)) :
            self.parameters_metric_clusters = {}
        else :
            self.parameters_metric_clusters = parameters_metric_clusters
        
        
        
        
                                    ########################################################
                                    # CHOICE OF THE DISTANCE FUNCTION BETWEEN TWO CLUSTERS #
                                    ########################################################
                    
                    
        
        # IF METRIC IS A FUNCTION WE STORE IT
        if( callable(metric_clusters)  or isinstance(metric_clusters, str)  ) :
            if( clusters is None) :
                raise ValueError("clusters can not be None if metric_clusters is not a distance matrix")
            else :
                self.clusters = clusters
                
            if(distance_points is None) :
                raise TypeError("If metric_clusters is not a distance matrix, distance_points must be different from None")
            
            self.distance_matrix_clusters = None
            
            
                                            # CHOICE OF GET_DISTANCE#
            
            # If metric cluster is a string then we associate the good function
            if(isinstance(metric_clusters, str) ) :
                if( metric_clusters == 'min' ) :
                    self.get_distance = self.min_dist
                elif( metric_clusters == 'max' ) :
                    self.get_distance = self.max_dist

                elif( metric_clusters == 'emd' ) :
                    self.get_distance = self.EMD_for_two_clusters
                    
                elif( metric_clusters == 'avg_ind' ) :
                    self.get_distance = self.mean_dist_indices

                else :
                    self.get_distance = self.mean_dist
                self.distance_matrix_clusters = None
            
            # If metric is a function we keep this function
            else :
                self.get_distance  = metric_clusters
            
            
                                            # CHOICE OF THE DATA PREPARATION #
            
            # If the user has his own way to prepare data from two list of indices, we let the user use this one                
            if(callable(data_preparation) ) :
                self.data_preparation = data_preparation

            # If the metric between clusters requires a distance matrix (between points) we store this method                
            elif( data_preparation == 'dist_mat' ) :
                self.data_preparation = self.get_submatrix
                
            elif( data_preparation == 'id' ) :
                self.data_preparation = self.identity

            else :
                # If there is no dataset we will need to return positions of points to access distance matrix
                if(distance_points.X is None ) :
                    self.data_preparation = self.identity
                    
                else :
                    self.data_preparation = self.get_sets_points
            
            
                
    
    
        # If metric is a distance matrix we store it method to access the distances
        else :
            
            if( metric_clusters.shape[0] != metric_clusters.shape[1])  :
                    raise ValueError("The distance matrix must have the same number of rows and columns")
            self.get_distance = self.access_dist_matrix_clusters
            self.distance_matrix_clusters = metric_clusters
            self.data_preparation = self.identity
            self.parameters_metric_clusters = {}
            
            if( not(clusters) ) :
                nb_clusters = len(self.distance_matrix_clusters)
                self.clusters = np.array(list(range(nb_clusters) ) ).reshape(nb_clusters,1)
            else :
                if( len(clusters) != metric_clusters.shape[0])  :
                    raise ValueError("The number of clusters must be the same than the number of columns and rows in the distance matrix")
                self.clusters = clusters
                
        
        
            
            
       
                
                
        
       
    """
    Compute the distance based on two clusters and the given parameters for this function
    Returns a float which is the distance between two clusters
    parameters : list of the first parameters that will be used can be cluster_1, cluster_2, can be only a distance matrix
    the only constraint is that it must be in the same order than the metric chosen between clusters

    """
    def compute_distance_clusters(self, parameters) :
        return self.get_distance( *parameters, **self.parameters_metric_clusters)
           
        
        
        
        
        
                                            ##################################
                                            # DISTANCES BETWEEN TWO CLUSTERS #
                                            ##################################
                                                ##########################
                                                # POSSIBLE GET_DISTANCES #
                                                ##########################

                    
    """
    From two clusters this functions compute the average distance between points belonging to cluster 1 and points belonging to cluster 2
    Returns a float which is the average distance betweent the two clusters
    Parameters :
    -as described above 
    """
    def mean_dist(self, X_1, X_2, distance_points = None ) :
        if(not(distance_points) ) :
            distance_points = self.distance_points
            
        compt = 0
        avg = 0
        for i in range(len(X_1) ):
            for j in range(len(X_2) ) :
                d = distance_points.compute_distance_points(X_1[i],X_2[j])
                avg+=d
                compt +=1

        return avg/compt

    """
    From two clusters this functions compute the minimum distance between points belonging to cluster 1 and points belonging to cluster 2
    Returns a float which is the minimum distance between the two clusters

    Parameters :
    -as described above 
    """
    def min_dist(self, X_1, X_2 , distance_points = None) :
        if(not(distance_points) ) :
            distance_points = self.distance_points
            
        mini = distance_points.compute_distance_points( X_1[0], X_2[0])
        compt = 0
        for i in range(len(X_1) ):
            for j in range(len(X_2) ) :
                compt +=1
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                if(mini > d) :
                    mini= d
        return mini


    """
    From two clusters this functions compute the maximum distance between points belonging to cluster 1 and points belonging to cluster 2

    Returns a float which is the maximum distance between the two clusters

    Parameters :
    -as described above 
    """
    def max_dist(self, X_1, X_2, distance_points = None) :
        if(not(distance_points) ) :
            distance_points = self.distance_points
            
        maxi = distance_points.compute_distance_points(X_1[0], X_2[0])
        #compt = 0
        for i in range(len(X_1) ):
            for j in range(len(X_2) ) :
                d = distance_points.compute_distance_points(X_1[i], X_2[j])
                if(maxi < d) :
                    maxi= d         
        return maxi


    """
    From two clusters this functions compute the Earth Mover distance between points belonging to cluster 1 and points belonging to 
    cluster 2

    Returns a float which is the maximum distance between the two clusters

    Parameters :
    - normalize : boolean, if true then we will divide the cost by the number of movements required for the transport, 
    if false we just return 
    the sum of dirt moved * distance between points
    -as described above 
    """
    def EMD_for_two_clusters(self, X_1, X_2, distance_points = None,  normalize= True) :
        if(not(distance_points) ) :
            distance_points = self.distance_points
        
        
        EMD = ot.da.EMDTransport()
        weight_matrix = EMD.fit( Xs = X_1, Xt= X_2) 
        # GET THE OPTIMIZE TRANSPORT OF DIRT FROM CLUSTER 1 TO CLUSTER 2 
        weight_matrix = EMD.coupling_

        row = weight_matrix.shape[0]
        col = weight_matrix.shape[1]
        d =0
        compt = 0
        # FOR EACH DIRT MOVEMENT, WE MULTIPLY IT BY THE DISTANCE BETWEEN THE TWO POINTS
        for i in range(row) :
            for j in range(col) :
                weight = weight_matrix[i,j]
                if(weight !=0 ) :
                    d += weight * distance_points.compute_distance_points(X_1[i], X_2[j])
                    compt+=1
        if(not(normalize) ) :
            compt = 1
        return d/compt
    
    
    
    
    """
    Function which returns the distance between two clusters in a distance matrix
    """
    def access_dist_matrix_clusters(self, cluster_1 , cluster_2) :
        return self.distance_matrix_clusters[ cluster_1[0] ][ cluster_2[0]  ]
    
    
    
    """
    From two indices of clusters this functions compute the average distance between points belonging to cluster 1 and points belonging to cluster 2. Here the distance only requires the indices to be computed
    Returns a float which is the average distance betweent the two clusters
    Parameters :
    -as described above 
    """
    def mean_dist_indices(self, X_1, X_2, distance_points = None ) :
        if(not(distance_points) ) :
            distance_points = self.distance_points
            
        compt = 0
        avg = 0
        for i in range(len(X_1) ):
            for j in range(len(X_2) ) :
                d = distance_points.compute_distance_points(X_1[i],X_2[j])
                avg+=d
                compt +=1

        return avg/compt
        
        

                    
                    
            
                                            ##################################
                                            # DISTANCES BETWEEN ALL CLUSTERS #
                                            ##################################        
        
        
    """
    Compute the distance between all clusters
    Returns :
    - keys : a list of integers which is the list of clusters's labels
    - edges : list of ordered edges , each edge is a list as followed [cluster_label_1, cluster_label_2, edge_length]
    they are ordered depending on their length
    """
    def compute_all_distances(self ) :
        if(self.distance_matrix_clusters is None ) :
            clusters = self.clusters
        else :
            nb_clusters = len(self.distance_matrix_clusters)
            clusters = np.array(list(range(nb_clusters) ) ).reshape(nb_clusters,1) 
            
            
        keys = list(range(1, len(clusters)+1))
        nb_clusters = keys[-1]
        edges = []
        nb_points_by_clusters = []

        if(nb_clusters ==1) :
            raise ValueError("Only one cluster given")

        for i in keys :
            C1 = clusters[i-1]
            nb_points_by_clusters.append([i, len(self.clusters[i-1]) , self.clusters[i-1] ])

            for j in keys :
                if(i < j ) :
                    C2 = clusters[j-1]
                    distance_between_clusters = self.compute_distance_clusters(self.data_preparation(C1,C2) )
                    edges = insert_sorted_list( edges, [i,j,distance_between_clusters] )

        return keys, edges, nb_points_by_clusters
        
        
        
        
        
        
                                        #####################################################
                                        # PREPARE DATA TO COMPUTE DISTANCE BETWEEN CLUSTERS #
                                        #####################################################
                    
                                                    #############################
                                                    # POSSIBLE DATA_PREPARATION #
                                                    #############################
                                
                                
        
    # METHODS THAT PREPARE DATA FROM A LIST OF INDICES , RETURN THE RIGHT DATA(S) FOR THE CHOSEN GET_DISTANCE TO WORK
    """
    From two list of indices, returns the distance matrix only concern by those two sets of indices
    can be require to use numpy.max for example
    
    Returns : One array which correspond to a distance matrix
    """
    def get_submatrix(self, cluster_1, cluster_2, distance_points = None) :
        if(not(distance_points)) :
            distance_points = self.distance_points

        return [ distance_points.distance_matrix_points[ cluster_1][ : , cluster_2] ]


    """
    From two list of indices, returns the set of corresponding points in the dataset
    """
    def get_sets_points(self, cluster_1, cluster_2, distance_points = None ) :
        if(not(distance_points)) :
            distance_points = self.distance_points

        return [ distance_points.X[cluster_1], distance_points.X[cluster_2] ]
    
    
    def identity(self, cluster_1, cluster_2) :
        return (cluster_1, cluster_2)
        
        
        
        
        
                                                            #################
                                                            # OTHER METHODS #
                                                            #################
                    
       
    def choose_distance(self, metric) :
        # IF METRIC IS NOT CALLABLE THEN IT SHOULD BE A DISTANCE MATRIX
        if( callable(metric) ) :
            return metric

        # If metric cluster is a string then we associate the good function
        elif(isinstance(metric_clusters, str) ) :
            if( metric_clusters == 'min' ) :
                return self.min_dist
            elif( metric_clusters == 'max' ) :
                return self.max_dist

            elif( metric_clusters == 'emd' ) :
                return self.EMD_for_two_clusters
            
            elif( metric_clusters == 'avg_ind' ) :
                return self.mean_dist_indices

            else :
                return self.mean_dist

        # If metric is a distance matrix we store it method to access the distances
        else :
            return self.access_dist_matrix_clusters
        
       
                    

    #-----------------------------------------------------------

    """
    Function which compute the size displayed of each cluster
    Return the list of the sizes [ [cluster1, diameter],[cluster2, diameter]...] 

    Parameters :
    metric : metric used to compute the distance between two points 
    general_metric : metric used to compute the distance between two group of points , this result
    with for each cluster with itself will be the diameter
    """

    # Has sense only if we have list of cluster, not possible with distance matrix
    def diameter_clusters(self, metric_clusters = None, distance_points = None,
                         data_preparation = None, parameters_metric_clusters = {}) :

        # If not metric between clusters given then we use the initial one, otherwise we take the one fitting
        if(not(metric_clusters) ) :
            metric_clusters = self.get_distance
        else :
            metric_clusters = self.choose_distance(metric_clusters)

        # If no data_preparation for metric between object given then we use the initial one
        if(not(data_preparation)) :
            data_preparation = self.data_preparation

        # If no distance_points object given then we use the initial one
        if(not(distance_points)) :
            distance_points = self.distance_points

        if(not(parameters_metric_clusters) ) :
            parameters_metric_clusters = self.parameters_metric_clusters 



        list_diameters = []

        for i in range(len(self.clusters)) :
            d = metric_clusters(*data_preparation(self.clusters[i], self.clusters[i], distance_points) ,
                                **parameters_metric_clusters )
            list_diameters.append([i+1, d])

        return list_diameters
            
      
        

#------------------------------------------------------------------------------------------------------------------------------------------        
        
        
                                                        ######################
                                                        # CREATION_DISTANCES #
                                                        ######################
                    
                    
                    
"""
Class which aim to create an object Distance_between_clusters in one step instead of creating separately Distance_between_points and after
Distance_between_clusters


Parameters :
all parameters are described for the two classes Distance_between_clusters and Distance_between_points
"""
class Creation_distances() :
    
    def __init__(self, 
                 # Parameters connected with Distance_between_clusters
                  metric_clusters = 'average', parameters_metric_clusters = {}, clusters = None, 
                 distance_points= None, data_preparation = 'usual',
                 
                 #Parameters connected with Distance_between_points
                 metric_points =  sp.euclidean,
                 parameters_metric_points = {},  
                 X = None  ) :
        
        
        # if metric is a function then we need to create the object distance between points
        if( callable(metric_clusters ) or isinstance(metric_clusters, str) ) :
            
            self.distance_points_ = Distances_between_points( metric_points =  metric_points,
                 parameters_metric_points = parameters_metric_points,  X =  X )
        
        else :
            distance_points_ = None
            
        self.distance_clusters_ = Distances_between_clusters(metric_clusters = metric_clusters,
                                                        parameters_metric_clusters =parameters_metric_clusters,
                                                             clusters= clusters, 
                                                             distance_points = self.distance_points_,
                                                            data_preparation =data_preparation)
        
        
        
    def get_distance_cluster(self) :
        return self.distance_clusters_
        
        
        
        
    
    
    
    
 
