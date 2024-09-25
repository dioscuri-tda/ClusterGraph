
import networkx as nx
import numpy as np

class Subsampling :

    def __init__ ( self, clusters , variable_clusters = "points_covered", perc = 0.5, seed= None ) :
        if perc < 0 or perc > 1:
            raise ValueError("Percentage should belong to the interval 0, 1 ")
        
        self.perc = perc

        if( type(clusters ) == nx.classes.graph.Graph ) :
            self.clusters = self.get_clusters_from_graph(clusters, variable_clusters)

        else :
            self.clusters = clusters

        self.subsampling_clusters()

        if( seed is not None ) :
            np.random.seed(seed)



    def get_clusters_from_graph(self, g_clusters, variable_clusters) :
        return np.array(  [ g_clusters.nodes[n][variable_clusters]  for n in g_clusters.nodes ]   )
    

    def subsampling_clusters( self ) :
        subclusters = []
        for sublist in self.clusters :
            sublist_size = len(sublist)
            sample_size = max(1, int(sublist_size * self.perc) )
            # rng = np.random.default_rng(seed=42)
            sampled_items = np.random.choice(sublist, size=sample_size, replace=False)

            subclusters.append(sampled_items)

        self.subsampled_clusters = np.array( subclusters )
        return self.subsampled_clusters
    


    def data_transformation(self, X) :
        restrict_indices = []
        for l in self.subsampled_clusters :
            restrict_indices.extend(l)
            
        restrict_indices = np.unique(restrict_indices) 

        # dictionary letting us from an old index get its new index in the restricted dataset
        self.dict_old_new_indices = {value : index for index, value in enumerate(restrict_indices)}
        # dictionary letting us from an new index get its old index in the old dataset
        self.dict_new_old_indices = {index : value for index, value in enumerate(restrict_indices)}
        self.X_restricted = X[restrict_indices, :  ]

        




    



    
    



