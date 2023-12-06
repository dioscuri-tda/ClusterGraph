from scipy.spatial.distance import euclidean
from networkx import add_path
import networkx as nx
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random






class Metric_distortion :
    
    def __init__(self, graph, knn_g , X, sample = 1,  k_n = 2 ,   weight_knn_g = 'label', k_compo = 2, merged_knn = False, dist_weight = True, speed ="best" ) :
        """_summary_

        Parameters
        ----------
        cg : _type_ networkx.Graph
            _description_ graph which will be pruned
        knn_g : _type_
        _description_
        X : _type_ numpy darray
            _description_ the dataset 
        sample : _type_ float
        _description_ percentage of samples kept per cluster in order to compute faster
        k_n : _type_
            _description_ number of neighbors in case we need to create a KNN graph (when subsampling)
        weight_knn_g : _type_ str
        _description_ name of the variable giving access to the distance between two nodes in the knn graph
        k_compo : _type_
            _description_ number of edges which will be added in order to merge disconnected components
        cg : _type_
        _description_
        """
        self.graph = graph
        self.knn_g = knn_g
        self.X = X
        self.weight_knn_g = weight_knn_g 
        self.label_points_covered_intr = "points_covered"
        self.new_knn = False
        self.nb_points_disco = 0
        self.nb_points_should_be_evaluated =0

        if(speed =="best") :
            self.prune = self.prune_edges_BF
        elif(speed == "medium") :
            self.prune = self.prune_edges_PS


        if(dist_weight ) :
            self.distortion_graph = self.distortion_graph_weight 
        else : 
            self.distortion_graph = self.distortion_graph_no_weight 


        if(sample < 1 ) :
            self.sample_clusters( sample )
            self.set_knn_intri( k_n )

        self.k_nn_compo = k_compo 
        self.dijkstra_length_dict = dict(nx.all_pairs_dijkstra_path_length( self.knn_g  , weight = weight_knn_g   ))
        
        # CREATION OF THE INTRINSIC CLUSTERGRAPH
        self.intri_cg = copy.deepcopy( graph )
        nodes = list(self.intri_cg.nodes())

        # Ajouter des aretes entre chaque paire de noeuds
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                self.intri_cg.add_edge(nodes[i], nodes[j])


        self.intrin_dist_cg()

     
        
    # COMPUTE INTRINSIC DISTANCE
    def intrin_dist_cg(self) :
        edges_between_compos, self.intri_cg = self.remove_edges( self.intri_cg )
        connected_components = [ self.intri_cg.subgraph(c) for c in nx.connected_components( self.intri_cg  )]
        for cc in connected_components :
            nb_points_inter = 0
            nb_points = len( self.X )
            nb_nodes =  len( list( cc.nodes ) )
            for n1 in cc.nodes :
                #print( cc.nodes[n1]  )
                for n2 in cc.nodes :
                    if( n1 < n2 ) :
                        intr_dist_c1_c2 = self.intr_two_clusters( cc.nodes[n1][self.label_points_covered_intr],
                                                                                cc.nodes[n2][self.label_points_covered_intr] )
                        self.intri_cg.edges[(n1,n2)]["intr_dist"] = intr_dist_c1_c2
        
        if(self.nb_points_disco > 0) : 
            print("Be careful, ", self.nb_points_disco , " intrinsic distance between points have not been evaluated in the metric distortion process. It represents ",
                  self.nb_points_disco /self.nb_points_should_be_evaluated , " of the distances.")

                         
    def intr_two_clusters(self, c1, c2  ) :
        l_c1 = len(c1)
        l_c2 = len(c2)
        intr_dist = 0
        nb_not_connected = 0
        for i in c1 :
            for j in c2 :
                dist = self.intr_two_points( i , j ) 
                if(dist >= 0 ) :
                    intr_dist += dist
                else :
                    nb_not_connected +=1
        self.nb_points_disco += nb_not_connected
        self.nb_points_should_be_evaluated += l_c1*l_c2
        return intr_dist/( l_c1*l_c2 - nb_not_connected )
                
                
    def intr_two_points(self, i , j ) :
        try :
            return self.dijkstra_length_dict[i][j]
        except :
            return -1
    
   
    def distortion_graph_no_weight(self, graph, intrinsic_graph ) :
        connected_components = [ graph.subgraph(c).copy() for c in nx.connected_components( graph  )]
        short_paths= dict(nx.all_pairs_dijkstra_path_length(graph, weight='label') )
        distortion_g = 0
        nb_pair =0
        for cc in connected_components :
            dist_cc = 0
            nodes = list(cc.nodes)
            nodes.sort() 
            for n1 in nodes :
                for n2 in nodes :
                    if( n1 < n2) :
                        nb_pair +=1
                        distortion_g +=  abs ( np.log10( ( short_paths[ n1 ][ n2 ] ) /
                                                    intrinsic_graph.edges[ ( n1 , n2 ) ]["intr_dist"]  )  ) 
                        
                        dist_cc += abs ( np.log10( ( short_paths[ n1 ][ n2 ] ) /
                                                    intrinsic_graph.edges[ ( n1 , n2 ) ]["intr_dist"]  )  ) 
                        
            #print("DISTORTION COMPONENT : ", dist_cc )
        if(nb_pair > 0) :
            return distortion_g/nb_pair
        else : 
            print("No edge to evaluate")
            return 0
        

    def distortion_graph_weight( self, graph, intrinsic_graph ) :
        #print("NEW WEIGHT")
        connected_components = [ graph.subgraph(c).copy() for c in nx.connected_components( graph  )]
        short_paths= dict(nx.all_pairs_dijkstra_path_length(graph, weight='label') )
        dist_global = 0
        nb_pair_global = 0 
        for cc in connected_components : 
            nodes = list(cc.nodes )
            weight_total = 0
            dist_compo = 0
            nb_pair_compo = 0
            for n1 in nodes :
                for n2 in nodes :
                    if( n1 < n2 ) :
                        weight_n1_n2 = self.nb_compo_cluster[n1] + self.nb_compo_cluster[n2]   #self.get_weight_n1_n2( graph, n1,n2 )
                        dist_pair = abs ( np.log10( ( short_paths[ n1 ][ n2 ] ) /
                                                    intrinsic_graph.edges[ ( n1 , n2 ) ]["intr_dist"]  )  ) 
                        
                        dist_compo += weight_n1_n2 * dist_pair
                        weight_total += weight_n1_n2
                        nb_pair_compo +=1
            if( nb_pair_compo > 0 ) :         
                dist_compo = dist_compo / (nb_pair_compo * weight_total)
                nb_pair_global += nb_pair_compo
                dist_global += dist_compo * nb_pair_compo
           
        if( nb_pair_global > 0) :
            dist_global = dist_global/nb_pair_global
            return dist_global
        else : 
            print("No edge to evaluate")
            return 0


    # for the given cluster, returns the component in the knn_g containing most of its points
    def associate_cluster_one_compo(self, cluster) :
        connected_components = [ self.knn_g.subgraph(c).copy() for c in nx.connected_components( self.knn_g  )   ]
        number_per_compo = []
        for cc in connected_components :
            nodes = list( cc.nodes )
            count =  0
            for pt in cluster :
                if ( pt in nodes ) :
                    count +=1
            number_per_compo.append( count )
            
        return np.argmax( number_per_compo  ), max(number_per_compo)
    

    # We return the number of points which belong in the most represented cluster for the two nodes
    # they are supposed to have the same most representative component
    def get_weight_n1_n2(self, graph, n1, n2 ) :
        nodes = [n1,n2]
        connected_components = [ self.knn_g.subgraph(c).copy() for c in nx.connected_components( self.knn_g  )   ]
        weight = 0
        for n in nodes :
            cluster = graph.nodes[n][self.label_points_covered_intr]
            number_per_compo = []
            for cc in connected_components :
                nodes = list( cc.nodes )
                count =  0
                for pt in cluster :
                    if ( pt in nodes ) :
                        count +=1
                number_per_compo.append( count )

            weight += max( number_per_compo  )
        
        return weight


    def associate_clusters_compo (self ) :
        nb_compo = nx.number_connected_components( self.knn_g  )
        compo_clusters = {}
        self.nb_compo_cluster = {}
        for i in range(nb_compo) :
            compo_clusters[i] = []
        for n in self.graph.nodes :
            compo, nb = self.associate_cluster_one_compo(  self.graph.nodes[n][ self.label_points_covered_intr ]   )
            compo_clusters[compo].append( n )
            self.nb_compo_cluster[n] = nb
 
        return compo_clusters


    def remove_edges (self, graph) :
        compo_clusters = self.associate_clusters_compo ( )
        keys = list( compo_clusters )
        edges_in_between = []
        for e in self.graph.edges :
            found = False
            for k in keys :
                if( (e[0] in  compo_clusters[k]) and (e[1] in  compo_clusters[k])  ) :
                    found = True
                    break
                    
            if( not( found ) ) :
                data = copy.deepcopy( self.graph.edges[e] )
                edges_in_between.append ( [e[0] , e[1], data    ]     )
                graph.remove_edge( e[0], e[1] )
            
        return edges_in_between, graph

    
    # FUNCTION WHICH RETURN THE GRAPH PRUNED WITH CHANGING THE CONNECTIVITY 
    def prune_edges_BF(self, graph,  nb_edges_pruned = -1 ,  md_plot = True ) :
    
        temp_graph = copy.deepcopy(graph)
        edges_between_compos, temp_graph = self.remove_edges( temp_graph)
        self.edges_between_compo = edges_between_compos
        nb_cc = nx.number_connected_components(temp_graph)
        f = list( temp_graph.edges )
        if(nb_edges_pruned <= 0 ) :
             nb_edges_pruned = len(f)
        else : 
            nb_edges_pruned = min( len(f), nb_edges_pruned )

        if( md_plot  ) :
            self.temp_list_md = [self.distortion_graph( temp_graph, self.intri_cg )]

        for i in range(nb_edges_pruned) :
            #print("i : ", i)
            e_smallest = False
            md_smallest = self.distortion_graph( temp_graph, self.intri_cg )
            #print("MD TO BEAT : ", md_smallest)

            # GET F\M   with F list of all edges
            f_minus_M = copy.deepcopy(f) 
            
            #print("FMINUS M : ", f_minus_M , "size  : ", len(f_minus_M) )
            for edge in f_minus_M :
                edge_data = copy.deepcopy( temp_graph.get_edge_data( edge[0] , edge[1] ) )                
                temp_graph.remove_edge( edge[0], edge[1] )
                nb_compo = nx.number_connected_components(temp_graph)

                if( nb_compo == nb_cc) :
                    md = self.distortion_graph( temp_graph, self.intri_cg )
                    if( md < md_smallest ) :
                        md_smallest = md
                        e_smallest = edge

                temp_graph.add_edge( edge[0], edge[1], **edge_data )

            #print("MD SMALLEST : " ,md_smallest )
            if( not(isinstance(e_smallest, bool) ) ) :

                # DELETE THE smallest FROM THE GRAPH          
                for i in range(len(f) ):
                    if(   f[i][0] == e_smallest[0] and  f[i][1] == e_smallest[1]     ) :
                        f.pop(i)
                        break

                temp_graph.remove_edge( e_smallest[0], e_smallest[1] ) 
                if(md_plot  ) :
                    self.temp_list_md.append( md_smallest )

            else :
                break
        if( md_plot  ) :
            return temp_graph, self.temp_list_md

        else :
            return temp_graph
    
    def prune_edges_PS(self, g, md_plot = True , nb_edges_pruned = None) :
        graph = copy.deepcopy( g)
        edges_between_compos, temp_graph = self.remove_edges( graph)
        self.edges_between_compo = edges_between_compos
        nb_cc = nx.number_connected_components( graph)
        f = list( graph.edges )
        if(nb_edges_pruned is None) :
            nb_edges_pruned = len(f)
        if( md_plot  ) :
            self.temp_list_md = [self.distortion_graph( graph, self.intri_cg )]
            
        for i in range(nb_edges_pruned) :
            e_smallest = False
            md_smallest = np.float('inf')
            
            # GET F\M
            f_minus_M = copy.deepcopy(f) 
            for edge in f_minus_M :
                edge_data = copy.deepcopy( temp_graph.get_edge_data( edge[0] , edge[1] ) )                
                graph.remove_edge( edge[0], edge[1] )
                nb_compo = nx.number_connected_components(graph)

                if( nb_compo == nb_cc) :
                    md_path =  abs( np.log10( 
                        nx.dijkstra_path_length(graph, edge[0], edge[1] , weight='label')/ 
                        self.intri_cg.edges[edge]["intr_dist"] 
                    ) )
                        
                    if( md_path < md_smallest ) :
                        md_smallest = md_path
                        e_smallest = edge

                graph.add_edge( edge[0], edge[1], **edge_data )

            if( not(isinstance(e_smallest, bool) ) ) :
                # DELETE THE LARGEST FROM THE GRAPH          
                for i in range(len(f) ):
                        if(   f[i][0] == e_smallest[0] and  f[i][1] == e_smallest[1]     ) :
                            f.pop(i)
                            break
                graph.remove_edge( e_smallest[0], e_smallest[1] )
            
                if( md_plot  ) :
                    self.temp_list_md.append (self.distortion_graph( graph, self.intri_cg ) )
                        
        return graph 
    

    def greedy_pruning(self, alpha = 0.5, nb_edges = -1, weight = "distortion") :
        graph =  copy.deepcopy(self.graph)
        edges_between_compos, graph = self.remove_edges( graph )
        self.edges_between_compo = edges_between_compos
        graph = self.set_distortion_edges(graph, weight = weight)
        nodes = list( graph.nodes )
        md = self.distortion_graph( graph , self.intri_cg)
        self.temp_list_md = [md]
        sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['distortion'], reverse=True)
        if(nb_edges < 0) :
            nb_edges = len(sorted_edges)
        i = 0
        nb_compo = nx.number_connected_components(graph)
        for e in sorted_edges :
            #print( "EDGEEE : ", e )
            #print("Dist : ", graph.edges[ (e[0], e[1])  ]["distortion"] )
            if( graph.edges[ (e[0], e[1])  ]["distortion"] < alpha ) :
                edge = copy.deepcopy( graph.edges[ (e[0], e[1])  ] )
                print("edge ", edge)
                graph.remove_edge( e[0], e[1] )

                if( nx.number_connected_components(graph) > nb_compo ) : 
                    #print("EDGE : ", edge )
                    graph.add_edge( e[0], e[1] , **edge )
                else :
                    i +=1 
                    md = self.distortion_graph( graph, self.intri_cg )
                    self.temp_list_md.append( md )
                if( i>=nb_edges) :
                    break
    
        #for n in self.graph.nodes :
            #print("N 2 : ", self.graph.nodes[n]  )
        return graph
    
    
    def set_distortion_edges(self, graph, weight = 'distortion') :
        for e in graph.edges :
            graph.edges[e][weight] =  abs(np.log10(graph.edges[e]["label"] /self.intri_cg.edges[e]["intr_dist"] ))
        
        return graph
        

    def plt_md_prune_computed(self, save = None) :
        indices = list(range(len(self.temp_list_md)))
        # Tracer le graphique
        plt.plot(indices, self.temp_list_md, marker= 'o')  # Utilisez 'mamder' pour ajouter des points sur le graphique
        plt.xlabel('Number of edges pruned')
        plt.ylabel('Metric distortion')
        plt.title('Metric distortion depending on the number of pruned edges')
        plt.grid(True)
        if save is not None:
            plt.savefig(fname= save  ,format = 'pdf')
        else :
            plt.show()


    def distortion_each_edge(self ) :
        for e in self.intri_cg.edges :
            self.graph.edges[e]["distortion"] = self.graph.edges[e]["label"] / self.intri_cg.edges[e]["intr_dist"]




    def get_distance_matrix_ccompo( self, pruned_graph  ) :
        #connected_components = [ self.graph.subgraph(c).copy() for c in nx.connected_components( self.graph  )]
        nodes = list( self.graph.nodes ) 
        nodes.sort()
        min_n = nodes[0]
        nb_nodes = len( nodes )
        print("NB ",  nb_nodes )
        dist_mat = np.array( [0]*(nb_nodes**2) ).reshape(nb_nodes,nb_nodes)

        # max ambient distance in cg
        edge_max = -1 
        for e in self.graph.edges :
            dist = self.graph.edges[e]['label']
            if(dist > edge_max) :
                edge_max = dist

        maxi = edge_max  + 1
        paths = dict(nx.all_pairs_dijkstra_path_length( pruned_graph , weight = 'label'  ))

        for n1 in nodes  :
                for n2 in nodes :
                    if( n1 < n2 ) :
                        try : 
                            # becomes maxi to avoid adding nodes belonging in the same component in the knn graph after
                            dist = paths[n1][n2]
                            dist = maxi 
                        except :
                            dist = self.graph.edges[ (n1, n2) ]["label"]

                        dist_mat[n1-min_n][n2-min_n] = dist
                        dist_mat[n2-min_n][n1-min_n] = dist
                    elif (n1 == n2) :
                        dist_mat[n1-min_n][n2-min_n] = maxi
        return dist_mat
     


    def create_knn_graph_merge_compo_CG(self, distance_matrix, k):
        num_nodes = distance_matrix.shape[0]

        # Step 1: Use scikit-learn's NearestNeighbors to find the k nearest neighbors
        nn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nn.fit(distance_matrix)

        nn_adjacency = nn.kneighbors_graph(X= distance_matrix , n_neighbors = k, mode='distance')
        nn_Graph = nx.from_scipy_sparse_array(nn_adjacency, edge_attribute = 'label')
        for n in nn_Graph.nodes :
            try :
                nn_Graph.remove_edge(n,n) 
            except :
                a= 1

        return nn_Graph
    

    def pruned_merged_graph_creation( self, plot = True ) :
        min_node = min( list (self.graph.nodes)  ) 
        pruned_gg = self.prune_edges()
        if(plot) :
            self.plt_md_prune_computed()
            
        dist_mat = self.get_distance_matrix_ccompo(  pruned_gg  )
        # Créez la heatmap en utilisant imshow
        plt.imshow(dist_mat, cmap='viridis', interpolation='nearest')

        # Ajoutez une barre de couleur pour indiquer les valeurs
        plt.colorbar()

        # Affichez la heatmap
        plt.show()

        print(" KKKKK : " , self.k_nn_compo  )
        graph_missing_edges = self.create_knn_graph_merge_compo_CG( dist_mat, self.k_nn_compo )
        nx.draw_networkx(graph_missing_edges)
        plt.show()
        for e in graph_missing_edges.edges :
            pruned_gg.add_edge( e[0] + min_node, e[1]+ min_node,  **self.graph.get_edge_data( e[0]+min_node , e[1]+min_node ) )
            
        self.pruned_g = copy.deepcopy(pruned_gg)

        return self.pruned_g
    

    def sample_clusters( self, sample ) :
        self.label_points_covered_intr = "sample_points_covered"
        self.new_clusters = []
        for n in self.graph.nodes :
            print("N : ", n, self.graph[ n ])
            points =  self.graph.nodes[ n ][ "points_covered"]
            nb_elt_s = int(len(points) * sample  )
            elt_s = random.sample( points , nb_elt_s )
            self.graph.nodes[ n ][ self.label_points_covered_intr ] = elt_s
            self.new_clusters.extend( elt_s )
        
        self.new_clusters.sort()

    


    # NEED TO CHANGE THE NODES LABELS
    def set_knn_intri(self, k ) :
        neigh = NearestNeighbors(n_neighbors = k,  radius = 1)
        neigh.fit(X= self.get_X()   )
        nn_adjacency = neigh.kneighbors_graph(X= self.get_X() , n_neighbors = k, mode='distance')

        nn_Graph = nx.from_scipy_sparse_array(nn_adjacency, edge_attribute = 'label')
        for node in nn_Graph.nodes:
            nn_Graph.remove_edge(node,node)

        self.knn_g = copy.deepcopy( nn_Graph )
        self.new_knn = True


    def get_X(self) :
        if( self.new_knn ) :
            return self.X[self.new_clusters,:]

        else :
            return self.X
        



#      MERGE DISCONNECTED COMPONENTS

    def add_edges(self, edges_in_between, graph) :
        for e in edges_in_between :
            graph.add_edge(  e[0], e[1],  **self.graph.edges[(e[0],e[1])]   ) 
        return graph


    def connectivity_graph(self, graph ) :
        nodes = list(graph.nodes)
        short_paths= dict(nx.all_pairs_dijkstra_path_length(graph, weight='label') )
        nb_nodes = len(nodes)
        C_V_E = 0
        nb_not_existing_path = 0
        for i in range(nb_nodes) :
            # We go twice for the same values, improve that 
            for j in range(i, nb_nodes) :
                if(i !=j) :
                    try :
                        C_V_E += 1/short_paths[ nodes[i] ][nodes[j] ]
                    except :
                        nb_not_existing_path +=1
                        
        if( nb_not_existing_path == 0 ) :
            C_V_E = C_V_E * 2/ ( nb_nodes *(nb_nodes-1) ) 
        else :
            C_V_E = C_V_E * (2/ ( nb_nodes *(nb_nodes-1) ) -1/nb_not_existing_path )
            
        return C_V_E


    def merge_compo_edges_conn(self, pruned_gg, nb_edges_pruned = None , add_edges = True ) :
        self.disco_graph_md = copy.deepcopy( pruned_gg )
        if(add_edges) :
            pruned_gg = self.add_edges(self.edges_between_compo,  pruned_gg  )
        
        self.edges_between_compo = self.remove_edges_between_pruned(pruned_gg)
        if(nb_edges_pruned is None) :
            nb_edges_pruned = len(self.edges_between_compo)

        best_g = None
        graph =  copy.deepcopy(pruned_gg)
        f = list( graph.edges )
        M = []
        based_rk = self.connectivity_graph(graph)
        self.list_rk = [1]
        for i in range(nb_edges_pruned) :
            rk_largest =  np.float('-inf')
            e_largest = False

            # GET F\M
            f_minus_M = copy.deepcopy(f) 
            if( len(f_minus_M) != len(f)  ) :
                raise(Exception)
                
            for e in M :
                for i in range(len(f_minus_M) ):
                    if( f_minus_M[i][0] == e[0] and  f_minus_M[i][1] == e[1]  ) :
                        f_minus_M.pop(i)
                        break
            
            c_fix_loop = self.connectivity_graph(graph)
            
            for edge in f_minus_M :
                edge_data = copy.deepcopy( graph.get_edge_data( edge[0] , edge[1] ) )
                edge_err = copy.deepcopy( edge_data['label'] )
                
                #print('REMOVE', edge)
                graph.remove_edge( edge[0], edge[1] )
                nb_compo = nx.number_connected_components(graph)
                
                if( nb_compo == 1) :
                    rk =  self.connectivity_graph(graph) / c_fix_loop  

                    if(rk > rk_largest ) :
                        rk_largest = rk
                        e_largest = edge
                        c_largest = self.connectivity_graph(graph) / based_rk
                    
                else :
                    M.append(edge)
                    
                graph.add_edge( edge[0], edge[1], **edge_data )
                

            if( not(isinstance(e_largest, bool) ) ) :
                # DELETE THE largest FROM THE GRAPH          
                for i in range(len(f) ):
                        if(   f[i][0] == e_largest[0] and  f[i][1] == e_largest[1]     ) :
                            f.pop(i)
                            break
                    
                graph.remove_edge( e_largest[0], e_largest[1] )   
                self.list_rk.append(c_largest )


        return graph

            
              
    def plt_conn_prune_computed(self) :
        indices = list(range(len(self.list_rk)))
        # Tracer le graphique
        plt.plot(indices, self.list_rk, marker= 'o')  # Utilisez 'mamder' pour ajouter des points sur le graphique
        plt.xlabel('Nb of edges pruned')
        plt.ylabel('Connectivity')
        plt.title('Connectivity depending on the number of pruned edges')
        plt.grid(True)
        plt.show()


    def remove_edges_between_pruned(self, g ) :
        new_edges_bet = []
        for e in self.edges_between_compo :
            if( g.has_edge(e[0], e[1]) ) :
                new_edges_bet.append( copy.deepcopy(e) )
        return new_edges_bet

            

                
 
    
    
    

        

