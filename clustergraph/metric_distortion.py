from math import log
from math import inf
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import copy 
import numpy as np
import matplotlib.pyplot as plt



"""
Function which from shortest path between two clusters and the dictionnary of shortest path in NNgraph and 2 clusters
return the average distortion from points from c1 and points from c2

"""

#!! Change this fonction in case of soft clustering algorithms !!
# si d_cg = 0 si d_k_x(x,y) avec x==y parce que soft clustering
# count the values which are ignored
def distortion_two_clusters(d_cg, c1, c2, dijkstra_length_dict ) :
    avg_distort = 0 
    nb=0
    for i in c1 :
        for j in c2 :
            
            # if points are not connected  => d_k_g = +inf => log = -inf
            try :
                # this try can be improved by storing first dict and try to access second after
                d_k_x = dijkstra_length_dict[i][j]
                
            except KeyError :
                # !! add a verification in case points are the same !!
                d_k_x = inf
            
            if(d_k_x == inf and d_cg == inf) :
                alpha_i_j = abs( log(1) )
                
            elif( d_k_x == inf ) :
                #ln(0)
                #alpha_i_j = - inf 
                alpha_i_j = - inf 
            
            # now we suppose d_cg !=0
            else :
                alpha_i_j = abs( log( d_cg/d_k_x ) )
                
            if( alpha_i_j != -inf) :   
                avg_distort += alpha_i_j
                nb +=1
                
    return avg_distort



def get_nb_points_from_g(cg, label = "points_covered") :
    nb_points =0 
    
    for node in cg.nodes :
        nb_points += len(cg.nodes[node][label])
        
    return nb_points
            





"""
Function which from a ClusterGraph and a Nearest Neighbors Graph returns a 
Graph with the distortion per edge

cg is the grph itself and not the CG objet
"""
def metric_distortion_edges_CG( cg, nn_graph, variable= 'label' ) :
    avg_dist =0
    nb_pair_clust = 0
    nb_points = get_nb_points_from_g(cg)
    
    nodes = cg.nodes 
    #new_cg = copy.deepcopy(cg)
    
    dijkstra_length_dict = dict(nx.all_pairs_dijkstra_path_length(nn_graph, weight = variable))
    
    dijstra_CG = dict(nx.all_pairs_dijkstra_path_length(cg, weight = variable))
    
    for n1 in nodes :
        c1 = cg.nodes[n1]['points_covered']
        for n2 in nodes :
            
            if(n1 != n2) :
                # if the two nodes are in the same CG components we compute the distortion for their path
                try : 
                    d_cg =  dijstra_CG[n1][n2]
                
                except KeyError :
                    d_cg = inf
                    print("nodes not same component")
                    
                c2 = cg.nodes[n2]['points_covered']

                # We compute the distortion for the two clusters 
                distortion_2_clusters = distortion_two_clusters(d_cg, c1, c2, dijkstra_length_dict )
                #new_cg.edges[(n1,n2)][variable] = distortion_2_clusters
                
                if(distortion_2_clusters == inf) :
                    return inf
                
                else :
                    weight_i_j = (len(c1) + len(c2) )
                    avg_dist += distortion_2_clusters * weight_i_j
                    nb_pair_clust += 1
                    
                    
    return (avg_dist/(nb_pair_clust*nb_points) )
                 
                

                                                    # IMPROVEMENT OF THE METRIC DISTORTION USING EDGE PRUNING #



def connectivity_graph( graph ) :
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






# FUNCTION WHICH RETURN THE GRAPH PRUNED WITH CHANGING THE CONNECTIVITY 
def prune_useless_edges(g) :
    best_g = None
    graph =  copy.deepcopy(g)
    f = list( graph.edges )
    M = []
    list_rk = [connectivity_graph(graph)]
    nb_edge_pruned =  len( list( graph.edges ) ) -     ( len( list( graph.nodes  ) ) -1 )
    for i in range(nb_edge_pruned) :
        rk_largest = float('-inf')
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
        
        c_fix_loop = connectivity_graph(graph)
        
        for edge in f_minus_M :
            edge_data = copy.deepcopy( graph.get_edge_data( edge[0] , edge[1] ) )
            edge_err = copy.deepcopy( edge_data['label'] )
            
            #print('REMOVE', edge)
            graph.remove_edge( edge[0], edge[1] )
            nb_compo = nx.number_connected_components(graph)
            
            if( nb_compo == 1) :
                rk =   connectivity_graph(graph) / c_fix_loop  

                if(rk > rk_largest ) :
                    rk_largest = rk
                    e_largest = edge
                    c_largest = connectivity_graph(graph)
                
            else :
                M.append(edge)
                
            graph.add_edge( edge[0], edge[1], **edge_data )
            

        if( not(isinstance(e_largest, bool) ) ) :
            # DELETE THE largest FROM THE GRAPH          
            for i in range(len(f) ):
                    if(   f[i][0] == e_largest[0] and  f[i][1] == e_largest[1]     ) :
                        f.pop(i)
                        break
            
            if(c_largest < list_rk[-1]) :
                best_g = copy.deepcopy(graph)
                #return best_g
                
                
            graph.remove_edge( e_largest[0], e_largest[1] )   
            list_rk.append(c_largest )
            
 
    return list_rk, best_g





                                    ################################################################
                                    # FROM A CLUSTERGRAPH AND A REFERENCED GRAPH TO AN ERROR GRAPH #
                                    ################################################################
                    
                    
                    
                    
# geodesic_dm's values are distances distances based on disjktra distance
# Function which from a normal complete clustergraph and a distance matrix based on dijsktra nb_clusters x nb_clusters,
# return a graph with weighted edges but with the absolute difference between dijsktra and the graph
def get_error_graph( cg_graph, geodesic_dm, weight ='label' ) :
    edge_to_remove = []
    
    # WE ASSUME GEODESICDM IS ORDERED AS THE NODES IN CLUSTERGRAPH
    nodes = list(cg_graph.nodes) 
    #min_node = min( list(cg_graph.nodes) )
    
    for edge in cg_graph.edges(data=True) :
        
        if( np.isnan(geodesic_dm[ nodes.index(edge[0])  , nodes.index(edge[1])  ] ) )  :
            edge_to_remove.append( ( edge[0]  , edge[1] ) )
            
        else :
            edge[2][weight]  =  abs( edge[2][weight] - geodesic_dm[ nodes.index(edge[0])  , nodes.index(edge[1])  ]  )
                                  
    cg_graph.remove_edges_from(edge_to_remove)
        
    return cg_graph





def geodesic_mat_clusters(nn_Graph, clusters, variable = 'label' ) :
    dijkstra_length_dict = dict(nx.all_pairs_dijkstra_path_length(nn_Graph, weight = variable))
    nb_clusters = len(clusters)
    geo_matrix = np.array([float('inf')]*nb_clusters**2).reshape(nb_clusters, nb_clusters)
    for i in range(nb_clusters) : 
        
        for j in range(i,nb_clusters) :
            if(i<j) :
                count =0
                val = 0
                # compute average for this pair of clusters
                for p in range( len(clusters[i]) ) :
                    for q in range( len(clusters[j]) ) :
                        try :
                            val += dijkstra_length_dict[ clusters[i][p] ][ clusters[j][q] ]
                            count +=1

                        except KeyError:
                            print("no path between", clusters[i][p], "and", clusters[j][q] )
                
                geod = val /count    
                geo_matrix[i,j] = geod
                geo_matrix[j,i] = geod
                
            elif (i==j) :
                geo_matrix[i,j] = 0
                
    return geo_matrix 





def md_cg_pruned(cg , nn_g, clusters, variable = 'label', plot_connectivity = False ) : 

    geo_mat = geodesic_mat_clusters(nn_g, clusters, variable = variable )

    err_graph = get_error_graph( cg , geo_mat, weight = variable )
    list_rk, g =  prune_useless_edges( err_graph   )

    # change the value of the edges
    for e in g.edges :
        g.edges[e][variable] = cg.edges[e][variable]
        
    if (plot_connectivity) :
        nb_rk = len(list_rk)
        nbs = list(range(1,nb_rk+1))
        plt.close('all')
        plt.scatter(nbs, list_rk )
        plt.xlabel("Number of pruned edges") 
        plt.ylabel("Connectivity")  
        plt.title("Evolution of the connectivity depending on the number of edges pruned")
        plt.show()

    return g
