import networkx as nx
import copy
import matplotlib.pyplot as plt


# From a grap measures the global connectivity which is the average of the weight of paths between nodes, ignoring non connected compo values
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



# From two graphs return how much information we still have with the second graph
def grade_graphs( graph_filtered, graph_base ) :
    return connectivity_graph(graph_filtered)/connectivity_graph(graph_base)





                                                        ##########################
                                                        # BRUTAL FORCE ALGORITHM #
                                                        ##########################
            
            
# Algorithm from which, each edge is deleted if it is the one with the lowest impact on the whole graph    
def BF_edge_choice(g, nb_edge_pruned = -1, conn_plot = False ) :
    graph =  copy.deepcopy(g)
    f = list( graph.edges )
    M = []
    conn_prune = [1]
    
    if(nb_edge_pruned==-1) :
        nb_edge_pruned = len(f)
        
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
                
            else :
                M.append(edge)
                
            graph.add_edge( edge[0], edge[1], **edge_data )
            

        if( not(isinstance(e_largest, bool) ) ) :
            # DELETE THE largest FROM THE GRAPH 
            conn_prune.append(rk_largest) 
            for i in range(len(f) ):
                    if(   f[i][0] == e_largest[0] and  f[i][1] == e_largest[1]     ) :
                        f.pop(i)
                        break
            #print("REMOVE", e_largest)
            graph.remove_edge( e_largest[0], e_largest[1] )
            
    if(not(conn_plot) ):
        return graph 
    else :
        plt.scatter(range(len(conn_prune) ), conn_prune )
        return graph, conn_prune





                                                    #################################
                                                    # PATH SIMPLIFICATION ALGORITHM #
                                                    #################################
            
            
# Each round, the less useful edge (for the two nodes it directly connects) is deleted (we don't measure the impact of that on the whole graph)          
def PS_edge_choice(g, nb_edge_pruned) :
    graph =  copy.deepcopy(g)
    f = list( graph.edges )
    M = []
    for i in range(nb_edge_pruned) :
        k_largest = float('-inf')
        e_largest = False
        
        # GET F\M
        f_minus_M = copy.deepcopy(f) 
        if( len(f_minus_M) != len(f)  ) :
            #print("MIN", f_minus_M)
            #print( "F", f)
            raise(Exception)
            
            
        #print(f_minus_M)
        for e in M :
            for i in range(len(f_minus_M) ):
                if( f_minus_M[i][0] == e[0] and  f_minus_M[i][1] == e[1]  ) :
                    f_minus_M.pop(i)
                    break
                    
                    
        for edge in f_minus_M :
            edge_data = copy.deepcopy( graph.get_edge_data( edge[0] , edge[1] ) )
            edge_err = copy.deepcopy( edge_data['label'] )
            
            #print('REMOVE', edge)
            graph.remove_edge( edge[0], edge[1] )
            
            try :
                min_path_error =  1/nx.dijkstra_path_length(graph, edge[0], edge[1] , weight='label')
                
            except nx.NetworkXNoPath :
                min_path_error = -1
                
            #print("ADD", edge)    
            graph.add_edge( edge[0], edge[1], **edge_data )
            
            
            if (min_path_error >= 1/edge_err ) :
                k = 1
                # Delete the edge
                for i in range(len(f) ):
                    if(   f[i][0] == edge[0] and  f[i][1] == edge[1]     ) :
                        f.pop(i)
                        #print('RE22222222', edge)
                        graph.remove_edge( edge[0], edge[1] )
                        e_largest = False
                        break
                break
                            
            elif ( 0 < min_path_error and  min_path_error < 1/edge_err ) :
                k = min_path_error/ (1/edge_err)
                
            else :
                k = float('-inf')
                M.append( [edge[0], edge[1] ] )
            
            if ( k > k_largest  ) :
                k_largest = k
                e_largest = copy.deepcopy( edge )
            

        if( not(isinstance(e_largest, bool) ) ) :
            # DELETE THE LARGEST FROM THE GRAPH          
            for i in range(len(f) ):
                    if(   f[i][0] == e_largest[0] and  f[i][1] == e_largest[1]     ) :
                        f.pop(i)
                        break
            graph.remove_edge( e_largest[0], e_largest[1] )
                        
        if( len(f) != len(graph.edges) ) :
            print( "EMERGENCY" )
            raise(Exception)
            
    
    return graph             







