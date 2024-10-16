
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


class ConnectivityPruning :

    def __init__(self,  algo ="bf", weight = "weight" ) :
        """_summary_

        Parameters
        ----------
        algo : str {"bf", “ps”} , optional
            Choice of the algorithm used to prune edges in the graph. “bf” correspond to the best and also the slowest algorithm (the brut force algorithm).
              “ps” is the quickest but does not ensure the best pruning, by default "bf"
        weight : str, optional
            The key underwhich the weight/size of edges is stored in the graph, by default "weight"
        """
        self.weight = weight
        if(algo == "bf") :
            self.prune = self.BF_edge_choice

        else :
            self.prune = self.PS_edge_choice

    
    def connectivity_graph(self,  graph ) :
        """_summary_
    Method which returns the global connectivity of a given graph.
    Parameters
    ----------
    graph :  networkx graph
         Graph for which the global connectivity is computed.
    Returns
    -------
     float
         Returns the global connectivity of the graph.
"""
        nodes = list(graph.nodes)
        short_paths= dict(nx.all_pairs_dijkstra_path_length(graph, weight= self.weight ) )
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
    

    def BF_edge_choice(self, g, nb_edges = -1, score = False ) :
        """_summary_
    Method which prunes a given number of edges by using the Brute Force algorithm based on the connectivity.
    Parameters
    ----------
    g : networkx.Graph
        Graph for which the edges are pruned.
    nb_edges : int
        Number of edges to be pruned
    score : bool
        If True, the method also returns the evolution of the connectivity.

    Returns
    -------
    networkx.Graph , list
        Returns the pruned graph and if the parameter score is True, returns also a list of float which corresponds to the evolution of the connectivity kept after each pruned edge compared to the original graph.
"""
        graph =  g.copy()
        f = list(graph.edges)
        removed_edges = []
        M = []
        conn_prune = [1]
        
        if(nb_edges==-1) :
            nb_edges = len(f)
            
        for i in range(nb_edges) :
            rk_largest = float('-inf')
            e_largest = False

            # GET F\M
            f_minus_M = deepcopy(f) 
            if( len(f_minus_M) != len(f)  ) :
                raise(Exception)
                
            for e in M :
                for i in range(len(f_minus_M) ):
                    if( f_minus_M[i][0] == e[0] and  f_minus_M[i][1] == e[1]  ) :
                        f_minus_M.pop(i)
                        break
            
            c_fix_loop = self.connectivity_graph(graph)
            
            for edge in f_minus_M :
                edge_data = deepcopy( graph.get_edge_data( edge[0] , edge[1] ) )
                #edge_err = deepcopy( edge_data[self.weight] )
                
                #print('REMOVE', edge)
                graph.remove_edge( edge[0], edge[1] )
                nb_compo = nx.number_connected_components(graph)
                
                if( nb_compo == 1) :
                    rk = self.connectivity_graph(graph) / c_fix_loop  

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
                removed_edges.append((e_largest[0], e_largest[1]))
                graph.remove_edge( e_largest[0], e_largest[1] )
                
        if(not(score) ):
            return graph, removed_edges
        else :
            return graph, removed_edges, conn_prune
        



    # Each round, the less useful edge (for the two nodes it directly connects) is deleted (we don't measure the impact of that on the whole graph)          
    def PS_edge_choice(self, g, nb_edges,  score = False) :
        """_summary_
        Method which prunes a given number of edges by using the Path Simplification algorithm. 
        Parameters
        ----------
        g : networkx.Graph
            Graph for which the edges are pruned.
        nb_edges : int
            Number of edges to be pruned
        score : bool
            If True, the method also returns the evolution of the evaluation criteria.

        Returns
        -------
        networkx.Graph , list
            Returns pruned graph and if the parameter score is True, 
            returns also a list of float which corresponds to the evolution of the evaluation criteria.
        """

        graph =  g.copy()
        f = list( graph.edges )
        M = []
        removed_edges =[]
        lost_prune = []
        for i in range(nb_edges) :
            k_largest = float('-inf')
            e_largest = False
            
            # GET F\M
            f_minus_M = deepcopy(f) 
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
                edge_data = deepcopy( graph.get_edge_data( edge[0] , edge[1] ) )
                edge_err = deepcopy( edge_data[self.weight] )
                
                #print('REMOVE', edge)
                graph.remove_edge( edge[0], edge[1] )
                
                try :
                    min_path_error =  1/nx.dijkstra_path_length(graph, edge[0], edge[1] , weight=self.weight)
                    
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
                    e_largest = deepcopy( edge )
                

            if( not(isinstance(e_largest, bool) ) ) :
                # DELETE THE LARGEST FROM THE GRAPH          
                for i in range(len(f) ):
                        if(   f[i][0] == e_largest[0] and  f[i][1] == e_largest[1]     ) :
                            f.pop(i)
                            break
                lost_prune.append( k_largest )
                removed_edges.append((e_largest[0], e_largest[1]))
                graph.remove_edge( e_largest[0], e_largest[1] )
                            
            if( len(f) != len(graph.edges) ) :
                print( "EMERGENCY" )
                raise(Exception)
            
        if(not(score) ) :
            return graph, removed_edges 
        else :
            plt.scatter(range(len(lost_prune) ), lost_prune )
            return graph, removed_edges, lost_prune
            

