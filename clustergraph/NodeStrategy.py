from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd




class NodeStrategy :
    
    def __init__(self, graph, size_strategy = "log", type_coloring = "label", palette = None , color_labels = None, 
                 X = None, variable = None,  choiceLabel = "max" , coloring_strategy_var = 'lin', MIN_SIZE_NODE = 0.1 ) :
        
        self.myPalette = palette
        self.color_labels = color_labels
        #print("COLOR LBELS 1", self.color_labels)
        self.dictLabelsCol = None
        self.MAX_VALUE_COLOR = None
        self.MIN_VALUE_COLOR = None
        self.graph = graph
        self.X = X
        self.variable = variable
        self.MIN_SIZE_NODE = MIN_SIZE_NODE

        #print("INIT Done")
        
        if( choiceLabel == "max") :
            #print("MAX")
            self.labelChoice = np.argmax
        elif( choiceLabel == "min") :
            #print("min")
            self.labelChoice = np.argmin
        else :
            raise ValueError("The choice of label must be 'min' or 'max' ")
        
        if(size_strategy == "log" ) :
            #print("log")
            self.get_size_node  = self.log_size
        elif(size_strategy == "lin"):
            #print("lin")
            self.get_size_node  = self.linear_size
        elif(size_strategy == "exp"):
            #print("exp")
            self.get_size_node = self.expo_size
        elif( size_strategy == "id") :
            self.get_size_node = self.id_size
        else :
            raise ValueError("Only 'log', 'lin' and 'exp' are accepted as a size_strategy " )
            
        # CHOICE OF STRATEGY TO GET THE WAY OF COLORING
        
        # WITH LABEL
        if( type_coloring== "label" ) :
            #print("label")
            if(self.myPalette is None ) :
                  #print("PALETTE LAB")
                  self.myPalette = cm.get_cmap(name="tab20c")

            self.fit_color = self.set_color_nodes_labels
            self.get_labels()
            self.get_labels_into_hexa()

            
             # Choice of function to set colors to each node 
            if(  color_labels is None  or  (len(color_labels) == len(list(self.graph.nodes))  )  ) :
                #print("COLOR UNIQUE")
                self.get_color_node = self.get_color_node_unique
                self.getDictNodeHexa()
            
            elif ( len(color_labels) > len(list(self.graph.nodes))   ) :
                #print("COLOR PERCENTAGE")
                self.get_color_node = self.get_color_node_points_covered
                  
            else :
                raise ValueError("Less labels than nodes or points in nodes " )
            
                
        # WITH VARIABLE
        elif ( type_coloring == "variable") :
            #print("VARIABLE")
            self.fit_color = self.set_color_nodes_variable

            if(self.myPalette is None ) :
                  #print("PALETTE VAR")
                  self.myPalette = cm.get_cmap(name="YlOrBr")  #cm.get_cmap(name="Blues") 
            
            if(coloring_strategy_var == 'log' ) :
                #print("COLOR log")
                self.get_color_var = self.get_color_var_log
                
            elif(coloring_strategy_var == 'lin' ) :
                #print("COLOR lin")
                self.get_color_var = self.get_color_var_lin
                
            elif(coloring_strategy_var == 'exp' ) : 
                #print("COLOR exp")
                self.get_color_var = self.get_color_var_exp
                
            else :
                raise ValueError("Only 'log', 'lin' and 'exp' are accepted for the 'coloring_strategy_var' " )
            
            if( not( X is None) ) :
                if isinstance(X, pd.DataFrame):
                    self.get_val_node = self.get_val_var_node_Xpand
                elif isinstance(X, np.ndarray):
                    self.get_val_node = self.get_val_var_node_Xnum
            else :
                #print("VARIABLE WITH GRAPH")
                self.get_val_node = self.get_val_var_node_graph
                    
        else :
            raise ValueError("Only 'label' and 'variable' are accepted for the 'type_coloring' " )
            
            
    def fit_nodes(self) :
        self.set_size_nodes()
        self.fit_color()
            
                                                        #################
                                                        # SIZE OF NODES #
                                                        #################
                        
            
    # Method which gives the maxi and minimum size of nodes in the graph
    def get_mini_maxi(self) :
        nodes = list( self.graph.nodes )
        # Get the maximum and minimum sizes of nodes
        mini = len( self.graph.nodes[ nodes[0] ]["points_covered"] )
        maxi = mini
        for node in nodes :
            size = len( self.graph.nodes[node]["points_covered" ] )
            if(size > maxi) :
                maxi = size
            if(size<mini) :
                mini = size
        return maxi, mini
    
    """                                                                            SIZE OF NODES                                                                      """
    def set_size_nodes(self ) :
        nodes = list( self.graph.nodes )
        max_size, min_size = self.get_mini_maxi()
        for node in nodes :
            size = len( self.graph.nodes[node]["points_covered"] )
            self.graph.nodes[node]["size_plot"] = self.get_size_node(size, min_size, max_size) 
    
    def log_size( self, size, mini_size, maxi_size ) :  
        return np.log10(1 + size / maxi_size  )       
        #return (np.log10(size  ) - np.log10(mini_size) ) / (np.log10(maxi_size) - np.log10(mini_size ) )
           
    def linear_size( self, size, mini_size, maxi_size ) : 
        return ( size  - mini_size ) / ( maxi_size - mini_size )
           
    def expo_size( self, size, mini_size, maxi_size ) : 
        return (np.exp(size  ) - np.exp(mini_size) ) / (np.exp(maxi_size) - np.exp(mini_size )  )
    
    def id_size( self, size, mini_size, maxi_size ) : 
        return 1
    
    
    
                                                                    ##################
                                                                    # COLOR OF NODES #
                                                                    ##################
                            
                
    """                                                                  COLOR WITH GIVEN LABELS                                                                       """
                    
    def set_color_nodes_labels( self ) :
        #print("COLOR LBELS 1.5", self.color_labels)
        # set labels and their corresponding hexa colors
        for node in self.graph.nodes :
            #get_color_node depends on the number of points in the label
            self.get_color_node(node) 


    # METHODS USED TO SET TO ONE NODE ITS CORRESPONDING COLOR AND OTHER DATA CONNECTED WITH COLOR

    # For a given node add the unique color to it        
    def get_color_node_unique( self, n) :
        self.graph.nodes[n]["color"] = self.NodeHexa[n]

    # LABELS PREPARATION 
    def get_labels(self) :
        #print("COLOR LBELS 3", self.color_labels)
        if(self.color_labels is None) :
            #print("NO COLOR GIVEN")
            self.color_labels = list( self.graph.nodes )

        
    # TRANSFORMATION OF THE GIVEN LABELS INTO HEXADECIMALS
    # GET HEXADECIMAL VALUE FOR EACH NODE
    def get_labels_into_hexa(self) :
        if( type(self.color_labels ) is dict  ) :
            #print("DICT")
            keys = list( self.color_labels )
            #print("KEYKS DICT", keys)
        else :
            #print("NOT DICT")
            keys = range( len(self.color_labels) )
        
        # TEST IF WE NEED TO TRANSFORM LABELS INTO HEXADECIMAL VALUES
        all_hex = True
        for k in keys :
            if( not( is_color_like( self.color_labels[k] )  )   ) :
                all_hex = False
                #print("NOT HEXA COLORS : " , k, "  ", self.color_labels[k] )
                break
        
        if( type(self.color_labels ) is dict ) :
        
            #  if color_labels is a dictionary and values are not hexadecimals we transform them
            if ( not(all_hex) ) :
                #print("NOT HEXA AND DICT")
                # Function to transform labels to hexa
                self.labColors = self.dictLabelToHexa()
                # Function to get the dictionary Node Hexa
                self.getDictNodeHexa = self.nodeColHexa_dictLabHexa

            else :
                #print("ALL ARE HEXA AND DICT")
                self.labColors = self.color_labels
                self.getDictNodeHexa = self.nodeColHexa_dictNodeHexa
            
        # IF WE HAVE A LIST
        else : 
            if ( not(all_hex) ) :
                # Function to transform labels to hexa
                self.labColors = self.listLabelToHexa()

            else :
                self.labColors = self.color_labels
                self.getDictLabelHexaIdentity()

            # Function to get the dictionary Node Hexa
            self.getDictNodeHexa = self.nodeColHexa_listHexa

        #self.getDictNodeHexa()
    


    # FUNCTIONS WHICH  TRANSFORM LABELS INTO HEXADECIMALS 
    def dictLabelToHexa(self) :
        #print("LABELS PREPARATION INTO HEXADECIMAL FROM DICT")
        values_labels = list( self.color_labels.values() )
        keys = list(self.color_labels)
        uniqueLabels = np.unique( values_labels )
        nbLabels = len( uniqueLabels )
        #print("TRANSFORM TO HEXA DICT")
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        return self.dictLabelsCol
        

    def listLabelToHexa(self) :
        #print("LABELS PREPARATION INTO HEXADECIMAL FROM LIST")
        uniqueLabels = np.unique( self.color_labels )
        nbLabels = len( uniqueLabels )
        #print("TRANSFORM TO HEXA LIST")
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        listLabels = [ self.dictLabelsCol[e] for e in self.color_labels ]
        return listLabels
    
    def getDictLabelHexaIdentity(self) :
        uniqueLabels = np.unique( self.color_labels )
        self.dictLabelsCol = dict( zip( uniqueLabels  , uniqueLabels )  )


    # CREATION OF THE DICTIONARY NODEHEXA FROM DICTIONARY OR LIST WITH HEXADECIMAL 

    # if the dictionary has a hexadecimal value per node
    def nodeColHexa_dictNodeHexa(self) :
        # we create the dictionary node hexadecimal
        #print("LABELS COLORS AFTER DICT IDENTITY PREPROCESS : ", self.labColors)
        self.NodeHexa = self.labColors

    # if the dictionary has a label per node
    def nodeColHexa_dictLabHexa(self) :
        #print("LABELS COLORS AFTER  DICT PREPROCESS : ", self.labColors)
        keys = list( self.color_labels )
        # we create the dictionary node hexadecimal
        self.NodeHexa = {}
        for k in keys :
            self.NodeHexa[k] = self.labColors[ self.color_labels[k]   ]


    # if color_labels is a list
    # labColors is a list with hexadecimal values
    def nodeColHexa_listHexa(self) :
        #print("LABELS COLORS LIST AFTER PREPROCESS : ", self.labColors)
        nodes = list( self.graph.nodes )
        #mini_node = min( nodes )
        #print("Mini node ", mini_node)
        # we create the dictionary node hexadecimal
        self.NodeHexa = {}
        for i,n in enumerate( nodes ) :
            #print("N : ", n)
            #print("I : ", i)
            self.NodeHexa[n] = self.labColors[i] #[   n - mini_node   ]

#                                                                   COLOR WITH POINTS COVERED                                                                      

# For a given node, returns what should be added for colors
#!! Need to add the list of label for this node + how to store each percentage of label ?
    def get_color_node_points_covered(self, n) :
        # we count how many times each labels is in the node
        points = self.graph.nodes[n]["points_covered"]
        nb_points = len(points)
        label_in_node, nb_each = np.unique( self.color_labels[points], return_counts=True )
        perc_each_label = [x/nb_points for x in nb_each]

        # SET THE CHOSEN COLOR
        index_max = self.labelChoice(nb_each)
        label = label_in_node[index_max]
        self.graph.nodes[n]["color"] = self.dictLabelsCol[label]   #self.labColors[label]  #self.dictLabelsCol[label] 

        # ADD THE POSSIBILITY TO SEE PERCENTAGE OF EACH LABEL
        per_label = ""
        for i in range(len(label_in_node)):
            per_label = (
                per_label
                + "label "
                + str(label_in_node[i])
                + " : "
                + str(
                    round( perc_each_label[i] , 3, )
                )  + ", " )

        self.graph.nodes[n]["perc_labels"] = per_label
        self.graph.nodes[n]["data_perc_labels"] = dict(zip(label_in_node, nb_each) )

    
    def ListLabelToDictHexa( self ) :
        #print("LABELS PREPARATION INTO HEXADECIMAL FROM LIST")
        uniqueLabels = np.unique( self.color_labels )
        nbLabels = len( uniqueLabels )
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        
 
        
    """                                                                          COLOR WITH VARIABLES                                                                  """
            
    
    def set_color_nodes_variable( self ) :
        self.set_min_max_mean_var()
        for node in self.graph.nodes :
            self.graph.nodes[node]['color'] = self.get_color_var( self.graph.nodes[node]['data_variable']  )


    def set_min_max_mean_var(self) :
        nodes = list( self.graph.nodes )
        MIN_VALUE = self.get_set_val_var_node( nodes[0] ) 
        MAX_VALUE =  MIN_VALUE
        for node in self.graph.nodes :
            mean_node = self.get_set_val_var_node( node ) 
            if mean_node > MAX_VALUE :
                MAX_VALUE = mean_node
            if mean_node < MIN_VALUE :
                MIN_VALUE = mean_node

        self.MAX_VALUE_COLOR = MAX_VALUE
        self.MIN_VALUE_COLOR = MIN_VALUE
        #print("MAX VAL NODES", self.MAX_VALUE_COLOR, type(self.MAX_VALUE_COLOR) )
        #print("MIN VAL NODE", self.MIN_VALUE_COLOR, type(self.MIN_VALUE_COLOR)  )



    ############# SET AND GET VARIABLE VALUE FOR A NODE PART ###############                     
    def get_set_val_var_node(self, node )  :
        val_intra_node =  self.get_val_node(node)
        self.graph.nodes[node]["data_variable"] = val_intra_node
        return val_intra_node

    def get_val_var_node_Xnum( self, node ) : 
        return self.X[: ,self.variable][  self.graph.nodes[node]["points_covered"]  ].mean()

    def get_val_var_node_Xpand( self, node ) : 
        if( type(self.variable) == str) :
            return self.X[self.variable][  self.graph.nodes[node]["points_covered"]  ].mean() 
        else :
            return self.X.iloc[ : ,self.variable][  self.graph.nodes[node]["points_covered"]  ].mean() 
        
    def get_val_var_node_graph( self, node ) : 
        #print("graph get val")
        return self.graph.nodes[node][self.variable]


    #############               GET COLOR FOR VARIABLE PART     ################################                  
    def get_color_var_exp(self, val ) :
        color_id = (np.exp(val) - np.exp(self.MIN_VALUE_COLOR)) / (np.exp(self.MAX_VALUE_COLOR) - np.exp(self.MIN_VALUE_COLOR))
        return to_hex( self.myPalette( color_id ) )

    def get_color_var_log(self, val ) :
        #print(" COLOR MIN ",self.MIN_VALUE_COLOR)
        #print(" COLOR MAX ", self.MAX_VALUE_COLOR)
        color_id = (np.log10( val ) - np.log10(self.MIN_VALUE_COLOR))  / (np.log10(self.MAX_VALUE_COLOR) - np.log10(self.MIN_VALUE_COLOR))
        #print("VAL : ", val)
        #print("COL ID : ", color_id)
        hex = to_hex(self.myPalette(color_id))
        #print("HEXA ", hex)
        return hex

    def get_color_var_lin(self, val) :
        color_id = ( val - self.MIN_VALUE_COLOR ) / (self.MAX_VALUE_COLOR - self.MIN_VALUE_COLOR )
        return to_hex(  self.myPalette(color_id)   )
    
        
    
    
    
    
    
    
    
    
  