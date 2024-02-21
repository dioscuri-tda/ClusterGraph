from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd




class NodeStrategy :
    
    def __init__(self, graph, size_strategy = "lin", type_coloring = "label", palette = None , color_labels = None, 
                 X = None, variable = None,  choiceLabel = "max" , coloring_strategy_var = 'lin', MIN_SIZE_NODE = 0.1 ) :
        """_summary_

        Parameters
        ----------
        graph : _type_ networkx Graph
            _description_ Graph which will be preprocessed by adding colors to nodes, normalizing their size and other properties.
        size_strategy : str, optional
            _description_ Defines the formula which is used to normalize the size of nodes. It can be "lin", "log", "exp" or "id". , by default "lin"
        type_coloring : str, optional
            _description_ Defines the way to add colors to nodes. It can be with "label" or with "variable". For "label", colors are added from a given list or dictionary. 
            If "variable" is chosen, the color is chosen depending from a feature of the dataset. It can be the average of a given feature inside each node for example. , by default "label"
        palette : _type_ matplotlib.colors.ListedColormap , optional
            _description_ The colormap from which color will be chosen for nodes. , by default None
        color_labels : _type_ list, dict or numpy array, optional
            _description_ Object from which the colors of nodes will be retrieved. The exact colors can be chosen by giving hexadecimal values.
            If a list or a numpy array is given and has the same length than the number of nodes, each node will be associated to a label. If the list is longer, the label associated to each node will depend on the which labels are represented inside each nodes by the points covered.
            If a dictionary is chosen, the keys should be the nodes and the values the labels on which the color will be chosen. 
            , by default None
        X : _type_ numpy darray, optional
            _description_ The dataset from which the value inside each node will be taken if the type_coloring is set to "variable"., by default None
        variable : _type_ str or int , optional
            _description_ If the parameter type_coloring is set to "variable", this parameter is giving access to the good feature in the dataset. It can be an index or the name of the variable.
              , by default None
        choiceLabel : str, optional
            _description_ Can be "max" or "min". When the parameter "type_coloring" is set to "label", it defines the way to choose the label inside each node to color them. If "max" is chosen, the most represented label inside each node will be chosen. If "min" is chosen it will be the least represented label.
            , by default "max"
        coloring_strategy_var : str, optional
            _description_ Can be "log", "lin" or "exp". When the parameter "type_coloring" is set to "variable", this parameter represents how fast color will changed between nodes depending on the variable's average value inside each node. 
            For example if "exp" is chosen, a slight change of average value between two nodes will represent an important change of colors.
            , by default 'lin'
        MIN_SIZE_NODE : float, optional
            _description_, by default 0.1

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        
        self.myPalette = palette
        self.color_labels = color_labels
        self.dictLabelsCol = None
        self.MAX_VALUE_COLOR = None
        self.MIN_VALUE_COLOR = None
        self.graph = graph
        self.X = X
        self.variable = variable
        self.MIN_SIZE_NODE = MIN_SIZE_NODE

        
        if( choiceLabel == "max") :
            self.labelChoice = np.argmax
        elif( choiceLabel == "min") :
            self.labelChoice = np.argmin
        else :
            raise ValueError("The choice of label must be 'min' or 'max' ")
        
        if(size_strategy == "log" ) :
            self.get_size_node  = self.log_size
        elif(size_strategy == "lin"):
            self.get_size_node  = self.linear_size
        elif(size_strategy == "exp"):
            self.get_size_node = self.expo_size
        elif( size_strategy == "id") :
            self.get_size_node = self.id_size
        else :
            raise ValueError("Only 'log', 'lin' and 'exp' are accepted as a size_strategy " )
            
        # CHOICE OF STRATEGY TO GET THE WAY OF COLORING
        
        # WITH LABEL
        if( type_coloring== "label" ) :
            if(self.myPalette is None ) :
                  self.myPalette = cm.get_cmap(name="tab20c")

            self.fit_color = self.set_color_nodes_labels
            self.get_labels()
            self.get_labels_into_hexa()

            
             # Choice of function to set colors to each node 
            if(  color_labels is None  or  (len(color_labels) == len(list(self.graph.nodes))  )  ) :
                self.get_color_node = self.get_color_node_unique
                self.getDictNodeHexa()
            
            elif ( len(color_labels) > len(list(self.graph.nodes))   ) :
                self.get_color_node = self.get_color_node_points_covered
                  
            else :
                raise ValueError("Less labels than nodes or points in nodes " )
            
                
        # WITH VARIABLE
        elif ( type_coloring == "variable") :
            self.fit_color = self.set_color_nodes_variable

            if(self.myPalette is None ) :
                  self.myPalette = cm.get_cmap(name="YlOrBr") 
            
            if(coloring_strategy_var == 'log' ) :
                self.get_color_var = self.get_color_var_log
                
            elif(coloring_strategy_var == 'lin' ) :
                self.get_color_var = self.get_color_var_lin
                
            elif(coloring_strategy_var == 'exp' ) : 
                self.get_color_var = self.get_color_var_exp
                
            else :
                raise ValueError("Only 'log', 'lin' and 'exp' are accepted for the 'coloring_strategy_var' " )
            
            if( not( X is None) ) :
                if isinstance(X, pd.DataFrame):
                    self.get_val_node = self.get_val_var_node_Xpand
                elif isinstance(X, np.ndarray):
                    self.get_val_node = self.get_val_var_node_Xnum
            else :
                self.get_val_node = self.get_val_var_node_graph
                    
        else :
            raise ValueError("Only 'label' and 'variable' are accepted for the 'type_coloring' " )
            
            
    def fit_nodes(self) :
        """_summary_
        Method which calls methods in order to set the size of nodes and their colors.
        """
        self.set_size_nodes()
        self.fit_color()
            
                                                    
    def get_mini_maxi(self) :
        """_summary_
        Method which returns the maximum and minimum sizes of nodes in the graph. 

        Returns
        -------
        _type_ int, int
            _description_ The maximum and minimum size of nodes of the graph.
        """
        nodes = list( self.graph.nodes )
        mini = len( self.graph.nodes[ nodes[0] ]["points_covered"] )
        maxi = mini
        for node in nodes :
            size = len( self.graph.nodes[node]["points_covered" ] )
            if(size > maxi) :
                maxi = size
            if(size<mini) :
                mini = size
        return maxi, mini
    
    def set_size_nodes(self ) :
        """_summary_
        
        """
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
        if(self.color_labels is None) :
            self.color_labels = list( self.graph.nodes )

        
    # TRANSFORMATION OF THE GIVEN LABELS INTO HEXADECIMALS
    # GET HEXADECIMAL VALUE FOR EACH NODE
    def get_labels_into_hexa(self) :
        if( type(self.color_labels ) is dict  ) :
            keys = list( self.color_labels )
        else :
            keys = range( len(self.color_labels) )
        
        # TEST IF WE NEED TO TRANSFORM LABELS INTO HEXADECIMAL VALUES
        all_hex = True
        for k in keys :
            if( not( is_color_like( self.color_labels[k] )  )   ) :
                all_hex = False
                break
        
        if( type(self.color_labels ) is dict ) :
        
            #  if color_labels is a dictionary and values are not hexadecimals we transform them
            if ( not(all_hex) ) :
                # Function to transform labels to hexa
                self.labColors = self.dictLabelToHexa()
                # Function to get the dictionary Node Hexa
                self.getDictNodeHexa = self.nodeColHexa_dictLabHexa

            else :
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
    


    # FUNCTIONS WHICH  TRANSFORM LABELS INTO HEXADECIMALS 
    def dictLabelToHexa(self) :
        values_labels = list( self.color_labels.values() )
        keys = list(self.color_labels)
        uniqueLabels = np.unique( values_labels )
        nbLabels = len( uniqueLabels )
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        return self.dictLabelsCol
        

    def listLabelToHexa(self) :
        uniqueLabels = np.unique( self.color_labels )
        nbLabels = len( uniqueLabels )
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
        self.NodeHexa = self.labColors

    # if the dictionary has a label per node
    def nodeColHexa_dictLabHexa(self) :
        keys = list( self.color_labels )
        # we create the dictionary node hexadecimal
        self.NodeHexa = {}
        for k in keys :
            self.NodeHexa[k] = self.labColors[ self.color_labels[k]   ]


    # if color_labels is a list
    # labColors is a list with hexadecimal values
    def nodeColHexa_listHexa(self) :
        nodes = list( self.graph.nodes )
        #mini_node = min( nodes )
        # we create the dictionary node hexadecimal
        self.NodeHexa = {}
        for i,n in enumerate( nodes ) :
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
        self.graph.nodes[n]["color"] = self.dictLabelsCol[label]   

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
        return self.graph.nodes[node][self.variable]


    #############               GET COLOR FOR VARIABLE PART     ################################                  
    def get_color_var_exp(self, val ) :
        color_id = (np.exp(val) - np.exp(self.MIN_VALUE_COLOR)) / (np.exp(self.MAX_VALUE_COLOR) - np.exp(self.MIN_VALUE_COLOR))
        return to_hex( self.myPalette( color_id ) )

    def get_color_var_log(self, val ) :
        color_id = (np.log10( val ) - np.log10(self.MIN_VALUE_COLOR))  / (np.log10(self.MAX_VALUE_COLOR) - np.log10(self.MIN_VALUE_COLOR))
        hex = to_hex(self.myPalette(color_id))
        return hex

    def get_color_var_lin(self, val) :
        color_id = ( val - self.MIN_VALUE_COLOR ) / (self.MAX_VALUE_COLOR - self.MIN_VALUE_COLOR )
        return to_hex(  self.myPalette(color_id)   )
    
        
    
    
    
    
    
    
    
    
  