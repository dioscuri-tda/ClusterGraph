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
        graph : networkx.Graph
            Graph which will be preprocessed by adding colors to nodes, normalizing their size and other properties.
        size_strategy : str, optional
            Defines the formula which is used to normalize the size of nodes. It can be "lin", "log", "exp" or "id". , by default "lin"
        type_coloring : str, optional
            Defines the way to add colors to nodes. It can be with "label" or with "variable". For "label", colors are added from a given list or dictionary. 
            If "variable" is chosen, the color is chosen depending from a feature of the dataset. It can be the average of a given feature inside each node for example. , by default "label"
        palette : matplotlib.colors.ListedColormap , optional
            The colormap from which color will be chosen for nodes. , by default None
        color_labels : list, dict or numpy array, optional
            Object from which the colors of nodes will be retrieved. The exact colors can be chosen by giving hexadecimal values.
            If a list or a numpy array is given and has the same length than the number of nodes, each node will be associated to a label. If the list is longer, the label associated to each node will depend on the which labels are represented inside each nodes by the points covered.
            If a dictionary is chosen, the keys should be the nodes and the values the labels on which the color will be chosen. 
            , by default None
        X : numpy darray, optional
            The dataset from which the value inside each node will be taken if the type_coloring is set to "variable"., by default None
        variable : str or int , optional
            If the parameter type_coloring is set to "variable", this parameter is giving access to the good feature in the dataset. It can be an index or the name of the variable.
              , by default None
        choiceLabel : str, optional
            Can be "max" or "min". When the parameter "type_coloring" is set to "label", it defines the way to choose the label inside each node to color them. If "max" is chosen, the most represented label inside each node will be chosen. If "min" is chosen it will be the least represented label.
            , by default "max"
        coloring_strategy_var : str, optional
            Can be "log", "lin" or "exp". When the parameter "type_coloring" is set to "variable", this parameter represents how fast color will changed between nodes depending on the variable's average value inside each node. 
            For example if "exp" is chosen, a slight change of average value between two nodes will represent an important change of colors.
            , by default 'lin'
        MIN_SIZE_NODE : float, optional
            The minimum size of nodes in the plot, by default 0.1

        Raises
        ------
        ValueError
            
        ValueError
            
        ValueError
            
        ValueError
            
        ValueError
            
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
        Method which returns the maximum and minimum sizes (number of points covered) of nodes in the graph. 

        Returns
        -------
        int, int
            The maximum and minimum size of nodes of the graph.
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
        Browse nodes to set the size in the plot of each node
        
        """
        nodes = list( self.graph.nodes )
        max_size, min_size = self.get_mini_maxi()
        for node in nodes :
            size = len( self.graph.nodes[node]["points_covered"] )
            self.graph.nodes[node]["size_plot"] = self.get_size_node(size, min_size, max_size) 
    
    def log_size( self, size, mini_size, maxi_size ) :
        """_summary_
        Method which returns the logarithmically normalized size of a node in the plot. 
        Parameters
        ----------
        size : int
            Size/number of points covered by a node.
        mini_size : int
            Minimum size of a node in the graph.
        maxi_size : int
            Maximum size of a node in the graph.

        Returns
        -------
        float
            Returns the logarithmically normalized size of a node in the plot. 
        """
        return np.log10(1 + size / maxi_size  )       
        
           
    def linear_size( self, size, mini_size, maxi_size ) :
        """_summary_
        Method which returns the logarithmically normalized size of a node in the plot. 
        Parameters
        ----------
        size : int
            Size/number of points covered by a node.
        mini_size : int
            Minimum size of a node in the graph.
        maxi_size : int
            Maximum size of a node in the graph.

        Returns
        -------
        float
            Returns the linearlly normalized size of a node in the plot. 
        """
        return ( size  - mini_size ) / ( maxi_size - mini_size )
           
    def expo_size( self, size, mini_size, maxi_size ) :
        """_summary_
        Method which returns the logarithmically normalized size of a node in the plot. 
        Parameters
        ----------
        size : int
            Size/number of points covered by a node.
        mini_size : int
            Minimum size of a node in the graph.
        maxi_size : int
            Maximum size of a node in the graph.

        Returns
        -------
        float
            Returns the exponentially normalized size of a node in the plot. 
        """ 
        return (np.exp(size  ) - np.exp(mini_size) ) / (np.exp(maxi_size) - np.exp(mini_size )  )
    
    def id_size( self, size, mini_size, maxi_size ) :
        """_summary_
        Method which returns the same size for every node. 
        Parameters
        ----------
        size : int
            Size/number of points covered by a node.
        mini_size : int
            Minimum size of a node in the graph.
        maxi_size : int
            Maximum size of a node in the graph.

        Returns
        -------
        float
            Returns 1 for every node. 
        """ 
        return 1
    
    
                           
                
                    
    def set_color_nodes_labels( self ) :
        """_summary_
        Method browsing nodes, in order to set the color of each node. Used when the coloring is chosen with labels.
        """
        # set labels and their corresponding hexa colors
        for node in self.graph.nodes :
            #get_color_node depends on the number of points in the label
            self.get_color_node(node) 
     
    def get_color_node_unique( self, n) :
        """_summary_
        Method setting the corresponding label and color to the node "n" when there is a unique label per node.

        Parameters
        ----------
        n : int
            Node for which the color should be set in the graph.
        """
        self.graph.nodes[n]["color"] = self.NodeHexa[n]

    # LABELS PREPARATION 
    def get_labels(self) :
        """_summary_
        Method setting “color_labels” at the list of nodes. It is used when no labels are given. Each node will then have a different color.
        """
        if(self.color_labels is None) :
            self.color_labels = list( self.graph.nodes )

        
    def get_labels_into_hexa(self) :
        """_summary_
        Method transforming the given “color_labels” into a dictionary in which nodes are keys and their corresponding hexadecimal colors as values.
          This method calls the right methods in order to achieve such task.
        """
        if( type(self.color_labels ) is dict  ) :
            keys = list( self.color_labels )
        else :
            keys = range( len(self.color_labels) )
        
        # TEST IF WE NEED TO TRANSFORM LABELS INTO HEXADECIMAL VALUES
        all_hex = True
        for k in keys :
            if( not( is_color_like( self.color_labels[k] )) ) :
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
        """_summary_
        Method creating a dictionary in which labels are the keys and values are the corresponding hexadecimal colors. 
        This method is used when “color_labels” is a dictionary with labels which are not hexadecimal colors.
        Returns
        -------
        dict
            Dictionary in which labels are the keys and values are the corresponding hexadecimal colors.
        """
        values_labels = list( self.color_labels.values() )
        keys = list(self.color_labels)
        uniqueLabels = np.unique( values_labels )
        nbLabels = len( uniqueLabels )
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        return self.dictLabelsCol
        

    def listLabelToHexa(self) :
        """_summary_
        Method creating a dictionary in which labels are keys and values are the corresponding hexadecimal colors. 
        This method is used when “color_labels” is a list of labels which are not hexadecimal colors.
        Returns
        -------
        list
            list in which each element is the color corresponding to the label of the node at the same index in the given list of labels.
        """
        uniqueLabels = np.unique( self.color_labels )
        nbLabels = len( uniqueLabels )
        hexLabels = [ to_hex(self.myPalette(i / nbLabels ) ) for i in range(nbLabels +1)  ]
        self.dictLabelsCol = dict( zip( uniqueLabels  , hexLabels)  )
        listLabels = [ self.dictLabelsCol[e] for e in self.color_labels ]
        return listLabels
    
    def getDictLabelHexaIdentity(self) :
        """_summary_
        Method creating a dictionary in which keys and values are the hexadecimal colors. 
        This method is used when “color_labels” is a list with only hexadecimal values.
        """
        uniqueLabels = np.unique( self.color_labels )
        self.dictLabelsCol = dict( zip( uniqueLabels  , uniqueLabels )  )


    # CREATION OF THE DICTIONARY NODEHEXA FROM DICTIONARY OR LIST WITH HEXADECIMAL 

    # if the dictionary has a hexadecimal value per node
    def nodeColHexa_dictNodeHexa(self) :
        """_summary_
        Method creating the dictionary "NodeHexa" in which, nodes are keys and the corresponding hexadecimal colors, the values. 
        It is used when the given “color_labels” was already a dictonary with only hexadecimal colors as values (labels were already colors).
        """
        self.NodeHexa = self.labColors

    # if the dictionary has a label per node
    def nodeColHexa_dictLabHexa(self) :
        """_summary_
        Method creating the dictionary "NodeHexa" in which, nodes are keys and the corresponding hexadecimal colors, the values.
        It is used when the given “color_labels” was a dictionary and its values (labels) were not colors.
        """
        keys = list( self.color_labels )
        self.NodeHexa = {}
        for k in keys :
            self.NodeHexa[k] = self.labColors[ self.color_labels[k]   ]


    # if color_labels is a list labColors is a list with hexadecimal values
    def nodeColHexa_listHexa(self) :
        """_summary_
        Method creating the dictionary "NodeHexa" in which, nodes are keys and the corresponding hexadecimal colors, the values.
        It is used when the given “color_labels” is a list. It creates the dictionary by associating each index to a node and the color of its labels.
        """
        nodes = list( self.graph.nodes )
        self.NodeHexa = {}
        for i,n in enumerate( nodes ) :
            self.NodeHexa[n] = self.labColors[i] 



    def get_color_node_points_covered(self, n) :
        """_summary_
        Sets the color of the node and stores the percentage of each label represented in the node in the graph as :
            - "data_perc_labels" with a dictionary with each labels present in the node as keys and the number of points belonging to this label as values 
            - "perc_labels" with a string value in which each label in associated to the percentage of points inside this node belonging to this label

        Parameters
        ----------
        n : int
            Node for which the color should be set. The node will also store the percentage of each label covered.
        """
        points = self.graph.nodes[n]["points_covered"]
        nb_points = len(points)
        label_in_node, nb_each = np.unique( self.color_labels[points], return_counts=True )
        perc_each_label = [x/nb_points for x in nb_each]


        index_max = self.labelChoice(nb_each)
        label = label_in_node[index_max]
        self.graph.nodes[n]["color"] = self.dictLabelsCol[label]   

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
            
    
    def set_color_nodes_variable( self ) :
        """_summary_
        Method which sets the color of each node depending on the chosen continuous variable of the node.
        """
        self.set_min_max_mean_var()
        for node in self.graph.nodes :
            self.graph.nodes[node]['color'] = self.get_color_var( self.graph.nodes[node]['data_variable']  )


    def set_min_max_mean_var(self) :
        """_summary_
        Method which  browses nodes in order to store the variable's value inside each node with the key “data_variable” and gets the self.MAX_VALUE_COLOR  and the self.MIN_VALUE_COLOR which correspond to the maximum and minimum values of the variable of the graph among all nodes.
        """
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

                  
    def get_set_val_var_node(self, node )  :
        """_summary_
        Method which, for a given node, stores the node's value inside the graph under “data_variable” and returns the value
        Parameters
        ----------
        node : int
            Node for which we want to store the average variable's value.

        Returns
        -------
        float
            Node's average variable value.
        """
        val_intra_node =  self.get_val_node(node)
        self.graph.nodes[node]["data_variable"] = val_intra_node
        return val_intra_node

    def get_val_var_node_Xnum( self, node ) :
        """_summary_
        Method which, for a given node, get the node's average value when the dataset is a numpy darray and returns the value.
        Parameters
        ----------
        node : int
            Node for which we want to get the average variable's value.

        Returns
        -------
        float
            Node's average variable value.
        """
        return self.X[: ,self.variable][  self.graph.nodes[node]["points_covered"]  ].mean()

    def get_val_var_node_Xpand( self, node ) : 
        """_summary_
        Method which, for a given node, get the node's average value when the dataset is a pandas dataframe and returns the value.
        Parameters
        ----------
        node : int
            Node for which we want to get the average variable's value.

        Returns
        -------
        float
            Node's average variable value.
        """
        if( type(self.variable) == str) :
            return self.X[self.variable][  self.graph.nodes[node]["points_covered"]  ].mean() 
        else :
            return self.X.iloc[ : ,self.variable][  self.graph.nodes[node]["points_covered"]  ].mean() 
        
    def get_val_var_node_graph( self, node ) :
        """_summary_
        Method which, for a given node, get the node's variable's value when it is stored in the graph and returns the value.
        Parameters
        ----------
        node : int
            Node for which we want to get the variable's value.

        Returns
        -------
        float
            Node's variable value.
        """ 
        return self.graph.nodes[node][self.variable]

                
    def get_color_var_exp(self, val ) :
        """_summary_
        Method transforming a real value in hexadecimal by doing an exponential normalization.

        Parameters
        ----------
        val : float
            Variable's value of a node.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = (np.exp(val) - np.exp(self.MIN_VALUE_COLOR)) / (np.exp(self.MAX_VALUE_COLOR) - np.exp(self.MIN_VALUE_COLOR))
        return to_hex( self.myPalette( color_id ) )

    def get_color_var_log(self, val ) :
        """_summary_
        Method transforming a real value in hexadecimal by doing a logarithmic normalization.

        Parameters
        ----------
        val : float
            Variable's value of a node.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = (np.log10( val ) - np.log10(self.MIN_VALUE_COLOR))  / (np.log10(self.MAX_VALUE_COLOR) - np.log10(self.MIN_VALUE_COLOR))
        hex = to_hex(self.myPalette(color_id))
        return hex

    def get_color_var_lin(self, val) :
        """_summary_
        Method transforming a real value in hexadecimal by doing a linear normalization.

        Parameters
        ----------
        val : float
            Variable's value of a node.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = ( val - self.MIN_VALUE_COLOR ) / (self.MAX_VALUE_COLOR - self.MIN_VALUE_COLOR )
        return to_hex(  self.myPalette(color_id)   )
    
        
    
    
    
    
    
    
    
    
  