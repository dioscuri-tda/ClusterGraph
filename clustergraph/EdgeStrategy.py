from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd





class EdgeStrategy :
    
    def __init__(self, graph,  palette = None, weight = "label", variable = None, norm_weight = "lin" ,
                 type_coloring = "label" , color_labels = None, coloring_strategy_var = 'lin'   ) :
        self.myPalette = palette
        self.graph = graph
        self.weight_edges = weight
        self.variable = variable
        self.MAX_VALUE_COLOR = None
        self.MIN_VALUE_COLOR = None
        self.color_labels = color_labels

        if(norm_weight == "log" ) :
            self.get_weight_e  = self.normalize_log_min_max
        elif(norm_weight == "lin"):
            self.get_weight_e  = self.normalize_lin_min_max
        elif(norm_weight == "exp"):
            self.get_weight_e = self.normalize_exp_min_max

        elif( norm_weight == "id"  ) :
            self.get_weight_e = self.identity_weight
        elif( norm_weight == "max"  ) :
            self.get_weight_e = self.normalize_max
            
        else :
            raise ValueError("Only 'log', 'lin', 'exp', 'id' and 'max' are accepted as a 'norm_weight' " )
        
        # WITH LABEL
        if( type_coloring== "label" ) :
            #print("label")
            if (self.myPalette is None ) :
                  #print("PALETTE LAB")
                  self.myPalette = cm.get_cmap(name="tab20b")

            self.fit_color = self.set_color_edges_labels
            self.get_labels()
            self.get_labels_into_hexa()
            self.get_color_edge = self.get_color_edge_unique
            self.getDictEdgeHexa()
                 
        # WITH VARIABLE
        elif ( type_coloring == "variable") :
            #print("VARIABLE")
            self.fit_color = self.set_color_edges_variable

            if(self.myPalette is None ) :
                    #print("PALETTE VAR")
                    self.myPalette = cm.get_cmap(name="autumn")  #cm.get_cmap(name="Blues") 
            
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
            
            
            self.get_val_edge = self.get_val_var_edge_graph
                    
        else :
            raise ValueError("Only 'label' and 'variable' are accepted for the 'type_coloring' " )
            


    

    def fit_edges(self) :
        self.set_weight_edges()
        self.fit_color()
        
        
                                                                        ###################
                                                                        # WEIGHT OF EDGES #
                                                                        ###################
                        
            
    # Method which gives the maxi and minimum weight of edges in the graph
    def get_mini_maxi(self) :
        edges = list( self.graph.edges )
        # Get the maximum and minimum weight of edges
        mini = self.graph.edges[edges[0] ][ self.weight_edges ] 
        maxi = mini
        for e in edges :
            weight = self.graph.edges[e][ self.weight_edges ] 
            if(weight > maxi) :
                maxi = weight
            if( weight < mini ) :
                mini = weight
        return maxi, mini
    
#                                                 WEIGHT OF EDGES
        
    # function normalizing the edges' length for the plot if required   
    def set_weight_edges(self ) :
        edges = list( self.graph.edges )
        max_weight, min_weight = self.get_mini_maxi()
        for e in edges :
            weight = self.graph.edges[e][self.weight_edges] 
            self.graph.edges[e]["weight_plot"] = self.get_weight_e(weight, min_weight, max_weight) 
    
    def normalize_log_min_max( self, weight, mini_weight, maxi_weight ) :        
        return (np.log10(weight  ) - np.log10(mini_weight) ) / (np.log10(maxi_weight) - np.log10(mini_weight ) )
           
    def normalize_lin_min_max( self, weight, mini_weight, maxi_weight ) : 
        return ( weight  - mini_weight ) / ( maxi_weight - mini_weight )
           
    def normalize_exp_min_max( self, weight, mini_weight, maxi_weight ) : 
        return (np.exp(weight) - np.exp(mini_weight) ) / (np.exp(maxi_weight) - np.exp(mini_weight )  )
    
    def normalize_max( self, weight, mini_weight, maxi_weight ) : 
        return ( weight  / maxi_weight  )
    
    def identity_weight(self, weight, mini_weight, maxi_weight) :
           return weight
    
    
     
                                                                    ##################
                                                                    # COLOR OF edges #
                                                                    ##################
                
                
#                                                                  COLOR WITH GIVEN LABELS
                    
    def set_color_edges_labels( self ) :
        #print("COLOR LBELS 1.5", self.color_labels)
        # set labels and their corresponding hexa colors
        for e in self.graph.edges :
            #get_color_edge depends on the number of points in the label
            self.get_color_edge(e) 

     # METHODS USED TO SET TO ONE edge ITS CORRESPONDING COLOR AND OTHER DATA CONNECTED WITH COLOR

    # For a given edge add the unique color to it        
    def get_color_edge_unique( self, e) :
        self.graph.edges[e]["color"] = self.EdgeHexa[e]

    # LABELS PREPARATION 
    def get_labels(self) :
        #print("COLOR LABELS", self.color_labels)
        if ( self.color_labels is None ) :
                 edges = list( self.graph.edges )
                 self.color_labels = len( edges )*["#000000"]
    

    # TRANSFORMATION OF THE GIVEN LABELS INTO HEXADECIMALS
    # GET HEXADECIMAL VALUE FOR EACH edge
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
                # Function to get the dictionary edge Hexa
                self.getDictEdgeHexa = self.edgeColHexa_dictLabHexa

            else :
                #print("ALL ARE HEXA AND DICT")
                self.labColors = self.color_labels
                self.getDictEdgeHexa = self.edgeColHexa_dictEdgeHexa
            
        # IF WE HAVE A LIST
        else : 
            if ( not(all_hex) ) :
                # Function to transform labels to hexa
                self.labColors = self.listLabelToHexa()

            else :
                self.labColors = self.color_labels
                self.getDictLabelHexaIdentity()

            # Function to get the dictionary edge Hexa
            self.getDictEdgeHexa = self.edgeColHexa_listHexa 


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


    # CREATION OF THE DICTIONARY EdgeHexa FROM DICTIONARY OR LIST WITH HEXADECIMAL 

    # if the dictionary has a hexadecimal value per edge
    def edgeColHexa_dictEdgeHexa(self) :
        # we create the dictionary edge hexadecimal
        #print("LABELS COLORS AFTER DICT IDENTITY PREPROCESS : ", self.labColors)
        self.EdgeHexa = self.labColors

    # if the dictionary has a label per edge
    def edgeColHexa_dictLabHexa(self) :
        #print("LABELS COLORS AFTER  DICT PREPROCESS : ", self.labColors)
        keys = list( self.color_labels )
        # we create the dictionary edge hexadecimal
        self.EdgeHexa = {}
        for k in keys :
            self.EdgeHexa[k] = self.labColors[ self.color_labels[k]   ]


    # if color_labels is a list
    # labColors is a list with hexadecimal values
    def edgeColHexa_listHexa(self) :
        #print("LABELS COLORS LIST AFTER PREPROCESS : ", self.labColors)
        edges = list( self.graph.edges )
        ##print("Mini edge ", mini_edge)
        # we create the dictionary edge hexadecimal
        self.EdgeHexa = {}
        for i, e in enumerate( edges ) :
            #print("E : ", e)
            self.EdgeHexa[e] = self.labColors[   i  ]

               
#                                                                         COLOR WITH VARIABLES
            

    def get_val_var_edge_graph( self, e ) : 
            #print("graph get val")
            return self.graph.edges[e][self.variable]
    
    def set_color_edges_variable( self ) :
        self.set_min_max_mean_var()
        for e in self.graph.edges :
            self.graph.edges[e]['color'] = self.get_color_var( self.graph.edges[e]['data_variable']  )

    def set_min_max_mean_var(self) :
        edges = list( self.graph.edges )
        MIN_VALUE = self.get_set_val_var_edge( edges[0] ) 
        MAX_VALUE =  MIN_VALUE
        for edge in self.graph.edges :
            mean_edge = self.get_set_val_var_edge( edge ) 
            if mean_edge > MAX_VALUE :
                MAX_VALUE = mean_edge
            if mean_edge < MIN_VALUE :
                MIN_VALUE = mean_edge

        self.MAX_VALUE_COLOR = MAX_VALUE
        self.MIN_VALUE_COLOR = MIN_VALUE
        #print("MAX VAL edges", self.MAX_VALUE_COLOR, type(self.MAX_VALUE_COLOR) )
        #print("MIN VAL edge", self.MIN_VALUE_COLOR, type(self.MIN_VALUE_COLOR)  )


    ############# SET AND GET VARIABLE VALUE FOR A NODE PART ###############                     
    def get_set_val_var_edge(self, e)  :
        val_intra_e =  self.get_val_edge(e)
        self.graph.edges[e]["data_variable"] = val_intra_e
        return val_intra_e
    



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
    


    
    
    
    
    
    
    