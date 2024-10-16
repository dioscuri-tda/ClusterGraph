from matplotlib.colors import to_hex
from .NodeStrategy import NodeStrategy
from .EdgeStrategy import EdgeStrategy
from copy import deepcopy


class GraphPreprocess:

    def __init__(self, graph=None, nodeStrat=None, edgeStrat=None):
        """_summary_

        Parameters
        ----------
        graph :  networkx.Graph, optional
             Graph to preprocess. This class will set colors and sizes to edges and nodes of this graph.
        nodeStrat :  NodeStrategy , optional
             The preprocessing configuration of edges can be given if it was created outside this class, by default None
        edgeStrat :  EdgeStrategy , optional
             The preprocessing configuration of edges can be given if it was created outside this class, by default None
        """
        if graph != None:
            self.graph_prepro = deepcopy(graph)

        self.node_strategy = nodeStrat
        self.edge_strategy = edgeStrat

    def get_graph_prepro(self):
        """
        Returns
        -------
        networkx.Graph
            Returns the graph of the class. This graph is preprocessed is the other methods were called first.
        """
        return self.graph_prepro

    def color_graph(
        self,
        node_size_strategy="log",
        node_type_coloring="label",
        node_palette=None,
        node_color_labels=None,
        node_X=None,
        node_variable=None,
        node_choiceLabel="max",
        node_coloring_strategy_var="lin",
        MIN_SIZE_NODE=0.1,
        edge_palette=None,
        edge_weight="weight",
        edge_variable=None,
        edge_norm_weight="id",
        edge_type_coloring="label",
        edge_color_labels=None,
        edge_coloring_strategy_var="lin",
    ):
        """_summary_
        Method which launch the fit_nodes() and fit_edges() methods with default parameters.
        """
        self.fit_nodes(
            node_size_strategy=node_size_strategy,
            node_type_coloring=node_type_coloring,
            node_palette=node_palette,
            node_color_labels=node_color_labels,
            node_X=node_X,
            node_variable=node_variable,
            node_choiceLabel=node_choiceLabel,
            node_coloring_strategy_var=node_coloring_strategy_var,
            MIN_SIZE_NODE=MIN_SIZE_NODE,
        )
        self.fit_edges(
            edge_palette=edge_palette,
            edge_weight=edge_weight,
            edge_variable=edge_variable,
            edge_norm_weight=edge_norm_weight,
            edge_type_coloring=edge_type_coloring,
            edge_color_labels=edge_color_labels,
            edge_coloring_strategy_var=edge_coloring_strategy_var,
        )

    def fit_nodes(
        self,
        node_size_strategy="log",
        node_type_coloring="label",
        node_palette=None,
        node_color_labels=None,
        node_X=None,
        node_variable=None,
        node_choiceLabel="max",
        node_coloring_strategy_var="lin",
        MIN_SIZE_NODE=0.1,
    ):
        """_summary_

        Parameters
        ----------
        node_size_strategy : str, optional
            Defines the formula which is used to normalize the size of nodes. It can be "lin", "log", "exp" or "id". , by default "lin"
        node_type_coloring : str, optional
            Defines the way to add colors to nodes. It can be with "label" or with "variable". For "label", colors are added from a given list or dictionary.
            If "variable" is chosen, the color is chosen depending from a feature of the dataset. It can be the average of a given feature inside each node for example. , by default "label"
        node_palette : matplotlib.colors.ListedColormap , optional
            The colormap from which color will be chosen for nodes , by default None
        node_color_labels : list, dict or numpy array, optional
            Object from which the colors of nodes will be retrieved. The exact colors can be chosen by giving hexadecimal values.
            If a list or a numpy array is given and has the same length than the number of nodes, each node will be associated to a label. If the list is longer, the label associated to each node will depend on the which labels are represented inside each nodes by the points covered.
            If a dictionary is chosen, the keys should be the nodes and the values the labels on which the color will be chosen.
            , by default None
        node_X : numpy darray, optional
            The dataset from which the value inside each node will be taken if the type_coloring is set to "variable"., by default None
        node_variable : str or int , optional
            If the parameter type_coloring is set to "variable", this parameter is giving access to the good feature in the dataset. It can be an index or the name of the variable.
              , by default None
        node_choiceLabel : str, optional
            Can be "max" or "min". When the parameter "type_coloring" is set to "label", it defines the way to choose the label inside each node to color them. If "max" is chosen, the most represented label inside each node will be chosen. If "min" is chosen it will be the least represented label.
            , by default "max"
        node_coloring_strategy_var : str, optional
            Can be "log", "lin" or "exp". When the parameter "type_coloring" is set to "variable", this parameter represents how fast color will changed between nodes depending on the variable's average value inside each node.
            For example if "exp" is chosen, a slight change of average value between two nodes will represent an important change of colors.
            , by default 'lin'
        MIN_SIZE_NODE : float, optional
            The minimum size of nodes in the plot, by default 0.1
        """
        # if self.node_strategy is None:
        self.node_strategy = NodeStrategy(
            self.graph_prepro,
            size_strategy=node_size_strategy,
            type_coloring=node_type_coloring,
            palette=node_palette,
            color_labels=node_color_labels,
            X=node_X,
            variable=node_variable,
            choiceLabel=node_choiceLabel,
            coloring_strategy_var=node_coloring_strategy_var,
            MIN_SIZE_NODE=MIN_SIZE_NODE,
        )

        self.node_strategy.fit_nodes()

    def fit_edges(
        self,
        edge_palette=None,
        edge_weight="weight",
        edge_variable=None,
        edge_norm_weight="id",
        edge_type_coloring="label",
        edge_color_labels=None,
        edge_coloring_strategy_var="lin",
    ):
        """_summary_

        Parameters
        ----------
        graph : networkx.Graph
            Graph to preprocess. Its edges will be colored and normalized.
        edge_palette : Colormap, optional
            Palette used to color edges, by default None
        edge_weight : str, optional
            Key in the graph underwhich the size/weight of edges is stored, by default "weight"
        edge_variable : str, optional
            Key giving access to the continuous variable used to color edges, by default None
        edge_norm_weight : str, optional
            Parameter letting the choice regarding the method used to normalize the size of edges, by default "id" does not normalize
        edge_type_coloring : str, optional
            If “type_coloring” is set to “label”,  each edge should have one label and “color_labels” should not be equal to None. If “type_coloring” is set to “variable”,
            the coloring will be continuous and the color will increase as the value increase. The “variable” should not be None , by default "label"
        edge_color_labels : list or dict, optional
            Parameter with labels of each edge. If it is a list, the first index correspond to the first edge. If it is a dictionary, the keys should be edges.
            The values should be labels, the exact colors can be chosen with hexadecimal labels, by default None
        edge_coloring_strategy_var : str, optional
            Parameter letting the choice of how fast the color will change depending on the “variable” 's value., by default 'lin'
        """

        # if self.edge_strategy is None:
        self.edge_strategy = EdgeStrategy(
            self.graph_prepro,
            palette=edge_palette,
            weight=edge_weight,
            variable=edge_variable,
            norm_weight=edge_norm_weight,
            type_coloring=edge_type_coloring,
            color_labels=edge_color_labels,
            coloring_strategy_var=edge_coloring_strategy_var,
        )
        self.edge_strategy.fit_edges()
