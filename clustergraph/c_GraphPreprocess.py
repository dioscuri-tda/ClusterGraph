from matplotlib.colors import to_hex
from .NodeStrategy import NodeStrategy
from .EdgeStrategy import EdgeStrategy
from copy import deepcopy


class GraphPreprocess:
    """
    A class for preprocessing a graph by assigning colors and sizes to nodes and edges based on different strategies.

    This class allows for customization of node and edge appearance based on various strategies such as size, coloring, and normalization.
    """

    def __init__(self, graph=None, nodeStrat=None, edgeStrat=None):
        """
        Initializes the GraphPreprocess object with the provided graph and preprocessing strategies.

        Parameters
        ----------
        graph : networkx.Graph, optional
            The graph to preprocess. If None, the graph will not be set initially. Default is None.
        nodeStrat : NodeStrategy, optional
            The strategy for node preprocessing. If None, a new strategy will be created. Default is None.
        edgeStrat : EdgeStrategy, optional
            The strategy for edge preprocessing. If None, a new strategy will be created. Default is None.
        """
        if graph is not None:
            self.graph_prepro = deepcopy(graph)

        self.node_strategy = nodeStrat
        self.edge_strategy = edgeStrat

    def get_graph_prepro(self):
        """
        Returns the preprocessed graph.

        Returns
        -------
        networkx.Graph
            The graph that has been preprocessed, assuming other methods were called first.
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
        """
        Applies both node and edge preprocessing strategies using default parameters.

        This method internally calls `fit_nodes()` and `fit_edges()` with the specified parameters to preprocess the graph.

        Parameters
        ----------
        node_size_strategy : str, optional
            Defines the formula used to normalize the size of nodes. Options are "lin", "log", "exp", or "id". Default is "lin".
        node_type_coloring : str, optional
            Defines how to color nodes. Options are "label" (color by label) or "variable" (color based on a feature). Default is "label".
        node_palette : matplotlib.colors.ListedColormap, optional
            The colormap for nodes. If None, the default colormap will be used. Default is None.
        node_color_labels : list, dict or numpy array, optional
            Object for retrieving colors of nodes. If a list or numpy array is given, it should have the same length as the number of nodes. Default is None.
        node_X : numpy.ndarray, optional
            The dataset to use when coloring nodes by a variable. Default is None.
        node_variable : str or int, optional
            The feature (column index or name) to use for coloring nodes. Default is None.
        node_choiceLabel : str, optional
            Defines how to choose the label when `node_type_coloring` is "label". Options are "max" (most represented label) or "min" (least represented label). Default is "max".
        node_coloring_strategy_var : str, optional
            Defines how the color will change based on the variable value when `node_type_coloring` is "variable". Options are "log", "lin", or "exp". Default is "lin".
        MIN_SIZE_NODE : float, optional
            The minimum size of nodes in the plot. Default is 0.1.
        edge_palette : matplotlib.colors.ListedColormap, optional
            The colormap for edges. Default is None.
        edge_weight : str, optional
            The key in the graph for edge weights. Default is "weight".
        edge_variable : str, optional
            The key in the graph for the variable used to color edges. Default is None.
        edge_norm_weight : str, optional
            The method for normalizing edge sizes. Default is "id", which does not normalize.
        edge_type_coloring : str, optional
            Defines how to color edges. Options are "label" (color by label) or "variable" (color based on a feature). Default is "label".
        edge_color_labels : list, dict or numpy array, optional
            Object for retrieving edge color labels. Default is None.
        edge_coloring_strategy_var : str, optional
            Defines how the color will change based on the edge variable value when `edge_type_coloring` is "variable". Options are "log", "lin", or "exp". Default is "lin".
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
        """
        Preprocesses the nodes of the graph based on the specified strategy.

        This method fits the node preprocessing strategy and applies it to the graph's nodes.

        Parameters
        ----------
        node_size_strategy : str, optional
            Defines the formula used to normalize the size of nodes. Options are "lin", "log", "exp", or "id". Default is "lin".
        node_type_coloring : str, optional
            Defines how to color nodes. Options are "label" (color by label) or "variable" (color based on a feature). Default is "label".
        node_palette : matplotlib.colors.ListedColormap, optional
            The colormap for nodes. Default is None.
        node_color_labels : list, dict or numpy array, optional
            Object for retrieving colors of nodes. Default is None.
        node_X : numpy.ndarray, optional
            The dataset to use when coloring nodes by a variable. Default is None.
        node_variable : str or int, optional
            The feature (column index or name) to use for coloring nodes. Default is None.
        node_choiceLabel : str, optional
            Defines how to choose the label when `node_type_coloring` is "label". Default is "max".
        node_coloring_strategy_var : str, optional
            Defines how the color will change based on the variable value when `node_type_coloring` is "variable". Default is "lin".
        MIN_SIZE_NODE : float, optional
            The minimum size of nodes in the plot. Default is 0.1.
        """
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
        """
        Preprocesses the edges of the graph based on the specified strategy.

        This method fits the edge preprocessing strategy and applies it to the graph's edges.

        Parameters
        ----------
        edge_palette : matplotlib.colors.ListedColormap, optional
            The colormap for edges. Default is None.
        edge_weight : str, optional
            The key in the graph for edge weights. Default is "weight".
        edge_variable : str, optional
            The key in the graph for the variable used to color edges. Default is None.
        edge_norm_weight : str, optional
            The method for normalizing edge sizes. Default is "id", which does not normalize.
        edge_type_coloring : str, optional
            Defines how to color edges. Options are "label" (color by label) or "variable" (color based on a feature). Default is "label".
        edge_color_labels : list, dict or numpy array, optional
            Object for retrieving edge color labels. Default is None.
        edge_coloring_strategy_var : str, optional
            Defines how the color will change based on the edge variable value when `edge_type_coloring` is "variable". Options are "log", "lin", or "exp". Default is "lin".
        """
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
