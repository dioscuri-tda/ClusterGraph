from matplotlib.colors import to_hex
from .NodeStrategy import NodeStrategy
from .EdgeStrategy import EdgeStrategy
from copy import deepcopy


class GraphPreprocess:

    def __init__(self, graph, nodeStrat=None, edgeStrat=None, renderer=None):
        self.graph = deepcopy(graph)
        self.node_strategy = nodeStrat
        self.edge_strategy = edgeStrat
        self.renderer = renderer

    def fit_graph(self):
        """_summary_
        Method which calls the get_distance method with all the demanded parameters
        Parameters
        ----------
        point_1 : _type_ int or numpy darray
            _description_ Corresponds to the index of a point in the dataset or to the vector directly
        point_2 : _type_ int or numpy darray
            _description_ Corresponds to the index of a point in the dataset or to the vector directly

        """
        self.fit_nodes()
        self.fit_edges()

    def fit_nodes(
        self,
        size_strategy="log",
        type_coloring="label",
        palette=None,
        color_labels=None,
        X=None,
        variable=None,
        choiceLabel="max",
        coloring_strategy_var="lin",
        MIN_SIZE_NODE=0.1,
    ):
        if self.node_strategy is None:
            self.node_strategy = NodeStrategy(
                self.graph,
                size_strategy=size_strategy,
                type_coloring=type_coloring,
                palette=palette,
                color_labels=color_labels,
                X=X,
                variable=variable,
                choiceLabel=choiceLabel,
                coloring_strategy_var=coloring_strategy_var,
                MIN_SIZE_NODE=MIN_SIZE_NODE,
            )

        self.node_strategy.fit_nodes()

    def fit_edges(
        self,
        palette=None,
        weight="label",
        variable=None,
        norm_weight="lin",
        type_coloring="label",
        color_labels=None,
        coloring_strategy_var="lin",
    ):

        if self.edge_strategy is None:
            self.edge_strategy = EdgeStrategy(
                self.graph,
                palette=palette,
                weight=weight,
                variable=variable,
                norm_weight=norm_weight,
                type_coloring=type_coloring,
                color_labels=color_labels,
                coloring_strategy_var=coloring_strategy_var,
            )
        self.edge_strategy.fit_edges()
