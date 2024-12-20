from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd


class EdgeStrategy:
    """
    Class for managing and preprocessing edge attributes in a graph,
    including edge colors and weights based on various strategies.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to preprocess. Its edges will be colored and normalized.
    palette : Colormap, optional
        Colormap used to color edges. Default is None, which uses a predefined colormap.
    weight : str, optional
        Key in the graph under which the size/weight of edges is stored. Default is "weight".
    variable : str, optional
        Key giving access to the continuous variable used to color edges. Default is None.
    norm_weight : str, optional
        Method used to normalize the weight of edges. Options are "log", "lin", "exp", "id", "max". Default is "id".
    type_coloring : str, optional
        Defines whether edge coloring is based on "label" or "variable". Default is "label".
    color_labels : list, dict or numpy array, optional
        Labels of each edge for coloring. Default is None.
    coloring_strategy_var : str, optional
        Strategy for coloring based on the "variable" key. Options are "log", "lin", "exp". Default is "lin".

    Raises
    ------
    ValueError
        If an invalid option is provided for `norm_weight`, `type_coloring`, or `coloring_strategy_var`.
    """

    def __init__(
        self,
        graph,
        palette=None,
        weight="weight",
        variable=None,
        norm_weight="id",
        type_coloring="label",
        color_labels=None,
        coloring_strategy_var="lin",
    ):
        self.myPalette = palette
        self.graph = graph
        self.weight_edges = weight
        self.variable = variable
        self.MAX_VALUE_COLOR = None
        self.MIN_VALUE_COLOR = None
        self.color_labels = color_labels

        # Select normalization method for edge weights
        if norm_weight == "log":
            self.get_weight_e = self.normalize_log_min_max
        elif norm_weight == "lin":
            self.get_weight_e = self.normalize_lin_min_max
        elif norm_weight == "exp":
            self.get_weight_e = self.normalize_exp_min_max
        elif norm_weight == "id":
            self.get_weight_e = self.identity_weight
        elif norm_weight == "max":
            self.get_weight_e = self.normalize_max
        else:
            raise ValueError(
                "Only 'log', 'lin', 'exp', 'id' and 'max' are accepted as a 'norm_weight' "
            )

        # Edge coloring based on labels
        if type_coloring == "label":
            if self.myPalette is None:
                self.myPalette = cm.get_cmap(name="tab20b")

            self.fit_color = self.set_color_edges_labels
            self.get_labels()
            self.get_labels_into_hexa()
            self.get_color_edge = self.get_color_edge_unique
            self.getDictEdgeHexa()

        # Edge coloring based on variable
        elif type_coloring == "variable":
            self.fit_color = self.set_color_edges_variable

            if self.myPalette is None:
                self.myPalette = cm.get_cmap(name="autumn")

            if coloring_strategy_var == "log":
                self.get_color_var = self.get_color_var_log
            elif coloring_strategy_var == "lin":
                self.get_color_var = self.get_color_var_lin
            elif coloring_strategy_var == "exp":
                self.get_color_var = self.get_color_var_exp
            else:
                raise ValueError(
                    "Only 'log', 'lin' and 'exp' are accepted for the 'coloring_strategy_var' "
                )

            self.get_val_edge = self.get_val_var_edge_graph

        else:
            raise ValueError(
                "Only 'label' and 'variable' are accepted for the 'type_coloring' "
            )

    def fit_edges(self):
        """
        Launches the methods to set the weight (size) and colors of edges.

        Returns
        -------
        None
        """
        self.set_weight_edges()
        self.fit_color()

    def get_mini_maxi(self):
        """
        Returns the minimum and maximum weight of the graph’s edges.

        Returns
        -------
        tuple of float
            Maximum and minimum weights of edges in the graph.
        """
        edges = list(self.graph.edges)
        mini = self.graph.edges[edges[0]][self.weight_edges]
        maxi = mini
        for e in edges:
            weight = self.graph.edges[e][self.weight_edges]
            if weight > maxi:
                maxi = weight
            if weight < mini:
                mini = weight
        return maxi, mini

    def set_weight_edges(self):
        """
        Sets the normalized weight of each edge under the key "weight_plot" in the graph.

        Returns
        -------
        None
        """
        edges = list(self.graph.edges)
        max_weight, min_weight = self.get_mini_maxi()
        for e in edges:
            weight = self.graph.edges[e][self.weight_edges]
            self.graph.edges[e]["weight_plot"] = self.get_weight_e(
                weight, min_weight, max_weight
            )

    def normalize_log_min_max(self, weight, mini_weight, maxi_weight):
        """
        Applies a logarithmic normalization of a given weight.

        Parameters
        ----------
        weight : float
            Weight to normalize.
        mini_weight : float
            Minimum weight in the graph.
        maxi_weight : float
            Maximum weight in the graph.

        Returns
        -------
        float
            The normalized weight.
        """
        return (np.log10(weight) - np.log10(mini_weight)) / (
            np.log10(maxi_weight) - np.log10(mini_weight)
        )

    def normalize_lin_min_max(self, weight, mini_weight, maxi_weight):
        """
        Applies a linear normalization of a given weight.

        Parameters
        ----------
        weight : float
            Weight to normalize.
        mini_weight : float
            Minimum weight in the graph.
        maxi_weight : float
            Maximum weight in the graph.

        Returns
        -------
        float
            The normalized weight.
        """
        return (weight - mini_weight) / (maxi_weight - mini_weight)

    def normalize_exp_min_max(self, weight, mini_weight, maxi_weight):
        """
        Applies an exponential normalization of a given weight.

        Parameters
        ----------
        weight : float
            Weight to normalize.
        mini_weight : float
            Minimum weight in the graph.
        maxi_weight : float
            Maximum weight in the graph.

        Returns
        -------
        float
            The normalized weight.
        """
        return (np.exp(weight) - np.exp(mini_weight)) / (
            np.exp(maxi_weight) - np.exp(mini_weight)
        )

    def normalize_max(self, weight, mini_weight, maxi_weight):
        """
        Applies a maximum normalization of a given weight.

        Parameters
        ----------
        weight : float
            Weight to normalize.
        mini_weight : float
            Minimum weight in the graph.
        maxi_weight : float
            Maximum weight in the graph.

        Returns
        -------
        float
            The normalized weight.
        """
        return weight / maxi_weight

    def identity_weight(self, weight, mini_weight, maxi_weight):
        """
        Returns the given weight without any normalization.

        Parameters
        ----------
        weight : float
            Weight to normalize.
        mini_weight : float
            Minimum weight in the graph.
        maxi_weight : float
            Maximum weight in the graph.

        Returns
        -------
        float
            The weight as is.
        """
        return weight

    def set_color_edges_labels(self):
        """
        Sets the color of each edge based on its label.

        Returns
        -------
        None
        """
        for e in self.graph.edges:
            self.get_color_edge(e)

    def get_color_edge_unique(self, e):
        """
        Sets a unique color for each edge based on its label.

        Parameters
        ----------
        e : tuple
            The edge for which the color should be set.

        Returns
        -------
        None
        """
        self.graph.edges[e]["color"] = self.EdgeHexa[e]

    def get_labels(self):
        """
        Sets the color labels for the edges. If no labels are provided,
        assigns black as the default color.

        Returns
        -------
        None
        """
        if self.color_labels is None:
            edges = list(self.graph.edges)
            self.color_labels = len(edges) * ["#000000"]

    def get_labels_into_hexa(self):
        """
        Transforms the color labels into hexadecimal color codes and stores them in `EdgeHexa`.

        Returns
        -------
        None
        """
        if type(self.color_labels) is dict:
            keys = list(self.color_labels)
        else:
            keys = range(len(self.color_labels))

        all_hex = True
        for k in keys:
            if not (is_color_like(self.color_labels[k])):
                all_hex = False
                break

        if type(self.color_labels) is dict:
            if not all_hex:
                self.labColors = self.dictLabelToHexa()
                self.getDictEdgeHexa = self.edgeColHexa_dictLabHexa
            else:
                self.labColors = self.color_labels
                self.getDictEdgeHexa = self.edgeColHexa_dictEdgeHexa
        else:
            if not all_hex:
                self.labColors = self.listLabelToHexa()
            else:
                self.labColors = self.color_labels
                self.getDictLabelHexaIdentity()
            self.getDictEdgeHexa = self.edgeColHexa_listHexa

    def dictLabelToHexa(self):
        """
        Converts labels in the form of a dictionary to hexadecimal colors.

        Returns
        -------
        dict
            A dictionary mapping labels to their corresponding hexadecimal color.
        """
        values_labels = list(self.color_labels.values())
        keys = list(self.color_labels)
        uniqueLabels = np.unique(values_labels)
        nbLabels = len(uniqueLabels)
        hexLabels = [to_hex(self.myPalette(i / nbLabels)) for i in range(nbLabels + 1)]
        self.dictLabelsCol = dict(zip(uniqueLabels, hexLabels))
        return self.dictLabelsCol

    def listLabelToHexa(self):
        """
        Converts labels in the form of a list to hexadecimal colors.

        Returns
        -------
        list
            A list of hexadecimal colors corresponding to each label.
        """
        uniqueLabels = np.unique(self.color_labels)
        nbLabels = len(uniqueLabels)
        hexLabels = [to_hex(self.myPalette(i / nbLabels)) for i in range(nbLabels + 1)]
        self.dictLabelsCol = dict(zip(uniqueLabels, hexLabels))
        listLabels = [self.dictLabelsCol[e] for e in self.color_labels]
        return listLabels

    def getDictLabelHexaIdentity(self):
        """
        Creates a dictionary where each label is mapped to its own hexadecimal color.

        Returns
        -------
        None
        """
        uniqueLabels = np.unique(self.color_labels)
        self.dictLabelsCol = dict(zip(uniqueLabels, uniqueLabels))

    def edgeColHexa_dictEdgeHexa(self):
        """
        Creates a dictionary mapping edges to their corresponding hexadecimal color.

        Returns
        -------
        None
        """
        self.EdgeHexa = self.labColors

    def edgeColHexa_dictLabHexa(self):
        """
        Creates a dictionary mapping edges to their corresponding hexadecimal color, using labels.

        Returns
        -------
        None
        """
        keys = list(self.color_labels)
        self.EdgeHexa = {}
        for k in keys:
            self.EdgeHexa[k] = self.labColors[self.color_labels[k]]

    def edgeColHexa_listHexa(self):
        """
        Creates a dictionary mapping edges to their corresponding hexadecimal color, based on a list of colors.

        Returns
        -------
        None
        """
        edges = list(self.graph.edges)
        self.EdgeHexa = {}
        for i, e in enumerate(edges):
            self.EdgeHexa[e] = self.labColors[i]

    def get_val_var_edge_graph(self, e):
        """
        Retrieves the value of the variable for a given edge stored in the graph.

        Parameters
        ----------
        e : tuple
            The edge for which to retrieve the variable value.

        Returns
        -------
        float
            The variable's value for the edge.
        """
        return self.graph.edges[e][self.variable]

    def set_color_edges_variable(self):
        """
        Sets the color of each edge based on the value of a continuous variable.

        Returns
        -------
        None
        """
        self.set_min_max_mean_var()
        for e in self.graph.edges:
            self.graph.edges[e]["color"] = self.get_color_var(
                self.graph.edges[e]["data_variable"]
            )

    def set_min_max_mean_var(self):
        """
        Sets the minimum and maximum values of the edge variable and stores the variable's value in each edge.

        Returns
        -------
        None
        """
        edges = list(self.graph.edges)
        MIN_VALUE = self.get_set_val_var_edge(edges[0])
        MAX_VALUE = MIN_VALUE
        for edge in self.graph.edges:
            mean_edge = self.get_set_val_var_edge(edge)
            if mean_edge > MAX_VALUE:
                MAX_VALUE = mean_edge
            if mean_edge < MIN_VALUE:
                MIN_VALUE = mean_edge

        self.MAX_VALUE_COLOR = MAX_VALUE
        self.MIN_VALUE_COLOR = MIN_VALUE

    def get_set_val_var_edge(self, e):
        """
        Stores the variable's value for an edge under the key "data_variable" and returns the value.

        Parameters
        ----------
        e : tuple
            The edge for which to store the variable value.

        Returns
        -------
        float
            The variable's value for the edge.
        """
        val_intra_e = self.get_val_edge(e)
        self.graph.edges[e]["data_variable"] = val_intra_e
        return val_intra_e

    def get_color_var_exp(self, val):
        """
        Transforms a value into a hexadecimal color using exponential normalization.

        Parameters
        ----------
        val : float
            The variable value of an edge.

        Returns
        -------
        str
            The hexadecimal color corresponding to the variable value.
        """
        color_id = (np.exp(val) - np.exp(self.MIN_VALUE_COLOR)) / (
            np.exp(self.MAX_VALUE_COLOR) - np.exp(self.MIN_VALUE_COLOR)
        )
        return to_hex(self.myPalette(color_id))

    def get_color_var_log(self, val):
        """
        Transforms a value into a hexadecimal color using logarithmic normalization.

        Parameters
        ----------
        val : float
            The variable value of an edge.

        Returns
        -------
        str
            The hexadecimal color corresponding to the variable value.
        """
        color_id = (np.log10(val) - np.log10(self.MIN_VALUE_COLOR)) / (
            np.log10(self.MAX_VALUE_COLOR) - np.log10(self.MIN_VALUE_COLOR)
        )
        hex = to_hex(self.myPalette(color_id))
        return hex

    def get_color_var_lin(self, val):
        """
        Transforms a value into a hexadecimal color using linear normalization.

        Parameters
        ----------
        val : float
            The variable value of an edge.

        Returns
        -------
        str
            The hexadecimal color corresponding to the variable value.
        """
        color_id = (val - self.MIN_VALUE_COLOR) / (
            self.MAX_VALUE_COLOR - self.MIN_VALUE_COLOR
        )
        return to_hex(self.myPalette(color_id))
