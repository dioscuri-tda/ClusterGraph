from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd


"""

"weight_plot" in the label for normalized edges


"""


class EdgeStrategy:

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
        """_summary_

        Parameters
        ----------
        graph : networkx.Graph
            Graph to preprocess. Its edges will be colored and normalized.
        palette : Colormap, optional
            Palette used to color edges, by default None
        weight : str, optional
            Key in the graph underwhich the size/weight of edges is stored, by default "weight"
        variable : str, optional
            Key giving access to the continuous variable used to color edges, by default None
        norm_weight : str, optional
            Parameter letting the choice regarding the method used to normalize the size of edges, by default "id" does not normalize
        type_coloring : str, optional
            If “type_coloring” is set to “label”,  each edge should have one label and “color_labels” should not be equal to None. If “type_coloring” is set to “variable”,
            the coloring will be continuous and the color will increase as the value increase. The “variable” should not be None , by default "label"
        color_labels : list, dict or numpy array, optional
            Parameter with labels of each edge. If it is a list, the first index correspond to the first edge. If it is a dictionary, the keys should be edges.
            The values should be labels, the exact colors can be chosen with hexadecimal labels, by default None
        coloring_strategy_var : str, optional
            Parameter letting the choice of how fast the color will change depending on the “variable” 's value, by default 'lin'

        Raises
        ------
        ValueError

        ValueError

        ValueError

        """
        self.myPalette = palette
        self.graph = graph
        self.weight_edges = weight
        self.variable = variable
        self.MAX_VALUE_COLOR = None
        self.MIN_VALUE_COLOR = None
        self.color_labels = color_labels

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

        # WITH LABEL
        if type_coloring == "label":
            if self.myPalette is None:
                self.myPalette = cm.get_cmap(name="tab20b")

            self.fit_color = self.set_color_edges_labels
            self.get_labels()
            self.get_labels_into_hexa()
            self.get_color_edge = self.get_color_edge_unique
            self.getDictEdgeHexa()

        # WITH VARIABLE
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
        """_summary_
        Method launching the methods to set the weight (size) and colors of edges.
        """
        self.set_weight_edges()
        self.fit_color()

    def get_mini_maxi(self):
        """_summary_
        Method returning the minimum and maximum weight of the graph’s edges.
        Returns
        -------
        float, float
            Maximum, minimum weights of edges.
        """
        edges = list(self.graph.edges)
        # Get the maximum and minimum weight of edges
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
        """_summary_
        Method which sets to each edge its normalized weight under the key "weight_plot" in the graph.
        """
        edges = list(self.graph.edges)
        max_weight, min_weight = self.get_mini_maxi()
        for e in edges:
            weight = self.graph.edges[e][self.weight_edges]
            self.graph.edges[e]["weight_plot"] = self.get_weight_e(
                weight, min_weight, max_weight
            )

    def normalize_log_min_max(self, weight, mini_weight, maxi_weight):
        """_summary_
        Method applying a logarithmic normalization of a given weight.
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
        """_summary_
        Method applying a linear normalization of a given weight.
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
        """_summary_
        Methods applying an exponential normalization of a given weight.
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
        """_summary_
        Method applying a maximum normalization of a given weight.
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
        """_summary_
        Method returning the given weight. It is used when no normalization is chosen.
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
        return weight

    def set_color_edges_labels(self):
        """_summary_
        Method browsing edges, in order to set the color of each edge. Used when the coloring is chosen with labels.
        """
        # set labels and their corresponding hexa colors
        for e in self.graph.edges:
            # get_color_edge depends on the number of points in the label
            self.get_color_edge(e)

    # METHODS USED TO SET TO ONE edge ITS CORRESPONDING COLOR AND OTHER DATA CONNECTED WITH COLOR

    # For a given edge add the unique color to it
    def get_color_edge_unique(self, e):
        """_summary_
        Method setting the corresponding label and color to the edge "e" when there is a unique label per edge.

        Parameters
        ----------
        e : tuple
            Edge for which the color should be set in the graph.
        """
        self.graph.edges[e]["color"] = self.EdgeHexa[e]

    # LABELS PREPARATION
    def get_labels(self):
        """_summary_
        Method setting “color_labels” at the list of edges. It is used when no labels are given. Each edge will then have the same color.
        """
        if self.color_labels is None:
            edges = list(self.graph.edges)
            self.color_labels = len(edges) * ["#000000"]

    # TRANSFORMATION OF THE GIVEN LABELS INTO HEXADECIMALS
    # GET HEXADECIMAL VALUE FOR EACH edge
    def get_labels_into_hexa(self):
        """_summary_
        Method transforming the given “color_labels” into a dictionary in which edges are keys and their corresponding hexadecimal colors as values.
          This method calls the right methods in order to achieve such task.
        """
        if type(self.color_labels) is dict:
            keys = list(self.color_labels)

        else:
            keys = range(len(self.color_labels))

        # TEST IF WE NEED TO TRANSFORM LABELS INTO HEXADECIMAL VALUES
        all_hex = True
        for k in keys:
            if not (is_color_like(self.color_labels[k])):
                all_hex = False
                break

        if type(self.color_labels) is dict:
            #  if color_labels is a dictionary and values are not hexadecimals we transform them
            if not (all_hex):
                # Function to transform labels to hexa
                self.labColors = self.dictLabelToHexa()
                # Function to get the dictionary edge Hexa
                self.getDictEdgeHexa = self.edgeColHexa_dictLabHexa

            else:
                self.labColors = self.color_labels
                self.getDictEdgeHexa = self.edgeColHexa_dictEdgeHexa

        # IF WE HAVE A LIST
        else:
            if not (all_hex):
                # Function to transform labels to hexa
                self.labColors = self.listLabelToHexa()

            else:
                self.labColors = self.color_labels
                self.getDictLabelHexaIdentity()

            # Function to get the dictionary edge Hexa
            self.getDictEdgeHexa = self.edgeColHexa_listHexa

    def dictLabelToHexa(self):
        """_summary_
        Method creating a dictionary in which labels are the keys and values are the corresponding hexadecimal colors.
        This method is used when “color_labels” is a dictionary with labels which are not hexadecimal colors.
        Returns
        -------
        dict
            Dictionary in which labels are the keys and values are the corresponding hexadecimal colors.
        """
        values_labels = list(self.color_labels.values())
        keys = list(self.color_labels)
        uniqueLabels = np.unique(values_labels)
        nbLabels = len(uniqueLabels)
        hexLabels = [to_hex(self.myPalette(i / nbLabels)) for i in range(nbLabels + 1)]
        self.dictLabelsCol = dict(zip(uniqueLabels, hexLabels))
        return self.dictLabelsCol

    def listLabelToHexa(self):
        """_summary_
        Method creating a dictionary in which labels are keys and values are the corresponding hexadecimal colors.
        This method is used when “color_labels” is a list of labels which are not hexadecimal colors.
        Returns
        -------
        list
            list in which each element is the color corresponding to the label of the node at the same index in the given list of labels.
        """
        uniqueLabels = np.unique(self.color_labels)
        nbLabels = len(uniqueLabels)
        hexLabels = [to_hex(self.myPalette(i / nbLabels)) for i in range(nbLabels + 1)]
        self.dictLabelsCol = dict(zip(uniqueLabels, hexLabels))
        listLabels = [self.dictLabelsCol[e] for e in self.color_labels]
        return listLabels

    def getDictLabelHexaIdentity(self):
        """_summary_
        Method creating a dictionary in which keys and values are the hexadecimal colors.
        This method is used when “color_labels” is a list with only hexadecimal values.
        """
        uniqueLabels = np.unique(self.color_labels)
        self.dictLabelsCol = dict(zip(uniqueLabels, uniqueLabels))

    def edgeColHexa_dictEdgeHexa(self):
        """_summary_
        Method creating the dictionary "EdgeHexa" in which, edges are keys and the corresponding hexadecimal colors, the values.
        It is used when the given “color_labels” was already a dictonary with only hexadecimal colors as values (labels were already colors).
        """
        self.EdgeHexa = self.labColors

    # if the dictionary has a label per edge
    def edgeColHexa_dictLabHexa(self):
        """_summary_
        Method creating the dictionary "EdgeHexa" in which, edges are keys and the corresponding hexadecimal colors, the values.
        It is used when the given “color_labels” was a dictionary and its values (labels) were not colors.
        """
        keys = list(self.color_labels)
        self.EdgeHexa = {}
        for k in keys:
            self.EdgeHexa[k] = self.labColors[self.color_labels[k]]

    # labColors is a list with hexadecimal values
    def edgeColHexa_listHexa(self):
        """_summary_
        Method creating the dictionary "EdgeHexa" in which, edges are keys and the corresponding hexadecimal colors, the values.
        It is used when the given “color_labels” is a list. It creates the dictionary by associating each index to an edge and the color of its labels.
        """
        edges = list(self.graph.edges)
        # we create the dictionary edge hexadecimal
        self.EdgeHexa = {}
        for i, e in enumerate(edges):
            self.EdgeHexa[e] = self.labColors[i]

    def get_val_var_edge_graph(self, e):
        """_summary_
        Method which, for a given edge, get the edge's variable's value when it is stored in the graph and returns the value.
        Parameters
        ----------
        e : tuple
            Edge for which we want to get the variable's value.

        Returns
        -------
        float
            Edge's variable value.
        """
        return self.graph.edges[e][self.variable]

    def set_color_edges_variable(self):
        """_summary_
        Method which sets the color of each edge depending on the chosen continuous variable of the edge.
        """
        self.set_min_max_mean_var()
        for e in self.graph.edges:
            self.graph.edges[e]["color"] = self.get_color_var(
                self.graph.edges[e]["data_variable"]
            )

    def set_min_max_mean_var(self):
        """_summary_
        Method which  browses edges in order to store the variable's value inside each edge with the key “data_variable” and gets the self.MAX_VALUE_COLOR  and the self.MIN_VALUE_COLOR which correspond to the maximum and minimum values of the variable of the graph among all edges.
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
        """_summary_
        Method which, for a given edge, stores the edge's value inside the graph under “data_variable” and returns the value
        Parameters
        ----------
        e : tuple
            Edge for which we want to store the variable's value.

        Returns
        -------
        float
            Edge's variable value.
        """
        val_intra_e = self.get_val_edge(e)
        self.graph.edges[e]["data_variable"] = val_intra_e
        return val_intra_e

    def get_color_var_exp(self, val):
        """_summary_
        Method transforming a real value in hexadecimal by doing an exponential normalization.

        Parameters
        ----------
        val : float
            Variable's value of an edge.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = (np.exp(val) - np.exp(self.MIN_VALUE_COLOR)) / (
            np.exp(self.MAX_VALUE_COLOR) - np.exp(self.MIN_VALUE_COLOR)
        )
        return to_hex(self.myPalette(color_id))

    def get_color_var_log(self, val):
        """_summary_
        Method transforming a real value in hexadecimal by doing a logarithmic normalization.

        Parameters
        ----------
        val : float
            Variable's value of an edge.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = (np.log10(val) - np.log10(self.MIN_VALUE_COLOR)) / (
            np.log10(self.MAX_VALUE_COLOR) - np.log10(self.MIN_VALUE_COLOR)
        )
        hex = to_hex(self.myPalette(color_id))
        return hex

    def get_color_var_lin(self, val):
        """_summary_
        Method transforming a real value in hexadecimal by doing a linear normalization.

        Parameters
        ----------
        val : float
            Variable's value of an edge.

        Returns
        -------
        str
            Hexadecimal color corresponding to the variable's value.
        """
        color_id = (val - self.MIN_VALUE_COLOR) / (
            self.MAX_VALUE_COLOR - self.MIN_VALUE_COLOR
        )
        return to_hex(self.myPalette(color_id))
