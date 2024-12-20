from matplotlib.colors import to_hex
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
import pandas as pd


class NodeStrategy:

    def __init__(
        self,
        graph,
        size_strategy="lin",
        type_coloring="label",
        palette=None,
        color_labels=None,
        X=None,
        variable=None,
        choiceLabel="max",
        coloring_strategy_var="lin",
        MIN_SIZE_NODE=0.1,
    ):
        """
        Initialize the NodeStrategy class for preprocessing graph nodes with colors and sizes.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to preprocess, including setting colors and sizes for nodes.
        size_strategy : str, optional
            Defines the formula for normalizing node sizes. Options are "lin", "log", "exp", or "id". Default is "lin".
        type_coloring : str, optional
            Defines the coloring method for nodes. Options are "label" or "variable". Default is "label".
        palette : matplotlib.colors.ListedColormap, optional
            The colormap used to color nodes. Default is None.
        color_labels : list, dict, or numpy array, optional
            Labels or colors for each node, either as a list or dictionary. Default is None.
        X : numpy ndarray, optional
            Dataset used to compute variable-based coloring. Default is None.
        variable : str or int, optional
            Feature used for variable-based coloring. Can be column name (str) or index (int).
        choiceLabel : str, optional
            If "max" or "min" is chosen, the label is selected based on the most or least frequent label.
            Default is "max".
        coloring_strategy_var : str, optional
            Defines how the color changes based on variable values. Options are "log", "lin", or "exp". Default is "lin".
        MIN_SIZE_NODE : float, optional
            Minimum size of nodes in the plot. Default is 0.1.

        Raises
        ------
        ValueError
            If invalid options are provided for size_strategy, type_coloring, or coloring_strategy_var.
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

        if choiceLabel == "max":
            self.labelChoice = np.argmax
        elif choiceLabel == "min":
            self.labelChoice = np.argmin
        else:
            raise ValueError("The choice of label must be 'min' or 'max' ")

        if size_strategy == "log":
            self.get_size_node = self.log_size
        elif size_strategy == "lin":
            self.get_size_node = self.linear_size
        elif size_strategy == "exp":
            self.get_size_node = self.expo_size
        elif size_strategy == "id":
            self.get_size_node = self.id_size
        else:
            raise ValueError(
                "Only 'log', 'lin' and 'exp' are accepted as a size_strategy "
            )

        # CHOICE OF STRATEGY TO GET THE WAY OF COLORING

        # WITH LABEL
        if type_coloring == "label":
            if self.myPalette is None:
                self.myPalette = cm.get_cmap(name="tab20c")

            self.fit_color = self.set_color_nodes_labels
            self.get_labels()
            self.get_labels_into_hexa()

            # Choice of function to set colors to each node
            if color_labels is None or (
                len(color_labels) == len(list(self.graph.nodes))
            ):
                self.get_color_node = self.get_color_node_unique
                self.getDictNodeHexa()

            elif len(color_labels) > len(list(self.graph.nodes)):
                self.get_color_node = self.get_color_node_points_covered

            else:
                raise ValueError("Less labels than nodes or points in nodes ")

        # WITH VARIABLE
        elif type_coloring == "variable":
            self.fit_color = self.set_color_nodes_variable

            if self.myPalette is None:
                self.myPalette = cm.get_cmap(name="YlOrBr")

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

            if not (X is None):
                if isinstance(X, pd.DataFrame):
                    self.get_val_node = self.get_val_var_node_Xpand
                elif isinstance(X, np.ndarray):
                    self.get_val_node = self.get_val_var_node_Xnum
            else:
                self.get_val_node = self.get_val_var_node_graph

        else:
            raise ValueError(
                "Only 'label' and 'variable' are accepted for the 'type_coloring' "
            )

    def fit_nodes(self):
        """
        Set the size and color of nodes based on the chosen strategies.

        This method updates the size and color of all nodes in the graph.
        """
        self.set_size_nodes()
        self.fit_color()

    def get_mini_maxi(self):
        """
        Calculate the maximum and minimum size (number of points covered) of nodes in the graph.

        Returns
        -------
        int, int
            The maximum and minimum sizes of nodes in the graph.
        """
        nodes = list(self.graph.nodes)
        mini = len(self.graph.nodes[nodes[0]]["points_covered"])
        maxi = mini
        for node in nodes:
            size = len(self.graph.nodes[node]["points_covered"])
            if size > maxi:
                maxi = size
            if size < mini:
                mini = size
        return maxi, mini

    def set_size_nodes(self):
        """
        Set the size of each node in the plot.

        This method assigns the appropriate size to each node based on the number of points covered.
        """
        nodes = list(self.graph.nodes)
        max_size, min_size = self.get_mini_maxi()
        for node in nodes:
            size = len(self.graph.nodes[node]["points_covered"])
            self.graph.nodes[node]["size_plot"] = self.get_size_node(
                size, min_size, max_size
            )

    def log_size(self, size, mini_size, maxi_size):
        """
        Logarithmically normalize the size of a node.

        Parameters
        ----------
        size : int
            The size/number of points covered by a node.
        mini_size : int
            Minimum size of a node.
        maxi_size : int
            Maximum size of a node.

        Returns
        -------
        float
            The logarithmically normalized size of the node.
        """
        return np.log10(1 + size / maxi_size)

    def linear_size(self, size, mini_size, maxi_size):
        """
        Linearly normalize the size of a node.

        Parameters
        ----------
        size : int
            The size/number of points covered by a node.
        mini_size : int
            Minimum size of a node.
        maxi_size : int
            Maximum size of a node.

        Returns
        -------
        float
            The linearly normalized size of the node.
        """
        return (size - mini_size) / (maxi_size - mini_size)

    def expo_size(self, size, mini_size, maxi_size):
        """
        Exponentially normalize the size of a node.

        Parameters
        ----------
        size : int
            The size/number of points covered by a node.
        mini_size : int
            Minimum size of a node.
        maxi_size : int
            Maximum size of a node.

        Returns
        -------
        float
            The exponentially normalized size of the node.
        """
        return (np.exp(size) - np.exp(mini_size)) / (
            np.exp(maxi_size) - np.exp(mini_size)
        )

    def id_size(self, size, mini_size, maxi_size):
        """
        Return the same size for every node.

        Parameters
        ----------
        size : int
            The size/number of points covered by a node.
        mini_size : int
            Minimum size of a node.
        maxi_size : int
            Maximum size of a node.

        Returns
        -------
        float
            Always returns 1 for every node.
        """
        return 1

    def set_color_nodes_labels(self):
        """
        Set the color of each node based on its label.

        This method assigns colors to nodes using the color_labels attribute.
        """
        for node in self.graph.nodes:
            self.get_color_node(node)

    def get_color_node_unique(self, n):
        """
        Assign a color to a node when there is a unique label for each node.

        Parameters
        ----------
        n : int
            The node for which to assign a color.
        """
        self.graph.nodes[n]["color"] = self.NodeHexa[n]

    def get_labels(self):
        """
        Set the color labels for nodes.

        If no labels are provided, each node is assigned a unique color.
        """
        if self.color_labels is None:
            self.color_labels = list(self.graph.nodes)

    def get_labels_into_hexa(self):
        """
        Convert the node labels into hexadecimal color values.

        This method uses the matplotlib colormap to assign colors to labels.
        """
        labels = np.unique(self.color_labels)
        self.dictLabelsCol = {
            k: to_hex(self.myPalette(k / len(labels))) for k in range(len(labels))
        }
        self.NodeHexa = {
            n: self.dictLabelsCol[l]
            for n, l in zip(self.graph.nodes, self.color_labels)
        }

    def get_color_node_points_covered(self, n):
        """
        Assign a color to a node based on the number of points it covers.

        Parameters
        ----------
        n : int
            The node for which to assign a color.
        """
        color = self.get_val_node(n)
        self.graph.nodes[n]["color"] = to_hex(color)

    def set_color_nodes_variable(self):
        """
        Set the color of each node based on the variable's value.

        This method assigns colors to nodes using the specified variable-based coloring strategy.
        """
        for node in self.graph.nodes:
            self.get_color_node(node)

    def get_color_var_log(self, n):
        """
        Assign a color to a node based on the logarithm of its variable value.

        Parameters
        ----------
        n : int
            The node for which to assign a color.
        """
        node_value = self.get_val_node(n)
        normalized_value = np.log10(1 + node_value)
        self.graph.nodes[n]["color"] = to_hex(self.myPalette(normalized_value))

    def get_color_var_lin(self, n):
        """
        Assign a color to a node based on the linear normalization of its variable value.

        Parameters
        ----------
        n : int
            The node for which to assign a color.
        """
        node_value = self.get_val_node(n)
        normalized_value = node_value / self.MAX_VALUE_COLOR
        self.graph.nodes[n]["color"] = to_hex(self.myPalette(normalized_value))

    def get_color_var_exp(self, n):
        """
        Assign a color to a node based on the exponential normalization of its variable value.

        Parameters
        ----------
        n : int
            The node for which to assign a color.
        """
        node_value = self.get_val_node(n)
        normalized_value = (np.exp(node_value) - np.exp(0)) / (
            np.exp(self.MAX_VALUE_COLOR) - np.exp(0)
        )
        self.graph.nodes[n]["color"] = to_hex(self.myPalette(normalized_value))

    def get_val_var_node_graph(self, n):
        """
        Retrieve the variable value of a node from the graph.

        Parameters
        ----------
        n : int
            The node for which to retrieve the variable value.

        Returns
        -------
        float
            The value of the variable for the node.
        """
        return self.graph.nodes[n].get(self.variable, 0)

    def get_val_var_node_Xnum(self, n):
        """
        Retrieve the variable value of a node from the numeric dataset.

        Parameters
        ----------
        n : int
            The node for which to retrieve the variable value.

        Returns
        -------
        float
            The value of the variable for the node.
        """
        return self.X[n][self.variable]

    def get_val_var_node_Xpand(self, n):
        """
        Retrieve the variable value of a node from the expanded dataset.

        Parameters
        ----------
        n : int
            The node for which to retrieve the variable value.

        Returns
        -------
        float
            The value of the variable for the node.
        """
        return self.X.loc[n, self.variable]
