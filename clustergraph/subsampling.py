import networkx as nx
import numpy as np


class Subsampling:
    """
    Class that performs subsampling on clusters of data, either from a graph or provided as input.

    Parameters
    ----------
    clusters : networkx.Graph or list of lists
        The clusters to subsample. If a graph is provided, clusters are extracted from the graph.
    variable_clusters : str, optional
        The node attribute to be used to extract clusters from the graph. Default is 'points_covered'.
    perc : float, optional
        The percentage of each cluster to retain after subsampling. Should be between 0 and 1. Default is 0.5.
    seed : int or None, optional
        The random seed for reproducibility. Default is None.

    Attributes
    ----------
    perc : float
        The percentage of the clusters to subsample.
    clusters : list of lists
        The clusters to subsample, either from the graph or provided directly.
    subsampled_clusters : ndarray
        The result of subsampling the clusters.
    dict_old_new_indices : dict
        A dictionary mapping old indices to new indices after subsampling.
    dict_new_old_indices : dict
        A dictionary mapping new indices to old indices.
    X_restricted : ndarray
        The restricted dataset after subsampling.
    """

    def __init__(
        self, clusters, variable_clusters="points_covered", perc=0.5, seed=None
    ):
        """
        Initializes the Subsampling object.

        Parameters
        ----------
        clusters : networkx.Graph or list of lists
            The clusters to subsample. If a graph is provided, clusters are extracted from the graph.
        variable_clusters : str, optional
            The node attribute to be used to extract clusters from the graph. Default is 'points_covered'.
        perc : float, optional
            The percentage of each cluster to retain after subsampling. Default is 0.5.
        seed : int or None, optional
            The random seed for reproducibility. Default is None.

        Raises
        ------
        ValueError
            If the percentage (`perc`) is not between 0 and 1.
        """
        if perc < 0 or perc > 1:
            raise ValueError("Percentage should belong to the interval 0, 1.")

        self.perc = perc

        if isinstance(clusters, nx.classes.graph.Graph):
            self.clusters = self.get_clusters_from_graph(clusters, variable_clusters)
        else:
            self.clusters = clusters

        self.subsampling_clusters()

        if seed is not None:
            np.random.seed(seed)

    def get_clusters_from_graph(self, g_clusters, variable_clusters):
        """
        Extracts the clusters from a graph based on a node attribute.

        Parameters
        ----------
        g_clusters : networkx.Graph
            The graph from which clusters will be extracted.
        variable_clusters : str
            The node attribute to use for extracting clusters.

        Returns
        -------
        ndarray
            A numpy array containing the clusters from the graph.
        """
        return np.array(
            [g_clusters.nodes[n][variable_clusters] for n in g_clusters.nodes]
        )

    def subsampling_clusters(self):
        """
        Subsamples the clusters based on the specified percentage.

        This method creates a subsampled version of each cluster, where a percentage of the original
        elements are randomly selected without replacement.

        Returns
        -------
        ndarray
            A numpy array containing the subsampled clusters.
        """
        subclusters = []
        for sublist in self.clusters:
            sublist_size = len(sublist)
            sample_size = max(1, int(sublist_size * self.perc))
            sampled_items = np.random.choice(sublist, size=sample_size, replace=False)
            subclusters.append(sampled_items)

        self.subsampled_clusters = np.array(subclusters)
        return self.subsampled_clusters

    def data_transformation(self, X):
        """
        Transforms the dataset `X` by selecting only the rows corresponding to the subsampled clusters.

        The method creates two dictionaries to map between old indices (from the original dataset)
        and new indices (from the restricted dataset), and returns the restricted dataset.

        Parameters
        ----------
        X : ndarray
            The original dataset from which data will be selected based on subsampled clusters.

        Returns
        -------
        None
            This method modifies the object in place and stores the transformed dataset in `self.X_restricted`.
        """
        restrict_indices = []
        for l in self.subsampled_clusters:
            restrict_indices.extend(l)

        restrict_indices = np.unique(restrict_indices)

        # Dictionary mapping old indices to new indices in the restricted dataset
        self.dict_old_new_indices = {
            value: index for index, value in enumerate(restrict_indices)
        }

        # Dictionary mapping new indices to old indices in the original dataset
        self.dict_new_old_indices = {
            index: value for index, value in enumerate(restrict_indices)
        }

        # Creating the restricted dataset
        self.X_restricted = X[restrict_indices, :]
