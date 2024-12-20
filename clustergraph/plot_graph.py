import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import interact, IntSlider
from IPython.display import display, HTML


def draw_graph_pie(
    graph,
    nb_edges=None,
    edge_variable="weight_plot",
    draw_edge_labels=True,
    scale_nodes=True,
    size_nodes=0.05,
    random_state=42,
    ax=None,
    **kwargs
):
    """
    Draw a graph with pie charts at each node representing its attributes.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be drawn.
    nb_edges : int, optional
        The number of edges to display. If specified, the graph is truncated to include only the smallest `nb_edges` edges.
    edge_variable : str, optional
        The edge attribute used for edge labels. Defaults to 'weight_plot'.
    draw_edge_labels : bool, optional
        If True, edge labels are drawn. Defaults to True.
    scale_nodes : bool, optional
        If True, node sizes are scaled according to the attribute 'size_plot'. Defaults to True.
    size_nodes : float, optional
        The baseline size of nodes if `scale_nodes` is False. Defaults to 0.05.
    random_state : int, optional
        The random state for the node positioning. Defaults to 42.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the graph. If None, the current axes are used.
    **kwargs : keyword arguments
        Additional arguments passed to `networkx.draw_networkx`.

    """
    if ax is None:
        ax = plt.gca()

    G = graph.copy()
    if nb_edges is not None:
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0))[
            :nb_edges
        ]
        G.clear_edges()
        G.add_edges_from(edges)

    edge_colors = [data["color"] for _, _, data in G.edges(data=True)]
    pos = nx.spring_layout(G, seed=random_state)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors)

    for node, data in G.nodes(data=True):
        attributes = G.nodes[node]["data_perc_labels"]
        keys = list(attributes)
        attrs = [attributes[k] for k in keys]

        if scale_nodes:
            plt.pie(
                attrs,  # s.t. all wedges have equal size
                center=pos[node],
                colors=[k for k in keys],
                radius=max(data["size_plot"] * 0.3, size_nodes),
                frame=True,
            )
        else:
            plt.pie(
                attrs,  # s.t. all wedges have equal size
                center=pos[node],
                colors=[k for k in keys],
                radius=size_nodes,
                frame=True,
            )

    # Display edge labels
    if draw_edge_labels:
        edge_labels = {
            (u, v): "{:.2f}".format(data[edge_variable])
            for u, v, data in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="black", font_size=9
        )


def draw_graph(
    graph,
    nb_edges=None,
    edge_variable="weight_plot",
    draw_edge_labels=True,
    scale_nodes=True,
    size_nodes=1000,
    random_state=42,
    precision=2,
    ax=None,
    **kwargs
):
    """
    Plot a graph with specified nodes and edges, with optional sorting of edges.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be displayed.
    nb_edges : int, optional
        The number of edges to display. If specified, only the first `nb_edges` edges (sorted by weight) are shown.
    edge_variable : str, optional
        The edge attribute to be used for edge labels. Defaults to 'weight_plot'.
    draw_edge_labels : bool, optional
        If True, edge labels are drawn. Defaults to True.
    scale_nodes : bool, optional
        If True, node sizes are scaled based on the 'size_plot' attribute. Defaults to True.
    size_nodes : int, optional
        The baseline size of nodes. Larger values make nodes bigger. Defaults to 1000.
    random_state : int or None, optional
        Random seed for the node layout. Defaults to 42.
    precision : int, optional
        Number of decimal places for edge labels. Defaults to 2.
    ax : matplotlib.axes.Axes, optional
        The axes to draw the graph on. If None, the current axes are used.
    **kwargs : keyword arguments
        Additional arguments passed to `networkx.draw_networkx`.

    """
    if ax is None:
        ax = plt.gca()

    G = graph.copy()

    if nb_edges is not None:
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0))[
            :nb_edges
        ]
        G.clear_edges()
        G.add_edges_from(edges)

    # Obtain node and edge colors
    node_colors = [data["color"] for _, data in G.nodes(data=True)]
    edge_colors = [data["color"] for _, _, data in G.edges(data=True)]

    # Obtain node sizes
    if scale_nodes:
        node_sizes = [data["size_plot"] * size_nodes for _, data in G.nodes(data=True)]
    else:
        node_sizes = [size_nodes for _ in G.nodes()]

    # Create the graph visualization
    if "pos" not in kwargs:
        pos = nx.spring_layout(G, seed=random_state)
        kwargs["pos"] = pos
    nx.draw_networkx(
        G,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
        ax=ax,
        **kwargs
    )

    # Display edge labels
    if draw_edge_labels:
        edge_labels = {
            (u, v): "{:.{}f}".format(data[edge_variable], precision)
            for u, v, data in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, edge_labels=edge_labels, ax=ax, **kwargs)


def plot_slider_graph(
    g,
    reverse=False,
    random_state=None,
    weight="weight",
    weight_shown="weight_plot",
    max_node_size=800,
    min_node_size=100,
):
    """
    Display an interactive graph with a slider to control the number of displayed edges.

    Parameters
    ----------
    g : networkx.Graph
        The graph to be displayed.
    reverse : bool, optional
        If True, edges are sorted from longest to shortest. Otherwise, they are sorted from shortest to longest.
    random_state : int or None, optional
        Random seed for node positioning. Defaults to None.
    weight : str, optional
        The edge attribute used for sorting the edges. Defaults to 'weight'.
    weight_shown : str, optional
        The edge attribute used for displaying edge labels. Defaults to 'weight_plot'.
    max_node_size : int, optional
        The maximum size of nodes in the plot. Defaults to 800.
    min_node_size : int, optional
        The minimum size of nodes in the plot. Defaults to 100.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider widget used to control the number of displayed edges.
    """
    graph = g.copy()
    graph.clear_edges()

    def get_colors_from_graph(G):
        """
        Retrieve the node and edge colors from the graph. If no colors are set, defaults are used.

        Parameters
        ----------
        G : networkx.Graph
            The graph for which node and edge colors are retrieved.

        Returns
        -------
        tuple of list of str
            List of node colors and edge colors.
        """
        try:
            node_colors = [data["color"] for _, data in G.nodes(data=True)]
        except KeyError:
            node_colors = "#1f78b4"

        try:
            edge_colors = [data["color"] for _, _, data in G.edges(data=True)]
        except KeyError:
            edge_colors = "k"

        return node_colors, edge_colors

    def get_size_nodes_from_graph(G, max_size=max_node_size, min_size=min_node_size):
        """
        Calculate the sizes of nodes based on their 'size_plot' attribute.

        Parameters
        ----------
        G : networkx.Graph
            The graph for which node sizes are calculated.
        max_size : int, optional
            The maximum size of nodes. Defaults to 800.
        min_size : int, optional
            The minimum size of nodes. Defaults to 100.

        Returns
        -------
        list of int
            List of node sizes.
        """
        return [
            data["size_plot"] * max_size + min_size for _, data in G.nodes(data=True)
        ]

    node_sizes = get_size_nodes_from_graph(graph)

    def update_graph(
        num_edges,
        g=g,
        reverse=reverse,
        random_state=random_state,
        weight=weight,
        weight_shown=weight_shown,
        node_sizes=node_sizes,
    ):
        """
        Update the displayed graph when the slider is moved. Controls the number of displayed edges.

        Parameters
        ----------
        num_edges : int
            The number of edges to display, based on the slider value.
        g : networkx.Graph
            The original graph.
        reverse : bool
            Whether to reverse the edge sorting order.
        random_state : int or None
            The random state for node layout.
        weight : str
            The edge attribute used for sorting.
        weight_shown : str
            The edge attribute used for displaying edge labels.
        node_sizes : list of int
            List of node sizes.

        """
        ax.clear()
        G = g.copy()

        if num_edges > 0:
            edges = sorted(
                G.edges(data=True), key=lambda x: x[2].get(weight, 0), reverse=reverse
            )[:num_edges]
            G.clear_edges()
            G.add_edges_from(edges)

        node_colors, edge_colors = get_colors_from_graph(G)

        pos = nx.spring_layout(G, weight=weight, seed=random_state)
        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
        )

        edge_labels = {
            (u, v): "{:.2f}".format(data[weight_shown])
            for u, v, data in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="red", ax=ax
        )

        fig.canvas.draw()

    node_colors, edge_colors = get_colors_from_graph(graph)

    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph, weight=weight, seed=random_state)
    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
    )

    edge_labels = {
        (u, v): "{:.2f}".format(data[weight_shown])
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_color="red", ax=ax
    )

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(
        ax=ax_slider,
        label="Number of edges",
        facecolor="lightgoldenrodyellow",
        valmin=0,
        valmax=len(list(g.edges)),
        valinit=0,
        valstep=1,
    )
    slider.on_changed(update_graph)

    return slider
