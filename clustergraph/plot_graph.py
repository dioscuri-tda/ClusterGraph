import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import interact, IntSlider
from IPython.display import display, HTML


def draw_graph_pie(
    graph,
    nb_edges=None,
    edge_variable="weight_plot",
    scale_nodes=True,
    size_nodes=0.05,
    random_state=42,
    ax=None,
    **kwargs
):

    if ax == None:
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

    # Afficher les labels des arêtes
    edge_labels = {
        (u, v): "{:.2f}".format(data[edge_variable])
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="black",
        font_size=9,
    )


def draw_graph(
    graph,
    nb_edges=None,
    edge_variable="weight_plot",
    scale_nodes=True,
    size_nodes=1000,
    random_state=42,
    precision=2,
    ax=None,
    **kwargs
):
    """_summary_
    Function which plots a graph with the asked number of edges sorted from shortest to longest or the opposite. The edges and nodes can be colored.
    ----------
    graph :  networkx graph
         The graph to be displayed.
    nb_edges :  int
         The number of edges which will be displayed in the visualization. The edges are sorted hence the shortest will be shown.
    size_nodes :  int
         Baseline for the node's size on the visualization. Bigger the number, bigger the nodes.
    random_state :  int
         The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.

    """

    if ax == None:
        ax = plt.gca()

    G = graph.copy()

    if nb_edges is not None:
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0))[
            :nb_edges
        ]
        G.clear_edges()
        G.add_edges_from(edges)

    # Obtenir les couleurs des nœuds et des arêtes
    node_colors = [data["color"] for _, data in G.nodes(data=True)]
    edge_colors = [data["color"] for _, _, data in G.edges(data=True)]

    # Obtenir la taille des nœuds (multipliée par 100 pour une meilleure visualisation)
    if scale_nodes:
        node_sizes = [data["size_plot"] * size_nodes for _, data in G.nodes(data=True)]
    else:
        node_sizes = [size_nodes for _ in G.nodes()]

    # Créer le dessin du graphique
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

    # Afficher les labels des arêtes
    if edge_variable is not None:
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
    """_summary_
    Method which plots into an interactive matplotlib window the graph g with a slider in order to choose the number of edges.
    ----------
    g :  networkx graph
         The graph which is displayed.
    reverse :  bool
         If reverse is True, the edges will be dispalyed from longest to shortest. Otherwise it will be from shortest to longest.
    random_state :  int
         The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.
    weight :  string
         Label underwhich the weight of the edges is stored. This weight is used to sort the edges.
    weight_shown :  string
          Label which will be displayed on the plot. Can be the normalized value of each edge.
    max_node_size :  int
         The maximum size of a node of the visualized graph.
    min_node_size :  int
         The minimum size of a node of the visualized graph.

    Returns
    -------
     Slider
         The slider which is displayed.
    """

    graph = g.copy()
    graph.clear_edges()

    def get_colors_from_graph(G):
        """_summary_
        Function which returns the labels for the nodes and edges of a given graph.
        Parameters
        ----------
        G :  networkx graph
             Corresponds to the Graph for which, the colors of nodes and edges are demanded.

        Returns
        -------
         list , list
             Returns the lists of colors for the nodes and for the edges of the graph G
        """
        # We try to get the colors with the given labels, if it fails we use a default value
        try:
            node_colors = [data["color"] for _, data in G.nodes(data=True)]
        except:
            node_colors = "#1f78b4"

        try:
            edge_colors = [data["color"] for _, _, data in G.edges(data=True)]

        except:
            edge_colors = "k"

        return node_colors, edge_colors

    def get_size_nodes_from_graph(G, max_size=max_node_size, min_size=min_node_size):
        """_summary_
        Function which returns the list of the size of nodes. Those sizes correspond to the size of each node in the visualization.
        ----------
        G :  networkx graph
             Corresponds to the Graph for which, the size of nodes is demanded.
        max_size :  int
             Corresponds to the maximum size of a node in the visualization.
        min_size :  int
             Corresponds to the minimum size of a node in the visualization.

        Returns
        -------
         list
             Returns the list of the size of the nodes for the visualization.
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
        """_summary_
        Function which will be called when the value of the slider changes. This function changes the number of edges displayed in the visualized graph.
        ----------
        num_edges :  int
             Number of edges to display. It is the value of the slider.
        g :  networkx graph
             The graph with the maximum number of edges which can be plotted. The baseline graph.
        reverse :  bool
             If reverse is True, the edges will be dispalyed from longest to shortest. Otherwise it will be from shortest to longest.
        random_state :  int
             The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.
        weight :  string
             Label underwhich the weight of the edges is stored. This weight is used to sort the edges.
        weight_shown :  string
              Label which will be displayed on the plot. Can be the normalized value of each edge.
        node_sizes :  list
             List of the size of the nodes in the visualization.
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
