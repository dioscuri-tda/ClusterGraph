import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import interact, IntSlider
from IPython.display import display, HTML


def plot_colored_graph(graph, nb_edges=None, variable="weight_plot", size_nodes = 1000, random_state = None ):

    """_summary_
    Function which plots a graph with the asked number of edges sorted from shortest to longest or the opposite. The edges and nodes can be colored.
    ----------
    g : _type_ networkx graph
        _description_ The graph to be displayed.
    nb_edges : _type_ int 
        _description_ The number of edges which will be displayed in the visualization. The edges are sorted hence the shortest or the longest will be shown.
    size_nodes : _type_ int 
        _description_ Baseline for the node's size on the visualization. Bigger the number, bigger the nodes.
    random_state : _type_ int 
        _description_ The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.

"""
    
    G = graph.copy()

    if nb_edges is not None:
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get("label", 0))[:nb_edges]
        G.clear_edges()
        G.add_edges_from(edges)

    # Obtenir les couleurs des nœuds et des arêtes
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

    # Obtenir la taille des nœuds (multipliée par 100 pour une meilleure visualisation)
    node_sizes = [data['size_plot']*size_nodes for _, data in G.nodes(data=True)]
    plt.figure(figsize=(8, 6))
    # Créer le dessin du graphique
    pos = nx.spring_layout(G, seed= random_state ) 
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
    )
   
    # Afficher les labels des arêtes
    edge_labels = {(u, v): "{:.2f}".format(data[variable]) for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    #plt.show()
    
    




def plot_slider_graph( g , reverse = False, random_state = None, weight = 'label', 
                      weight_shown = "weight_plot" , max_node_size = 800,
                       min_node_size = 100  ):

    """_summary_
    Method which plots into an interactive matplotlib window the graph g with a slider in order to choose the number of edges.
    ----------
    g : _type_ networkx graph
        _description_ The graph which is displayed.
    reverse : _type_ bool 
        _description_ If reverse is True, the edges will be dispalyed from longest to shortest. Otherwise it will be from shortest to longest.
    random_state : _type_ int
        _description_ The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.
    weight : _type_ string
        _description_ Label underwhich the weight of the edges is stored. This weight is used to sort the edges.
    weight_shown : _type_ string
        _description_  Label which will be displayed on the plot. Can be the normalized value of each edge. 
    max_node_size : _type_ int
        _description_ The maximum size of a node of the visualized graph.
    min_node_size : _type_ int
        _description_ The minimum size of a node of the visualized graph.

    Returns
    -------
    _type_ Slider
        _description_ The slider which is displayed.
    """
    
    
    graph = g.copy()
    graph.clear_edges()
    
    def get_colors_from_graph( G ) :
        """_summary_
    Function which returns the labels for the nodes and edges of a given graph.
    Parameters
    ----------
    G : _type_ networkx graph
        _description_ Corresponds to the Graph for which, the colors of nodes and edges are demanded.

    Returns
    -------
    _type_ list , list
        _description_ Returns the lists of colors for the nodes and for the edges of the graph G
    """
        # We try to get the colors with the given labels, if it fails we use a default value  
        try :    
            node_colors = [data['color'] for _, data in G.nodes(data=True)]
        except :
            node_colors = "#1f78b4"
            
        try :
            edge_colors = [data['color'] for _, _, data in G.edges(data=True)]
            
        except :
            edge_colors = 'k'
        
        return node_colors, edge_colors
    
    def get_size_nodes_from_graph(G, max_size = max_node_size , min_size = min_node_size  ) :
        """_summary_
    Function which returns the list of the size of nodes. Those sizes correspond to the size of each node in the visualization.
    ----------
    G : _type_ networkx graph
        _description_ Corresponds to the Graph for which, the size of nodes is demanded.
    max_size : _type_ int
        _description_ Corresponds to the maximum size of a node in the visualization.
    min_size : _type_ int
        _description_ Corresponds to the minimum size of a node in the visualization.

    Returns
    -------
    _type_ list
        _description_ Returns the list of the size of the nodes for the visualization.
    """
        return [  data['size_plot']*max_size  + min_size for _, data in G.nodes(data=True)]
    
    node_sizes = get_size_nodes_from_graph(graph)
    

    def update_graph(num_edges, g= g , reverse = reverse , random_state =  random_state, 
                     weight = weight, weight_shown = weight_shown  ,  node_sizes = node_sizes ):
        
        """_summary_
    Function which will be called when the value of the slider changes. This function changes the number of edges displayed in the visualized graph.
    ----------
    num_edges : _type_ int 
        _description_ Number of edges to display. It is the value of the slider.
    g : _type_ networkx graph
        _description_ The graph with the maximum number of edges which can be plotted. The baseline graph.
    reverse : _type_ bool 
        _description_ If reverse is True, the edges will be dispalyed from longest to shortest. Otherwise it will be from shortest to longest.
    random_state : _type_ int
        _description_ The random state which will be used to plot the graph. If the value is None, the position of the graph will change each time.
    weight : _type_ string
        _description_ Label underwhich the weight of the edges is stored. This weight is used to sort the edges.
    weight_shown : _type_ string
        _description_  Label which will be displayed on the plot. Can be the normalized value of each edge. 
    node_sizes : _type_ list
        _description_ List of the size of the nodes in the visualization.
    """

        ax.clear()
        G = g.copy()

        if num_edges > 0:
            edges = sorted(G.edges(data=True), key=lambda x: x[2].get( weight , 0) , reverse = reverse )[:num_edges]
            G.clear_edges()
            G.add_edges_from(edges)
            
        node_colors, edge_colors = get_colors_from_graph( G )
        
        pos = nx.spring_layout(G , weight= weight ,seed = random_state)
        nx.draw_networkx(
            G,
            pos, ax =ax ,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
        )
        
        edge_labels = {(u, v): "{:.2f}".format(data[weight_shown]) for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax = ax)
        
        fig.canvas.draw()
        

    node_colors, edge_colors = get_colors_from_graph( graph )
    
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph , weight= weight ,seed = random_state)
    nx.draw_networkx(
            graph, pos, ax =ax, with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
        )
        
    edge_labels = {(u, v): "{:.2f}".format(data[weight_shown]) for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', ax = ax)
        
    
    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(ax=ax_slider, label='Number of edges', facecolor='lightgoldenrodyellow', valmin=0, valmax=len(list(g.edges)), valinit=0, valstep=1)
    slider.on_changed(update_graph)
    
    return slider 