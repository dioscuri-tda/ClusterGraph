import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import interact, IntSlider
from IPython.display import display, HTML


def plot_colored_graph(graph, num_edges=None, variable="weight_plot", size_nodes = 1000 ):
    #plt.close('all')
    G = graph.copy()

    if num_edges is not None:
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get("labels", 0))[:num_edges]
        G.clear_edges()
        G.add_edges_from(edges)

    # Obtenir les couleurs des nœuds et des arêtes
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

    # Obtenir la taille des nœuds (multipliée par 100 pour une meilleure visualisation)
    node_sizes = [data['size_plot']*size_nodes for _, data in G.nodes(data=True)]
    plt.figure(figsize=(10, 8))
    # Créer le dessin du graphique
    pos = nx.spring_layout(G, seed=42) #, weight='label')  # You can use a different layout if needed
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
    plt.show()
    
    

    
def display_graph_with_slider(graph, size_nodes = 1000 ):
    @interact(num_edges=IntSlider(min=0, max=len(graph.edges()), step=1, value=0))
    def interact_display(num_edges):
        plot_colored_graph(graph, num_edges=num_edges , size_nodes = size_nodes)
    
    display(interact_display)