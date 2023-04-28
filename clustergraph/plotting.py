import matplotlib.pyplot as plt
from bokeh.io import show, save, output_file
from bokeh.plotting import figure
from bokeh.layouts import layout, column, row, grid
from bokeh.models import (
    BoxZoomTool,
    Circle,
    HoverTool,
    MultiLine,
    Plot,
    Range1d,
    ResetTool,
    BoxSelectTool,
    ColumnDataSource,
    LabelSet,
    TapTool,
    WheelZoomTool,
    PanTool,
    ColorBar,
    LinearColorMapper,
    BasicTicker,
    ContinuousTicker,
    Button,
    TextInput,
    CustomJS,
    MultiChoice,
    SaveTool,
    FixedTicker,
    Slider,
)

from bokeh.transform import transform
from bokeh.models import CustomJSTransform
from bokeh.events import Tap, SelectionGeometry
from bokeh.plotting import from_networkx
from matplotlib import cm
from matplotlib.colors import to_hex
from .utils import get_sorted_edges, max_size_node_graph, insert_sorted_list
import networkx as nx
import numpy as np
import pandas as pd


def reshape_vertices(graph, variable=None, max_size=600):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    variable : _type_, optional
        _description_, by default None
    max_size : int, optional
        _description_, by default 600

    Returns
    -------
    _type_
        _description_
    """
    size_vertices = []
    # IF NO VARIABLE THEN ALL THE SAME SIZE
    nodes = list(graph.nodes)
    try:
        test = graph.nodes[nodes[0]][variable]

    except KeyError:
        variable = None

    if not (variable):
        size_vertices = [max_size / 2] * len(nodes)

    else:
        maxi_size_vertices = max_size_node_graph(graph, variable, nodes=None)
        for node in nodes:
            size = round(
                (len(graph.nodes[node][variable]) / maxi_size_vertices) * max_size, 3
            )
            size_vertices.append(size)

    return size_vertices

   

def draw_distances_graph(
    graph,
    edges=None,
    vertices=None,
    variable_points="points_covered",
    variable_edges="label",
    nb_decimal=2,
):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    edges : _type_, optional
        _description_, by default None
    vertices : _type_, optional
        _description_, by default None
    variable_points : str, optional
        _description_, by default "points_covered"
    variable_edges : str, optional
        _description_, by default "label"
    nb_decimal : int, optional
        _description_, by default 2

    Returns
    -------
    _type_
        _description_
    """
    global temp_graph
    global temp_edges_labelled
    global temp_vertices
    global temp_size_vertices
    global temp_variable_edges
    global nb_decimal_temp
    plt.close("all")
    temp_graph = True
    temp_edges_labelled = []
    temp_vertices = []
    temp_size_vertices = []
    temp_variable_edges = variable_edges

    # UPDATE GRAPH BY CHANGING SLIDER VALUE
    def slider_update_graph(value):
        """_summary_

        Parameters
        ----------
        value : _type_
            _description_
        """
        global temp_graph
        global temp_edges_labelled
        global temp_vertices
        global temp_size_vertices
        global nb_decimal_temp  #
        temp_graph.clear()
        ax.clear()
        for vertice in temp_vertices:
            temp_graph.add_node(vertice)

        for i in range(value):
            length = round(temp_edges_labelled[i][2], nb_decimal_temp)  #
            temp_graph.add_edge(
                temp_edges_labelled[i][0], temp_edges_labelled[i][1], label=length
            )  #

        # Draw the graph
        # pos=nx.spring_layout(temp_graph ,  weight= temp_variable_edges )
        pos = nx.spring_layout(temp_graph)
        # pos = nx.kamada_kawai_layout( temp_graph  , weight= temp_variable_edges)
        labels = nx.get_edge_attributes(temp_graph, temp_variable_edges)
        nx.draw_networkx(
            temp_graph,
            pos=pos,
            node_size=temp_size_vertices,
            with_labels=True,
            font_weight="bold",
            ax=ax,
        )
        nx.draw_networkx_edge_labels(temp_graph, pos=pos, edge_labels=labels, ax=ax)
        fig.canvas.draw_idle()
        pass

    # END UPDATE GRAPH

    # CREATION OF THE GRAPH DISPLAYED
    temp_graph = nx.Graph()
    temp_graph.clear()

    maxi_size_vertices = None
    # Creation temp VERTICES
    if not (vertices):
        temp_vertices = list(graph.nodes)
    else:
        temp_vertices = vertices.sort()

    # ADDING THE WANTED VERTICES TO THE GRAPH
    for node in temp_vertices:
        node_att = graph.nodes(data=True)[node]
        temp_graph.add_node(node, **node_att)

    temp_size_vertices = reshape_vertices(temp_graph, variable_points)
    nb_decimal_temp = nb_decimal

    # EDGES we add only the edges corresponding to the nodes ?
    if not (edges):
        edges = list(graph.edges(data=True))
        temp_edges_labelled = []

        # look for corresponding edges
        for edge in edges:
            if edge[0] in temp_vertices and edge[1] in temp_vertices:
                temp_edges_labelled = insert_sorted_list(
                    temp_edges_labelled, [edge[0], edge[1], edge[2][variable_edges]]
                )

    else:
        temp_edges_labelled = []
        for edge in edges:
            if edge[0] in temp_vertices and edge[1] in temp_vertices:
                temp_edges_labelled = insert_sorted_list(
                    temp_edges_labelled, [edge[0], edge[1], edge[2]]
                )

    plt.close("all")
    fig, ax = plt.subplots()
    pos = nx.spring_layout(temp_graph)
    labels = nx.get_edge_attributes(temp_graph, "label")
    nx.draw_networkx(
        temp_graph,
        pos,
        node_size=temp_size_vertices,
        with_labels=True,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(temp_graph, pos, edge_labels=labels)

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(
        ax=ax_slider,
        label="Number of edges",
        facecolor="lightgoldenrodyellow",
        valmin=0,
        valmax=len(temp_edges_labelled),
        valinit=0,
        valstep=1,
    )
    slider.on_changed(slider_update_graph)
    plt.show()

    return slider



def add_colorbar(
    plot, palette, variable="variable", num_ticks=100, low=None, high=None
):
    """_summary_

    Parameters
    ----------
    plot : _type_
        _description_
    palette : _type_
        _description_
    variable : str, optional
        _description_, by default "variable"
    num_ticks : int, optional
        _description_, by default 100
    low : _type_, optional
        _description_, by default None
    high : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    ticks = [i for i in np.linspace(low, high, 5)]

    color_ticks = FixedTicker(ticks=ticks)

    color_mapper = LinearColorMapper(
        palette=[
            to_hex(palette(color_id)) for color_id in np.linspace(0, 1, num_ticks)
        ],
        low=low,
        high=high,
    )

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=color_ticks,
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title_text_font_size="30px",
        major_label_text_font_size="30px",  # major_label_text_font_size = '11px' ,
        title=variable,
    )

    plot.add_layout(color_bar, "right")
    return plot





def graph_from_2D(X_cloud, size_rescaled=1):
    """_summary_

    Parameters
    ----------
    X_cloud : _type_
        _description_
    size_rescaled : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    # Coordinate max and min on each axis fro display correctly the point (zoom inaff) in the plot
    min_x = X_cloud[0][0]
    max_x = X_cloud[0][0]
    min_y = X_cloud[0][1]
    max_y = X_cloud[0][1]
    graph = nx.Graph()

    for i in range(len(X_cloud)):
        graph.add_node(
            i,
            points_covered=[i],
            size=1,
            pos=(X_cloud[i][0], X_cloud[i][1]),
            size_rescaled=size_rescaled,
        )

        if min_x > X_cloud[i][0]:
            min_x = X_cloud[i][0]

        if max_x < X_cloud[i][0]:
            max_x = X_cloud[i][0]

        if min_y > X_cloud[i][1]:
            min_y = X_cloud[i][1]

        if max_y < X_cloud[i][1]:
            max_y = X_cloud[i][1]

    return graph, (min_x, max_x), (min_y, max_y)


def nodes_covered_by_label(graph, labels):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    labels : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    dict_labels = {}
    nodes = list(graph.nodes)
    if isinstance(labels, dict):
        unique_labels = [labels[key] for key in list(labels)]
        unique_labels = np.unique(unique_labels)
        for label in unique_labels:
            dict_labels[int(label)] = []

        for node in nodes:
            dict_labels[int(labels[node])].append(node)

    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            dict_labels[int(label)] = []

        if len(labels) == len(nodes):
            for i in range(len(nodes)):
                dict_labels[int(labels[i])].append(int(nodes[i]))

        else:
            for node in nodes:
                points = graph.nodes[node]["points_covered"]
                labels_in_node = np.unique(labels[points])

                for label in labels_in_node:
                    dict_labels[int(label)].append(int(node))

    return dict_labels, list(unique_labels)



def color_nodes_labels(bokeh_graph, labels, palette):
    """_summary_

    Parameters
    ----------
    bokeh_graph : _type_
        _description_
    labels : _type_
        _description_
    palette : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    nb_labels = len(labels)
    nodes = list(bokeh_graph.nodes)
    nb_nodes = len(nodes)

    if isinstance(labels, dict):
        list_nodes = list(labels)
        all_labels = np.array([])
        for node in list_nodes:
            all_labels = np.append(all_labels, int(labels[node]))

        unique_labels = np.unique(all_labels)
        nb_label = len(unique_labels)
        color_list = [to_hex(palette(i / nb_label)) for i in range(nb_label + 1)]

        for node in nodes:
            bokeh_graph.nodes[node]["color"] = color_list[int(labels[node])]

    else:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        nb_label = len(unique_labels)
        color_list = [to_hex(palette(i / nb_label)) for i in range(nb_label + 1)]

        #  for each nodes we take a look at which label is dominating
        for node in nodes:
            # we count how many times each labels is in the node, the maximum will get the color
            points = bokeh_graph.nodes[node]["points_covered"]
            label_in_node, nb_each = np.unique(labels[points], return_counts=True)

            # FIND THE LABEL APPEARING THE MAXIMUM TIMES AND GIVE THE COLOR
            index_max = np.argmax(nb_each)
            label = label_in_node[index_max]
            index_label = np.where(unique_labels == label)[0][0]
            # print("INDEX",index_label)
            bokeh_graph.nodes[node]["color"] = color_list[int(index_label)]

            # ADD THE POSSIBILITY TO SEE PERCENTAGE OF EACH LABEL

            per_label = ""
            for i in range(len(label_in_node)):
                per_label = (
                    per_label
                    + "label "
                    + str(label_in_node[i])
                    + " : "
                    + str(
                        round(
                            nb_each[i] / len(bokeh_graph.nodes[node]["points_covered"]),
                            3,
                        )
                    )
                    + ", "
                )

            if per_label != "":
                bokeh_graph.nodes[node]["perc_labels"] = per_label


            # per_label = 42
            bokeh_graph.nodes[node]["perc_labels"] = per_label

        # print(bokeh_graph.nodes(data=True))
        nb_labels = nb_label

    return bokeh_graph, nb_labels



def color_nodes_variable(
    X,
    bokeh_graph,
    palette,
    color_variable,
    MIN_VALUE=10000,
    MAX_VALUE=-10000,
    logscale=False,
):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    bokeh_graph : _type_
        _description_
    palette : _type_
        _description_
    color_variable : _type_
        _description_
    MIN_VALUE : int, optional
        _description_, by default 10000
    MAX_VALUE : int, optional
        _description_, by default -10000
    logscale : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    for node in bokeh_graph.nodes:
        # ASSIGN AT EACH NODE ITS AVERAGE FOR THE GIVEN VARIABLE
        bokeh_graph.nodes[node][color_variable] = X[
            bokeh_graph.nodes[node]["points_covered"]
        ].mean()

        # TEST IF IT'S MAX OR MIN TO GET THE RANGE
        if bokeh_graph.nodes[node][color_variable] > MAX_VALUE:
            MAX_VALUE = bokeh_graph.nodes[node][color_variable]
        if bokeh_graph.nodes[node][color_variable] < MIN_VALUE:
            MIN_VALUE = bokeh_graph.nodes[node][color_variable]

    # FOR EACH NODE GET THE COLOR THAT WILL BE USED
    for node in bokeh_graph.nodes:
        if not pd.isna(bokeh_graph.nodes[node][color_variable]):
            color_id = (bokeh_graph.nodes[node][color_variable] - MIN_VALUE) / (
                MAX_VALUE - MIN_VALUE
            )
            if logscale:
                color_id = (
                    np.log10(bokeh_graph.nodes[node][color_variable])
                    - np.log10(MIN_VALUE)
                ) / (np.log10(MAX_VALUE) - np.log10(MIN_VALUE))

            bokeh_graph.nodes[node]["color"] = to_hex(palette(color_id))
        else:
            bokeh_graph.nodes[node]["color"] = "black"

    return bokeh_graph, MIN_VALUE, MAX_VALUE




def add_covered_points_from_file(graph, title, delimiter=","):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    title : _type_
        _description_
    delimiter : str, optional
        _description_, by default ","

    Returns
    -------
    _type_
        _description_
    """
    dico = {}
    MAX_NODE_SIZE = 0
    with open(title) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            temp = list(map(int, row))
            graph.add_node(temp[0], size=len(temp[1:]), points_covered=temp[1:])
            if len(temp[1:]) > MAX_NODE_SIZE:
                MAX_NODE_SIZE = len(temp[1:])

    return graph, MAX_NODE_SIZE




def add_edges_from_file(graph, title, nb_edges=-1, delimiter=","):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    title : _type_
        _description_
    nb_edges : int, optional
        _description_, by default -1
    delimiter : str, optional
        _description_, by default ","

    Returns
    -------
    _type_
        _description_
    """
    with open(title) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        count = 1
        for row in csv_reader:
            graph.add_edge(int(row[0]), int(row[1]), label=float(row[2]))
            if count == nb_edges:
                break
            count += 1
        return graph




def get_bokeh_graph_from_info(
    graph,
    my_palette=cm.get_cmap(name="Reds"),
    nb_edges=-1,
    edges=None,
    nb_decimal=2,
    choice_col_type=None,
    X_col=None,
    name_var_col=None,
    labels_col=None,
    MIN_SCALE=7,
    MAX_SCALE=20,
):
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    my_palette : _type_, optional
        _description_, by default cm.get_cmap(name="Reds")
    nb_edges : int, optional
        _description_, by default -1
    edges : _type_, optional
        _description_, by default None
    nb_decimal : int, optional
        _description_, by default 2
    choice_col_type : _type_, optional
        _description_, by default None
    X_col : _type_, optional
        _description_, by default None
    name_var_col : _type_, optional
        _description_, by default None
    labels_col : _type_, optional
        _description_, by default None
    MIN_SCALE : int, optional
        _description_, by default 7
    MAX_SCALE : int, optional
        _description_, by default 20

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    #######################################
    # GET THE GRAPH IS NOT DIRECTLY GIVEN #

    # If not networkx graph, we try if this is a dictionary to get it from files
    if not (isinstance(graph, nx.classes.graph.Graph)):
        try:
            g = nx.Graph()
            g, MAX_NODE_SIZE = add_covered_points_from_file(g, graph["title_vertices"])
            g = add_edges_from_file(g, graph["title_edges"], nb_edges=nb_edges)
            graph = g

        except:
            raise TypeError(
                "Info graph must be a networkx graph or a dictionary with title_vertices and title_edges corresponding to the files to rebuild the graph"
            )

            ##########################################################
            # MODIFICATION OF THE GIVEN GRAPH + GET SOME INFORMATION #

    MAX_NODE_SIZE = 0
    nodes = graph.nodes(data=True)

    # If we have no info on the nb of edges we copy all the graph
    if not (edges) and nb_edges == -1:  # not(edges) or
        bokeh_graph = graph
        for node in nodes:
            if node[1]["size"] > MAX_NODE_SIZE:
                MAX_NODE_SIZE = node[1]["size"]

    else:
        bokeh_graph = nx.Graph()
        for node in nodes:
            bokeh_graph.add_node(int(node[0]), **node[1])
            if node[1]["size"] > MAX_NODE_SIZE:
                MAX_NODE_SIZE = node[1]["size"]

        if not (edges):
            edges = get_sorted_edges(graph)

        if nb_edges < 0:
            nb_edges = len(edges)

        for i in range(nb_edges):
            bokeh_graph.add_edge(
                edges[i][0], edges[i][1], label=round(edges[i][2], nb_decimal)
            )

            #######################################
            # ADDING NEW ATTRIBUTES TO BOKEH GRAPH#

    nodes = list(bokeh_graph.nodes)
    # I) COLOR BY VARIABLE
    if choice_col_type == "by_var":
        bokeh_graph, MIN_CBAR, MAX_CBAR = color_nodes_variable(
            X_col, bokeh_graph, my_palette, name_var_col
        )

    # II) COLOR BY LABEL
    elif choice_col_type == "labels":
        # Bokeh colored by labels or most represented label
        bokeh_graph, nb_labels = color_nodes_labels(bokeh_graph, labels_col, my_palette)
        MIN_CBAR, MAX_CBAR = 0, nb_labels

    # III) NO DIFFERENCE OF COLOR AT THE BEGINNING
    else:
        color_list = [to_hex(my_palette(i / 100)) for i in range(101)]
        for node in nodes:
            bokeh_graph.nodes[node]["color"] = color_list[0]
        MIN_CBAR, MAX_CBAR = 0, 100

    # Size rescaled and coverage
    for node in nodes:
        # rescale the size for display
        bokeh_graph.nodes[node]["size_rescaled"] = (
            MAX_SCALE * bokeh_graph.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SCALE
        )
        bokeh_graph.nodes[node]["coverage"] = 0

    return bokeh_graph, MIN_CBAR, MAX_CBAR



def create_html_plot(
    graph,
    x_range=(-1.1, 1.1),
    y_range=(-1.1, 1.1),
    title="Unamed plot",
    pos=False,
    color_bar=False,
    MIN_CBAR=None,
    MAX_CBAR=None,
    palette=cm.get_cmap(name="Reds"),
    title_color_bar="Variable",
    font_size_edges_labels="40px",
    percentage_label=True,
):
    # MUST BE CHECKED, NOT GENERATED AUTOMATICALLY
    """_summary_

    Parameters
    ----------
    graph : _type_
        _description_
    x_range : _type_, optional
        _description_, by default (-1.1, 1.1)
    y_range : _type_, optional
        _description_, by default (-1.1, 1.1)
    color_bar : _type_, optional
        _description_, by default False
    MIN_CBAR : _type_, optional
        _description_, by default None
    MAX_CBAR : _type_, optional
        _description_, by default None
    my_palette : _type_, optional
        _description_, by default cm.get_cmap(name="Reds")
    title_color_bar : str, optional
        _description_, by default "Variable"
    font_size_edges_labels : str, optional
        _description_, by default "40px"
    percentage_label : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """

    plot = Plot(
        plot_width=800,
        plot_height=800,
        x_range=Range1d(x_range[0], x_range[1]),
        y_range=Range1d(y_range[0], y_range[1]),
        sizing_mode="stretch_both",
        title=title,
    )

    if percentage_label == True:
        # HERE WE CAN TEST DIFFERENT THINGS TO ADD ONLY THE ONES WE WANT
        node_hover_tool = HoverTool(
            tooltips=[
                ("index", "@index"),
                ("size", "@size"),
                ("percentage_labels", "@{perc_labels}"),
            ]
        )
    else:
        # HERE WE CAN TEST DIFFERENT THINGS TO ADD ONLY THE ONES WE WANT
        node_hover_tool = HoverTool(
            tooltips=[
                ("index", "@index"),
                ("size", "@size"),
                ("points_covered", "@{points_covered}"),
                ("perc_labels", "@{perc_labels}"),
            ]
        )

    plot.add_tools(
        PanTool(),
        node_hover_tool,
        BoxZoomTool(),
        WheelZoomTool(),
        ResetTool(),
        TapTool(),
        BoxSelectTool(),
        SaveTool(),
    )

    # PREPARATION TO DISPLAY THE GRAPH

    # If we want to control the position of the node then 'pos==True'
    if pos == True:
        # print("Graph nodes", graph.nodes(data=True))
        graph_renderer = from_networkx(
            graph,
            nx.get_node_attributes(graph, "pos"),
            seed=42,
            scale=1,
            center=(0, 0),
            k=10 / np.sqrt(len(graph.nodes)),
            iterations=2000,
        )

    else:
        graph_renderer = from_networkx(
            graph, nx.spring_layout, seed=42, scale=1, center=(0, 0), iterations=200
        )

    # get the coordinates of each node
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    # Create a dictionary with each node position and the label
    source = ColumnDataSource(
        {"x": x, "y": y, "node_id": [node for node in graph.nodes]}
    )

    labels = LabelSet(
        x="x",
        y="y",
        text="node_id",
        source=source,
        text_color="black",
        text_alpha=1,
        visible=False,
    )

    graph_renderer.node_renderer.data_source.data["color"] = [
        graph.nodes[n]["color"] for n in graph.nodes
    ]
    # NODES
    graph_renderer.node_renderer.glyph = Circle(
        size="size_rescaled", fill_color="color", fill_alpha=0.8
    )

    # SHAPE OF THE EDGES
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="black", line_alpha=1.6, line_width=2
    )  # line_alpha=0.8, line_width=1)

    # LABELS EDGES
    if list(graph.edges) != []:
        # add the labels to the edge renderer data source
        source_2 = graph_renderer.edge_renderer.data_source
        source_2.data["names"] = ["%.2f" % dist for dist in source_2.data["label"]]

        code_edge_label = """
        const result = new Float64Array(xs.length)
        const coords = provider.get_edge_coordinates(source)[%s]
        for (let i = 0; i < xs.length; i++) {
            result[i] = (coords[i][0] + coords[i][1])/2
        }
        return result
    """

        xcoord = CustomJSTransform(
            v_func=code_edge_label % "0",
            args=dict(provider=graph_renderer.layout_provider, source=source_2),
        )
        ycoord = CustomJSTransform(
            v_func=code_edge_label % "1",
            args=dict(provider=graph_renderer.layout_provider, source=source_2),
        )

        # Use the transforms to supply coords to a LabelSet
        labels_edges = LabelSet(
            x=transform("start", xcoord),
            y=transform("start", ycoord),
            text="names",
            text_font_size=font_size_edges_labels,  # text_font_size="12px",
            x_offset=5,
            y_offset=5,
            source=source_2,
            render_mode="canvas",
        )

        plot.add_layout(labels_edges)

    plot.renderers.append(graph_renderer)
    plot.renderers.append(labels)

    if color_bar == True:
        plot = add_colorbar(
            plot,
            palette,
            variable=title_color_bar,
            num_ticks=100,
            low=MIN_CBAR,
            high=MAX_CBAR,
        )

    return plot, graph_renderer

   


# CAN BE IMPROVED BY POSSIBILITY TO GET ONLY ONE PLOT WHEN WE SELECT LABELS TO BE COVERED
def show_save_with_graphs(
    plot_1,
    graph_1,
    gr_1,
    name_file=None,
    plot_2=None,
    graph_2=None,
    gr_2=None,
    choice_col_type=None,
    labels_col=None,
    labels_2=None,
    palette=cm.get_cmap(name="Reds"),
    color_non_selected_nodes="black",
):
    """_summary_

    Parameters
    ----------
    plot_1 : _type_
        _description_
    graph_1 : _type_
        _description_
    gr_1 : _type_
        _description_
    name_file : str, optional
        _description_, by default None
    plot_2 : _type_, optional
        _description_, by default None
    graph_2 : _type_, optional
        _description_, by default None
    gr_2 : _type_, optional
        _description_, by default None
    choice_col_type : _type_, optional
        _description_, by default None
    labels_col : _type_, optional
        _description_, by default None
    my_palette : _type_, optional
        _description_, by default cm.get_cmap(name="Reds")
    color_non_selected_nodes : str, optional
        _description_, by default "black"

    Returns
    -------
    _type_
        _description_
    """
    
    
    #####################
    # COLORING BY LABEL #

    # COLORING BY LABELS WE WILL GET A SLIDER
    nodes_1 = list(graph_1.nodes)
    min_node_1 = min(nodes_1)

    if choice_col_type == "labels":
        # Creation initial color dictionnaries
        if not (isinstance(labels_col, dict)):
            init_col_g1 = nx.get_node_attributes(graph_1, "color")
            unique_labels = np.unique(labels_col)
        else:
            init_col_g1 = labels_col
            unique_labels = [labels_col[key] for key in list(labels_col)]
            unique_labels = np.unique(unique_labels)

        # GOOD WHEN SAME LABELS GIVEN (same unique labels)
        unique_labels = [int(x) for x in unique_labels]
        nb_labels = int(len(unique_labels))
        # Preparation for the slider code
        color_list = [to_hex(palette(i / nb_labels)) for i in range(nb_labels + 1)]
        g1_label_node, list_labels_1 = nodes_covered_by_label(graph_1, labels_col)

        # If there is a second graph we initialize it
        if graph_2:
            nodes_2 = list(graph_2.nodes)
            min_node_2 = min(list(nodes_2))
            # Creation initial color dictionnaries
            if not (isinstance(labels_col, dict)):
                init_col_g2 = nx.get_node_attributes(graph_2, "color")
                # HERE ADD SOMETHING WHEN ADD POSSIBILITY TO HAVE TWO DIFFERENT SETS OF LABELS

            else:
                if labels_2 != None:
                    init_col_g2 = labels_2

                else:
                    init_col_g2 = labels_col

            g2_label_node, list_labels_2 = (
                nodes_covered_by_label(graph_2, labels_2)
                if (labels_2 != None)
                else nodes_covered_by_label(graph_2, labels_col)
            )

        # If only one graph we put all values corresponding to graph_2 to False
        else:
            gr_2 = False
            g2_label_node = False
            nodes_2 = False
            init_col_g2 = False
            min_node_2 = False
            list_labels_2 = False
            nodes_2 = False

        code_slider = """
            // GRAPH...source.data['color'] is just an array then indices must be from 0 then nodes[i] - smallest_node in graph
            if (cb_obj.value == -1){
                // graph_1 
                for ( let i =0 ; i < nodes_1.length; i++) {
                    gr_1.node_renderer.data_source.data['color'][nodes_1[i]-min_node_1] =  init_g1[nodes_1[i]] ;
                }
                
                if(nodes_2 != false){
                    // graph_2
                    for ( let i =0 ; i < nodes_2.length; i++) {
                        gr_2.node_renderer.data_source.data['color'][nodes_2[i]-min_node_2] =  init_g2[nodes_2[i]] ;
                    } 
                }
            }
            
            // browe all nodes, if the node is in the nodes covered by this label we give the right color, otherwise color 0
            else {
                // graph_1
                var nodes_label_1 = g1_label_node[ list_labels_1[cb_obj.value]   ] ;
                
                for (let i =0 ; i < nodes_1.length; i++) {
                    if( nodes_label_1.includes( nodes_1[i])) { 
                        gr_1.node_renderer.data_source.data['color'][nodes_1[i]-min_node_1] = color_list[cb_obj.value]; 
                        }
                  
                    else {
                        gr_1.node_renderer.data_source.data['color'][nodes_1[i]-min_node_1] = color_non_selected_nodes ;
                    }
                }
                
                if(nodes_2 != false){
                // graph_2
                    var nodes_label_2 = g2_label_node[list_labels_2[cb_obj.value]] ;
                    for ( let i =0 ; i < nodes_2.length; i++) {
                        if( nodes_label_2.includes( nodes_2[i])) {
                            gr_2.node_renderer.data_source.data['color'][nodes_2[i]-min_node_2] = color_list[cb_obj.value];
                        }

                        else {
                            gr_2.node_renderer.data_source.data['color'][nodes_2[i]-min_node_2] =color_non_selected_nodes ; 
                        }
                    }
                }
            }
            gr_1.node_renderer.data_source.change.emit();
            if(nodes_2 != false) gr_2.node_renderer.data_source.change.emit();
        """

        callback_slider = CustomJS(
            args=dict(
                gr_1=gr_1,
                g1_label_node=g1_label_node,
                nodes_1=nodes_1,
                init_g1=init_col_g1,
                min_node_1=min_node_1,
                list_labels_1=list_labels_1,
                gr_2=gr_2,
                g2_label_node=g2_label_node,
                nodes_2=nodes_2,
                init_g2=init_col_g2,
                min_node_2=min_node_2,
                list_labels_2=list_labels_2,
                color_non_selected_nodes=color_non_selected_nodes,
                color_list=color_list,
            ),
            code=code_slider,
        )
        slider = Slider(
            title="Label colored", start=-1, end=int(len(unique_labels)) - 1, value=-1
        )
        slider.js_on_change("value", callback_slider)

        # Composition of the html window
        layout = (
            grid([[plot_1, plot_2], [slider]])
            if (plot_2)
            else grid([[plot_1], [slider]])
        )

        ##############################
        # COLORING BY POINTS COVERED #

    # If by points_covered
    elif choice_col_type != "by_var":
        color_list = [to_hex(palette(i / 100)) for i in range(101)]

        ################
        # color button #
        ################
        color_button = Button(label="COLOR", height_policy="fit", button_type="success")

        if not (plot_2):
            gr_2 = False
        # This code is called when the color_button object is clicked
        color_button_code = """ 
                if(gr_2== false) {
                    var node_data_1 = gr_1.node_renderer.data_source.data;
                    var selected_nodes = multi_choice.value.map(Number);

                    // color selected nodes of 1 to red, not selected to white
                    for (var i = 0; i < node_data_1['index'].length; i++) {
                        // node indices starts from 1

                        if (selected_nodes.includes(i+1)) {
                           gr_1.node_renderer.data_source.data['color'][i] = color_list[color_list.length - 1];
                        } 
                        else {
                           gr_1.node_renderer.data_source.data['color'][i] = color_list[0];
                        }
                    }      
                }
                else {
                    var node_data_1 = gr_1.node_renderer.data_source.data;
                    var node_data_2 = gr_2.node_renderer.data_source.data;

                    var selected_nodes = multi_choice.value.map(Number);

                    // color selected nodes of 1 to red, not selected to white
                    for (var i = 0; i < node_data_1['index'].length; i++) {
                        // node indices starts from 1

                        if (selected_nodes.includes(i+1)) {
                           gr_1.node_renderer.data_source.data['color'][i] = color_list[color_list.length - 1];
                        } 
                        else {
                           gr_1.node_renderer.data_source.data['color'][i] = color_list[0];
                        }
                    }      

                    // get list of points in selected_nodes
                    var points_in_selected_nodes = new Array();
                    for (var i = 0; i < selected_nodes.length; i++) {
                        points_in_selected_nodes.push(...gr_1.node_renderer.data_source.data['points_covered'][selected_nodes[i]-1]);
                    }

                    // select only unique values, convert it to a set
                    points_in_selected_nodes = new Set(points_in_selected_nodes);


                    // color nodes in graph_2 according to the percentage of points that are in POINTS_IN_SELECTED_NODES
                    for (var i = 0; i < node_data_2['index'].length; i++) {

                        const points_2 = new Set(node_data_2['points_covered'][i]);

                        const intersection = new Set([...points_in_selected_nodes].filter(value=>points_2.has(value)));

                        var coverage = intersection.size / points_2.size ;

                        //gr_2.node_renderer.data_source.data['coverage'][i] = coverage;
                        gr_2.node_renderer.data_source.data['color'][i] = color_list[Math.round(coverage*100)];
                    }      
                    
                    gr_2.node_renderer.data_source.change.emit();
                }
                gr_1.node_renderer.data_source.change.emit();
            """

        ###################
        # multichoice box #
        ###################
        OPTIONS = [str(n) for n in graph_1.nodes]
        # possibility to choose different value in a list
        multi_choice = MultiChoice(
            value=[], options=OPTIONS, max_height=80, height_policy="fit"
        )

        # WHAT WILL HAPPEN AND VARIABLE GIVEN WHEN WE CLICK ON THE COLOR BUTTON
        color_button.js_on_click(
            CustomJS(
                args=dict(
                    gr_1=gr_1,
                    gr_2=gr_2,
                    multi_choice=multi_choice,
                    color_list=color_list,
                ),
                code=color_button_code,
            )
        )

        # this code is called when nodes on the left are selected
        tap_code = """ 
                console.log(chi);
                var nodes_clicked_int = gr_1.node_renderer.data_source.selected.indices;
                for (var i = 0; i < nodes_clicked_int.length; i++){
                    nodes_clicked_int[i] += 1;
                }

                var nodes_clicked_str = new Array();
                for (var i = 0; i < nodes_clicked_int.length; i++){
                    nodes_clicked_str.push(nodes_clicked_int[i].toString());
                }
                multi_choice.value = [...new Set([...multi_choice.value, ...nodes_clicked_str])];
            """

        plot_1.js_on_event(
            SelectionGeometry,
            CustomJS(
                args=dict(gr_1=gr_1, multi_choice=multi_choice, chi="Selection"),
                code=tap_code,
            ),
        )

        # Composition of the html window
        layout = (
            grid([[color_button], [multi_choice], [plot_1, plot_2]])
            if (plot_2)
            else grid([[color_button], [multi_choice], [plot_1]])
        )

        ########################
        # COLORING BY VARIABLE #

    # Everything is fixed then just show the plot or plots
    else:
        if not (plot_2):
            layout = grid([[plot_1]])
        else:
            layout = grid([[plot_1, plot_2]])

    # If the file has no name then we don't save it
    if name_file != None:
        # SAVE AND OPEN
        OUTPUT_PATH = name_file
        save(layout, "{}.html".format(OUTPUT_PATH))

    show(layout)

   

def compare_graph_cloud(
    info_graph,
    X_cloud,
    nb_edges=-1,
    edges=None,
    title_left="Title left",
    title_right="Title right",
    name_file=None,
    palette=cm.get_cmap(name="Reds"),
    choice_col_type=None,
    X_col=None,
    name_var_col=None,
    labels_col=None,
    labels_2=None,
    color_non_selected_nodes="black",
    size_points=6,
    MIN_SCALE=7,
    MAX_SCALE=20,
    font_size_edges_labels="40px",
):
    """_summary_

    Parameters
    ----------
    info_graph : _type_
        _description_
    X_cloud : _type_
        _description_
    nb_edges : int, optional
        _description_, by default -1
    edges : _type_, optional
        _description_, by default None
    title_left : str, optional
        _description_, by default "Title left"
    title_right : str, optional
        _description_, by default "Title right"
    name_file : _type_, optional
        _description_, by default None
    palette : _type_, optional
        _description_, by default cm.get_cmap(name="Reds")
    choice_col_type : _type_, optional
        _description_, by default None
    X_col : _type_, optional
        _description_, by default None
    name_var_col : _type_, optional
        _description_, by default None
    labels_col : _type_, optional
        _description_, by default None
    labels_2 : _type_, optional
        _description_, by default None
    color_non_selected_nodes : str, optional
        _description_, by default "black"
    size_points : int, optional
        _description_, by default 6
    MIN_SCALE : int, optional
        _description_, by default 7
    MAX_SCALE : int, optional
        _description_, by default 20
    font_size_edges_labels : str, optional
        _description_, by default "40px"
    """
    # GRAPH CREATION
    graph_1, MIN_CBAR_1, MAX_CBAR_1 = get_bokeh_graph_from_info(
        info_graph,
        my_palette=palette,
        MIN_SCALE=MIN_SCALE,
        MAX_SCALE=MAX_SCALE,
        nb_edges=nb_edges,
        edges=edges,
        choice_col_type=choice_col_type,
        X_col=X_col,
        name_var_col=name_var_col,
        labels_col=labels_col,
    )

    # CREATION BOKEH GRAPH CREATION FOR POINT CLOUD
    graph_2, x_range, y_range = graph_from_2D(X_cloud)

    # if there is a second labels, it means it is an association
    if labels_2 != None:
        labels_col = labels_2
    # COLORING, SIZE... FOR THE GRAPH TO BE READY TO PLOT
    graph_2, MIN_CBAR_2, MAX_CBAR_2 = get_bokeh_graph_from_info(
        graph_2,
        my_palette=palette,
        MIN_SCALE=size_points,
        MAX_SCALE=0,
        choice_col_type=choice_col_type,
        X_col=X_col,
        name_var_col=name_var_col,
        labels_col=labels_col,
    )

    # CHOOSE TITLE COLOR BAR
    if name_var_col:
        title_color_bar = name_var_col
    elif choice_col_type == "labels":
        title_color_bar = "labels"
    else:
        title_color_bar = "percentage points covered"
    print("Xrange", x_range)
    print("Yrange", y_range)

    # CREATION OF THE TWO PLOTS AND GRAPH RENDERERS
    plot_1, gr_1 = create_html_plot(
        graph_1,
        title=title_left,
        pos=False,
        palette=palette,
        font_size_edges_labels=font_size_edges_labels,
    )

    # Creation of the ranges
    x_range = (
        x_range[0] + size_points / x_range[0],
        x_range[1] + size_points / x_range[1],
    )
    y_range = (
        y_range[0] + size_points / y_range[0],
        y_range[1] + size_points / y_range[1],
    )

    plot_2, gr_2 = create_html_plot(
        graph_2,
        x_range=x_range,
        y_range=y_range,
        title=title_right,
        pos=True,
        color_bar=True,
        MIN_CBAR=MIN_CBAR_1,
        MAX_CBAR=MAX_CBAR_1,
        palette=palette,
        title_color_bar=title_color_bar,
        font_size_edges_labels=font_size_edges_labels,
    )

    # Creation of the global file with the different global tool in the window and show/save
    show_save_with_graphs(
        plot_1,
        graph_1,
        gr_1,
        name_file=name_file,
        plot_2=plot_2,
        graph_2=graph_2,
        gr_2=gr_2,
        choice_col_type=choice_col_type,
        labels_col=labels_col,
        labels_2=labels_2,
        color_non_selected_nodes=color_non_selected_nodes,
        palette=palette,
    )

 


def show_graphs(
    info_graph_1,
    nb_edges_1=-1,
    edges_1=None,
    info_graph_2=None,
    nb_edges_2=-1,
    edges_2=None,
    title_left="Title left",
    title_right="Title right",
    name_file=None,
    palette=cm.get_cmap(name="Reds"),
    choice_col_type=None,
    X_col=None,
    name_var_col=None,
    labels_col=None,
    labels_2=None,
    color_non_selected_nodes="black",
    MIN_SCALE=7,
    MAX_SCALE=20,
    font_size_edges_labels="40px",
    percentage_label=True,
):
    """_summary_

    Parameters
    ----------
    info_graph_1 : _type_
        _description_
    nb_edges_1 : int, optional
        _description_, by default -1
    edges_1 : _type_, optional
        _description_, by default None
    info_graph_2 : _type_, optional
        _description_, by default None
    nb_edges_2 : int, optional
        _description_, by default -1
    edges_2 : _type_, optional
        _description_, by default None
    title_left : str, optional
        _description_, by default "Title left"
    title_right : str, optional
        _description_, by default "Title right"
    name_file : _type_, optional
        _description_, by default None
    palette : _type_, optional
        _description_, by default cm.get_cmap(name="Reds")
    choice_col_type : _type_, optional
        _description_, by default None
    X_col : _type_, optional
        _description_, by default None
    name_var_col : _type_, optional
        _description_, by default None
    labels_col : _type_, optional
        _description_, by default None
    labels_2 : _type_, optional
        _description_, by default None
    color_non_selected_nodes : str, optional
        _description_, by default "black"
    MIN_SCALE : int, optional
        _description_, by default 7
    MAX_SCALE : int, optional
        _description_, by default 20
    font_size_edges_labels : str, optional
        _description_, by default "40px"
    percentage_label : bool, optional
        _description_, by default True
    """
    # GRAPHS CREATION
    graph_1, MIN_CBAR_1, MAX_CBAR_1 = get_bokeh_graph_from_info(
        info_graph_1,
        my_palette=palette,
        MIN_SCALE=MIN_SCALE,
        MAX_SCALE=MAX_SCALE,
        nb_edges=nb_edges_1,
        edges=edges_1,
        choice_col_type=choice_col_type,
        X_col=X_col,
        name_var_col=name_var_col,
        labels_col=labels_col,
    )

    # CHOOSE TITLE COLOR BAR
    if name_var_col:
        title_color_bar = name_var_col
    elif choice_col_type == "labels":
        title_color_bar = "labels"
    else:
        title_color_bar = "percentage points covered"

    # Creation of the plot and the renderer
    plot_1, gr_1 = (
        create_html_plot(
            graph_1,
            title=title_left,
            palette=palette,
            percentage_label=percentage_label,
            font_size_edges_labels=font_size_edges_labels,
        )
        if (info_graph_2)
        else create_html_plot(
            graph_1,
            title=title_left,
            color_bar=True,
            MIN_CBAR=MIN_CBAR_1,
            MAX_CBAR=MAX_CBAR_1,
            palette=palette,
            title_color_bar=title_color_bar,
            percentage_label=percentage_label,
            font_size_edges_labels=font_size_edges_labels,
        )
    )

    # If there is a second graph we create it and the plot
    if info_graph_2 != None:
        # if there is a second labels, it means it is an association
        if labels_2 != None:
            labels_col = labels_2

        graph_2, MIN_CBAR_2, MAX_CBAR_2 = get_bokeh_graph_from_info(
            info_graph_2,
            my_palette=palette,
            MIN_SCALE=MIN_SCALE,
            MAX_SCALE=MAX_SCALE,
            nb_edges=nb_edges_2,
            edges=edges_2,
            choice_col_type=choice_col_type,
            X_col=X_col,
            name_var_col=name_var_col,
            labels_col=labels_col,
        )

        plot_2, gr_2 = create_html_plot(
            graph_2,
            title=title_right,
            color_bar=True,
            MIN_CBAR=MIN_CBAR_1,
            MAX_CBAR=MAX_CBAR_1,
            palette=palette,
            title_color_bar=title_color_bar,
            font_size_edges_labels=font_size_edges_labels,
            percentage_label=percentage_label,
        )

    else:
        plot_2 = None
        gr_2 = None
        graph_2 = None

    # Creation of the global file with the different global tool in the window and show/save
    show_save_with_graphs(
        plot_1,
        graph_1,
        gr_1,
        name_file=name_file,
        plot_2=plot_2,
        graph_2=graph_2,
        gr_2=gr_2,
        choice_col_type=choice_col_type,
        labels_col=labels_col,
        labels_2=labels_2,
        color_non_selected_nodes=color_non_selected_nodes,
        palette=palette,
    )
