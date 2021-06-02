import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import base64

from enum import IntEnum, unique
from random import choice


@unique
class NodeType(IntEnum):
    Effect = 0
    Remedy = 1
    Possible_Remedy = 2

    def __str__(self):
        return self.name.replace("_", " ")


@st.cache
def load_data():
    remedy_edges = pd.read_parquet("data/remedy_edges.parquet")
    source_nodes = pd.read_parquet("data/source_nodes.parquet")
    target_nodes = pd.read_parquet("data/target_nodes.parquet")

    substance_categories = source_nodes.category.unique().tolist()
    effect_categories = target_nodes.category.unique().tolist()

    nodes = source_nodes.append(target_nodes).drop_duplicates()

    return substance_categories, effect_categories, nodes, remedy_edges


@st.cache
def assign_colors(nodes: pd.DataFrame):
    colors = px.colors.qualitative.Light24
    categories = nodes["category"].unique()
    color_assignments = {}

    for category in categories:
        color_assignments[category] = choice(colors)

    return color_assignments


def filter_on_category(
    category: str,
    filter_type: str,
    nodes: pd.DataFrame,
):
    return nodes[
        nodes["label"].eq(filter_type.upper()) & nodes["category"].eq(category)
    ].id.tolist()


def filter_on_node(
    filter_node: str,
    filter_type: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):
    if filter_type == "Substance":
        filtered_edges = edges[edges["from"].eq(filter_node)]
        filtered_nodes = nodes[
            nodes["id"].eq(filter_node) | nodes["id"].isin(filtered_edges["to"])
        ]
    elif filter_type == "Effect":
        filtered_edges = edges[edges["to"].eq(filter_node)]
        filtered_nodes = nodes[
            nodes["id"].eq(filter_node) | nodes["id"].isin(filtered_edges["from"])
        ]
    else:
        raise ValueError("Invalid filter type. Try 'Substance' or 'Effect'.")
    return filtered_nodes, filtered_edges


def filter_on_edge_weights(
    ppmi_range: tuple,
    edge_count_range: tuple,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):
    filtered_edges = edges[edges["ppmi"].between(ppmi_range[0], ppmi_range[1])]
    filtered_edges = filtered_edges[
        filtered_edges["edge_count"].between(edge_count_range[0], edge_count_range[1])
    ]
    filtered_nodes = nodes[
        nodes["id"].isin(filtered_edges["from"])
        | nodes["id"].isin(filtered_edges["to"])
    ]
    return filtered_nodes, filtered_edges


def get_table_download_link(df: pd.DataFrame, node: str):
    """
    Generates a link allowing the data in a given dataframe to be downloaded
    in:  dataframe, filtered node
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{node}_edges.csv">Download csv file</a>'
    return href


def make_grouped_layout(G: nx.Graph, filter_node: str):
    # start from a circular layout
    pos = nx.spring_layout(G)

    # prep center points (along circle perimeter) for the clusters
    node_categories = list(set(nx.get_node_attributes(G, "category").values()))
    angles = np.linspace(0, 2 * np.pi, 1 + len(node_categories))
    reposition = {}
    radius = 3
    for category, angle in zip(node_categories, angles):
        if angle > 0:
            reposition[category] = np.array(
                [radius * np.cos(angle), radius * np.sin(angle)]
            )
        else:
            reposition[category] = np.array([0, 0])

    # adjust each point to be relative to its category's center point
    for node in pos.keys():
        if G.nodes[node]["id"] == filter_node:
            pass
        category = G.nodes[node]["category"]
        pos[node] += reposition[category]

    return pos


def make_edge_traces(G: nx.Graph):
    """
    Makes edge traces for plotting a network. Must make a separate trace for each edge
    in order to vary width by PPMI. Also makes a transparent node trace for the edge's
    midpoint in order to add hover actions to the edges.
    in: networkx Graph of filtered data
    out: list of edge traces, midpoint trace
    """
    edge_traces = []
    edge_midpoint_x = []
    edge_midpoint_y = []
    edge_midpoint_text = []

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        ppmi = G.edges[edge]["ppmi"]
        width = max(ppmi, 0.25)
        edge_count = G.edges[edge]["edge_count"]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color="gray"),
            mode="lines",
            showlegend=False,
        )
        edge_traces.append(trace)

        edge_midpoint_x.append((x0 + x1) / 2)
        edge_midpoint_y.append((y0 + y1) / 2)
        edge_midpoint_text.append(
            f"# connections = {edge_count}<br>ppmi = {round(ppmi, 2)}"
        )

    edge_midpoint_trace = go.Scatter(
        x=edge_midpoint_x,
        y=edge_midpoint_y,
        mode="markers",
        text=edge_midpoint_text,
        hoverinfo="text",
        marker=dict(color="lightgrey", opacity=0, size=50),
        showlegend=False,
    )
    return edge_traces, edge_midpoint_trace


def make_node_traces(G: nx.Graph):
    node_traces = []
    node_categories = list(set(nx.get_node_attributes(G, "category").values()))

    def add_trace(node, category):
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]["count_log"] * 5)
        node_name.append(node)
        hover_info.append(
            f"{node}<br>count = {G.nodes[node]['count']}<br>category = {category}"
        )

    for category in node_categories:

        node_x = []
        node_y = []
        node_size = []
        node_name = []
        hover_info = []

        for node in G.nodes():
            if G.nodes[node]["category"] == category:
                add_trace(node, category)
            else:
                pass

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_name,
            customdata=hover_info,
            hovertemplate="%{customdata}",
            marker=dict(
                color=color_assignments[category],
                size=node_size,
                line_width=2,
                opacity=1,
            ),
            name=str(category),
            textposition="bottom center",
        )
        node_traces.append(trace)

    return node_traces


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Withdrawal Remedy Explorer")
    col1, col2 = st.beta_columns([2, 3])

    substance_categories, effect_categories, nodes, edges = load_data()
    color_assignments = assign_colors(nodes)

    with col1:
        filter_type = st.radio(
            label="Filter by Substance or Effect?", options=("Substance", "Effect")
        )
        if filter_type == "Substance":
            filter_cat = st.selectbox(
                label="Select Substance Category",
                options=substance_categories,
            )
        elif filter_type == "Effect":
            filter_cat = st.selectbox(
                label="Select Effect Category",
                options=effect_categories,
            )
        filtered_node_list = filter_on_category(
            category=filter_cat, filter_type=filter_type, nodes=nodes
        )
        filter_node = st.selectbox(
            label=f"Select {filter_type}",
            options=filtered_node_list,
        )
        filtered_nodes, filtered_edges = filter_on_node(
            filter_node=filter_node,
            filter_type=filter_type,
            nodes=nodes,
            edges=edges,
        )
        PPMI_MAX = float(filtered_edges.ppmi.max())
        EDGE_COUNT_MAX = int(filtered_edges.edge_count.max())

        ppmi_range = st.slider(
            label="Select PPMI edge weight range to include",
            min_value=0.0,
            max_value=PPMI_MAX,
            value=(0.0, PPMI_MAX),
        )
        edge_count_range = st.slider(
            label="Select edge count range to include",
            min_value=0,
            max_value=EDGE_COUNT_MAX,
            value=(0, EDGE_COUNT_MAX),
        )
        filtered_nodes, filtered_edges = filter_on_edge_weights(
            ppmi_range=ppmi_range,
            edge_count_range=edge_count_range,
            nodes=filtered_nodes,
            edges=filtered_edges,
        )
        st.write(filtered_edges)
        st.markdown(
            get_table_download_link(filtered_edges, filter_node), unsafe_allow_html=True
        )

    with col2:
        G = nx.from_pandas_edgelist(
            filtered_edges,
            source="from",
            target="to",
            edge_attr=True,
        )

        # Set each column we'll use from nodes df as a node attribute
        for col in ["id", "category", "count", "label", "count_log"]:
            nx.set_node_attributes(
                G,
                pd.Series(
                    filtered_nodes[col].values, index=filtered_nodes["id"]
                ).to_dict(),
                name=col,
            )

        # Lay out the network and add node position as a node attribute
        pos = make_grouped_layout(G, filter_node)
        nx.set_node_attributes(G, pos, name="pos")

        # Make traces for plotting
        edge_traces, edge_midpoint_trace = make_edge_traces(G)
        node_traces = make_node_traces(G)

        # Make the plotly figure
        fig = go.Figure(
            layout=go.Layout(
                height=720,
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        )
        for trace in edge_traces:
            fig.add_trace(trace)
        for trace in node_traces:
            fig.add_trace(trace)
        fig.add_trace(edge_midpoint_trace)

        st.plotly_chart(fig, use_container_width=True)
