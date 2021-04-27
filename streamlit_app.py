import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import base64


@st.cache
def load_data():
    remedy_edges = pd.read_parquet("data/remedy_edges.parquet")
    source_nodes = pd.read_parquet("data/source_nodes.parquet")
    target_nodes = pd.read_parquet("data/target_nodes.parquet")

    unique_remedies = source_nodes.id.unique().tolist()
    unique_effects = target_nodes.id.unique().tolist()

    nodes = source_nodes.append(target_nodes).drop_duplicates()

    return unique_remedies, unique_effects, nodes, remedy_edges


def filter_on_node(
    filter_node: str,
    filter_type: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
):
    if filter_type == "Remedy":
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
        raise ValueError("Invalid filter type. Try 'Remedy' or 'Effect'.")
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
        edge_count = G.edges[edge]["edge_count"]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=ppmi / 2, color="gray"),
            mode="lines",
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
        marker=dict(color="grey", opacity=0, size=50),
    )
    return edge_traces, edge_midpoint_trace


def make_node_trace(G: nx.Graph):
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        if G.nodes[node]["label"] == "EFFECT":
            node_color.append("coral")
        elif G.nodes[node]["remedy_type"] == "Remedy":
            node_color.append("cornflowerblue")
        else:
            node_color.append("lightgreen")
        node_size.append(G.nodes[node]["count_log"] * 5)
        node_text.append(f"{node}<br>count = {G.nodes[node]['count']}")

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line_width=2, opacity=1),
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Withdrawal Remedy Explorer")
    col1, col2 = st.beta_columns([2, 3])

    unique_remedies, unique_effects, nodes, edges = load_data()

    with col1:
        filter_type = st.radio(
            label="Filter by Remedy or Effect?", options=("Remedy", "Effect")
        )
        if filter_type == "Remedy":
            filter_node = st.selectbox(
                label="Select Remedy",
                options=unique_remedies,
            )
        elif filter_type == "Effect":
            filter_node = st.selectbox(
                label="Select Effect",
                options=unique_effects,
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
        for col in ["block_level_0", "count", "label", "remedy_type", "count_log"]:
            nx.set_node_attributes(
                G,
                pd.Series(
                    filtered_nodes[col].values, index=filtered_nodes["id"]
                ).to_dict(),
                name=col,
            )

        # Lay out the network and add node position as a node attribute
        pos = nx.drawing.layout.spring_layout(G)
        nx.set_node_attributes(G, pos, name="pos")

        # Make traces for plotting
        edge_traces, edge_midpoint_trace = make_edge_traces(G)
        node_trace = make_node_trace(G)

        # Make the plotly figure
        fig = go.Figure(
            layout=go.Layout(
                height=720,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        )
        for trace in edge_traces:
            fig.add_trace(trace)
        fig.add_trace(node_trace)
        fig.add_trace(edge_midpoint_trace)

        st.plotly_chart(fig, use_container_width=True)
