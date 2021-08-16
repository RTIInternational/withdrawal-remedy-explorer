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
    nodes.sort_values(by=["category"])

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


def add_within_category_edges(nodes: pd.DataFrame, edges: pd.DataFrame):
    # make edges among nodes of same category
    self_join = nodes.merge(nodes, on="category")
    self_join = self_join[self_join.id_x.ne(self_join.id_y)]
    one_edge_per_category = self_join.groupby("id_x").first().reset_index()

    # append them to actual edges after reshaping
    within_category_edges = one_edge_per_category[["id_x", "id_y", "category"]]
    within_category_edges.rename(
        columns={"id_x": "from", "id_y": "to", "category": "category_source"},
        inplace=True,
    )
    within_category_edges["category_target"] = within_category_edges["category_source"]
    within_category_edges["edge_count"] = 1
    within_category_edges["ppmi"] = 0.5

    return edges.append(within_category_edges)


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
        if G.edges[edge]["category_source"] != G.edges[edge]["category_target"]:
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
        readme = st.beta_expander("README")
        with readme:
            st.markdown(
                """
                Withdrawal Remedy Explorer lets you explore the ways people manage opioid withdrawal
                symptoms. 

                The data come from Reddit - [r/opiates](https://www.reddit.com/r/opiates/) and [r/OpiatesRecovery](https://www.reddit.com/r/OpiatesRecovery/) - where people
                discuss a variety of opioid-related topics. From the data, unique *substances* and *effects*
                were identified. Connections were drawn when a *substance* and *effect* were mentioned in
                the same sentence. A forthcoming paper will discuss our methodology in detail.

                #### Overview
                Withdrawal Remedy Explorer shows you all of the *subtances/effects* connected to
                a single *substance/effect* which you choose.

                You can use the category dropdown to narrow down to the type of *substance* or 
                *effect* you're interested in.

                PPMI and Edge Count sliders let you filter the results so you can hone in on a certain
                type of connection. For example, you might choose to filter for a relatively high 
                PPMI *and* edge count to only view connections which are both noteworthy and frequently 
                mentioned.

                Hover on the (?) icon next to each input for additional info.

                #### Filtering by Substance
                If you filter by substance, you can view all of the effects connected to a single
                substance. For example, you can filter to "heroin" and see all the effects people
                mention when they talk about heroin. 

                This is a good place to start if you're interested in a particular substance.
                For example, pharmaceutical manufacturers may be interested in the ways a certain
                drug is being used to manage opioid withdrawal symptoms.

                #### Filtering by Effect
                If you filter by effect, you can view all of the substances connected to a single
                effect. For example, you can filter to "insomnia" and see all the substances people
                mention when they talk about insomnia. 

                This is a good place to start if you're interested in a particular withdrawal
                symptom. For example, psychiatrists may be interested in the various substances
                people use to manage a particular withdrawal symptom.

                #### Notes/Limitations
                - Despite the title, *effects* are not limited to withdrawal symptoms, and *substances*
                are not limited to remedies. Although those are our primary interest, we include 
                all substances and effects present in the Reddit data.

                - Connections between a substance and effect do not necessarily imply that the
                substance was used to treat the effect. Connections could also mean that the substance
                caused the effect, or they could be mentioned in a sentence together for another reason.
                Future work will disentangle these connection types.

                - Connections do not imply that the substance was effective at treating an effect.
                **This tool in no way intends to give medical advice.**
                """
            )
        filter_type = st.radio(
            label="Filter by Substance or Effect?",
            options=("Substance", "Effect"),
            help="If you choose substance, you can view all the effects connected to a single substance. If you choose effect, you can view all the substances connected to a single effect.",
        )
        if filter_type == "Substance":
            filter_cat = st.selectbox(
                label="Select Substance Category",
                options=substance_categories,
                help="Substances are grouped into categories by their pharmacological profile and use in clinical practice. When you select a category, the list below will show substances in that category.",
            )
        elif filter_type == "Effect":
            filter_cat = st.selectbox(
                label="Select Effect Category",
                options=effect_categories,
                help="Effects are grouped into categories by their relationship to opioid use. When you select a category, the list below will show effects in that category.",
            )
        filtered_node_list = filter_on_category(
            category=filter_cat, filter_type=filter_type, nodes=nodes
        )
        filter_node = st.selectbox(
            label=f"Select {filter_type}",
            options=filtered_node_list,
            help="Choose a substance/effect from the category selected above. If you don't see the substance/effect you're looking for, try a different category.",
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
            help="PPMI is Positive Pointwise Mutual Information, a measure of strength of association. PPMI is high when the probability of a substance and effect co-ocurring is high relative to their individual probabilities of occurrence. Connections with high PPMI are noteworthy, but not necessarily common.",
        )
        edge_count_range = st.slider(
            label="Select edge count range to include",
            min_value=0,
            max_value=EDGE_COUNT_MAX,
            value=(0, EDGE_COUNT_MAX),
            help="Edge Count is the number of times a substance and effect were mentioned together. Connections with high edge count are frequently mentioned, but not necessarily noteworthy, since the edge count depends on how often the individual substances and effects were mentioned.",
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
        combined_edgelist = add_within_category_edges(filtered_nodes, filtered_edges)
        G = nx.from_pandas_edgelist(
            combined_edgelist,
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
        pos = nx.drawing.layout.spring_layout(G)
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

        st.markdown(
            "Copyright 2021 [RTI International](https://www.rti.org/). Withdrawal Remedy Explorer is an open source project. The code base is on [GitHub](https://github.com/RTIInternational/withdrawal-remedy-explorer)."
        )
