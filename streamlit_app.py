from pyvis.network import Network
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
        net = Network(height="750px", width="100%", bgcolor="white", font_color="black")
        net.barnes_hut()  # layout solver
        for index, row in filtered_nodes.iterrows():
            if row["label"] == "EFFECT":
                net.add_node(
                    n_id=row["id"],
                    label=row["id"],
                    shape="square",
                    size=row["count_log"],
                    color="rgb(255, 193, 112)",
                )
            elif row["remedy_type"] == "Remedy":
                net.add_node(
                    n_id=row["id"],
                    label=row["id"],
                    shape="diamond",
                    size=row["count_log"],
                    color="rgb(146, 224, 167)",
                )
            elif row["remedy_type"] == "Possible Remedy":
                net.add_node(
                    n_id=row["id"],
                    label=row["id"],
                    shape="diamond",
                    size=row["count_log"],
                    color="rgb(146, 177, 224)",
                )
        for index, row in filtered_edges.iterrows():
            net.add_edge(row["from"], row["to"], value=row["ppmi"])
        net.set_options(
            """
            var options = {
            "nodes": {
                "color": {
                "highlight": {
                    "border": "rgba(132,45,233,1)",
                    "background": "rgba(221,213,255,1)"
                },
                "hover": {
                    "border": "rgba(233,125,26,1)",
                    "background": "rgba(255,229,223,1)"
                }
                },
                "font": {
                "size": 14
                }
            },
            "edges": {
                "color": {
                "inherit": true
                },
                "scaling": {
                "max": 10
                },
                "smooth": false
            }
            }
        """
        )
        net.show("net.html")
        with open("net.html") as html_file:
            html = html_file.read()
        st.components.v1.html(html, width=1000, height=700)
