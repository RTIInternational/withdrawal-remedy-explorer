import pandas as pd
import numpy as np


def ppmi(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """
    Calculates edge weights in terms of Positive Pointwise Mutual Information (PPMI).

    PPMI scales the probability of two nodes co-occurring to the individual probability of
    each node occurring. This decouples edge weights from node counts. Node pairs have high
    PPMI when the probability of co-occurrence is only slightly lower than the probabilities
    of occurrence of each node.
    """
    sum_node_counts = nodes["count"].sum()
    sum_edge_counts = edges["edge_count"].sum()
    edge_weights_ppmi = pd.Series(
        np.log(
            (edges["edge_count"] / sum_edge_counts)
            / (
                (edges["count_source"] / sum_node_counts)
                * (edges["count_target"] / sum_node_counts)
            )
        ),
    )
    edge_weights_ppmi = np.where(edge_weights_ppmi < 0, 0, edge_weights_ppmi)
    return edge_weights_ppmi


def load_data():
    edges = pd.read_parquet("data/edges.parquet")
    nodes = pd.read_parquet("data/nodes.parquet")

    # Fix mislabeling and format labels
    nodes.loc[nodes.node.eq('insomnia'), 'category'] = 'DSM 5 symptom of opioid withdrawal'
    nodes.category = np.where(
        nodes.category == 'Not a symptom of opiod use or withdrawal', 
        'Not a DSM-5 symptom of opiod use or withdrawal', 
        nodes.category
    )
    nodes.category = np.where(
        nodes.category == 'DSM 5 symptom of opioid withdrawal', 
        'DSM-5 symptom of opioid withdrawal', 
        nodes.category
    )
    
    # This section transforms the input data into the format of the old
    # edges_with_blocks.xlsx, which we used to use as input data for the app. Now we've
    # transitioned to using the vanilla edge and node data, so some additional data
    # processing is necessary.

    # Filter out nodes with fewer than 5 occurrences and edges connecting to those nodes
    nodes = nodes.query("count >= 5").copy()

    # Filter out mislabeled nodes
    nodes = nodes[~nodes.category.isna()]
    nodes = nodes[~nodes.node.eq("other")]
    nodes = nodes[~nodes.node.eq("catnip")]

    # Filter out edges to nodes that were removed in the above operations
    edges = edges.query("source in @nodes['index'] and target in @nodes['index']")

    edges["edge_count"] = edges["count"]
    edges.drop(columns=["count"], inplace=True)
    edges = edges.merge(nodes, how="inner", left_on="source", right_on="index")
    edges = edges.merge(
        nodes,
        how="inner",
        left_on="target",
        right_on="index",
        suffixes=("_source", "_target"),
    )
    edges.drop(columns=["index_source", "index_target"], inplace=True)
    edges.rename(
        columns={"node_source": "source_text", "node_target": "target_text"},
        inplace=True,
    )
    edges = edges.query("source_text != 'outlier' and target_text != 'outlier'")
    edges_reversed = edges.copy()
    edges_reversed.rename(
        columns={
            "source_text": "target_text",
            "label_source": "label_target",
            "count_source": "count_target",
            "category_source": "category_target",
            "target_text": "source_text",
            "label_target": "label_source",
            "count_target": "count_source",
            "category_target": "category_source",
        },
        inplace=True,
    )
    edges = edges.append(edges_reversed)
    remedy_edges = edges.query(
        "label_source == 'SUBSTANCE' and label_target == 'EFFECT'"
    ).copy()
    remedy_nodes = nodes.query(
        "node in @remedy_edges.source_text or node in @remedy_edges.target_text"
    )
    remedy_edges["ppmi"] = ppmi(remedy_nodes, remedy_edges)

    # The rest of this function is the same data processing we used with
    # edges_with_blocks.xlsx

    source_nodes = remedy_edges[
        [
            "source_text",
            "count_source",
            "label_source",
            "category_source",
        ]
    ].drop_duplicates()
    source_nodes = source_nodes.rename(
        columns={
            "source_text": "id",
            "count_source": "count",
            "label_source": "label",
            "category_source": "category",
        }
    )
    source_nodes["count_log"] = np.log(source_nodes["count"])

    target_nodes = remedy_edges[
        [
            "target_text",
            "count_target",
            "label_target",
            "category_target",
        ]
    ].drop_duplicates()
    target_nodes = target_nodes.rename(
        columns={
            "target_text": "id",
            "count_target": "count",
            "label_target": "label",
            "category_target": "category",
        }
    )
    target_nodes["count_log"] = np.log(target_nodes["count"])

    remedy_edges = remedy_edges[
        [
            "source_text",
            "target_text",
            "edge_count",
            "ppmi",
            "category_source",
            "category_target",
        ]
    ]
    remedy_edges = remedy_edges.rename(
        columns={
            "source_text": "from",
            "target_text": "to",
        }
    )

    return source_nodes, target_nodes, remedy_edges


if __name__ == "__main__":
    source_nodes, target_nodes, remedy_edges = load_data()

    source_nodes.to_parquet("data/source_nodes.parquet", index=False)
    target_nodes.to_parquet("data/target_nodes.parquet", index=False)
    remedy_edges.to_parquet("data/remedy_edges.parquet", index=False)
