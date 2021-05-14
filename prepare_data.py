import pandas as pd
import numpy as np
from itertools import compress


def load_data():
    edges = pd.read_excel(
        "data/edges_with_blocks.xlsx", sheet_name="edges", engine="openpyxl"
    )

    remedy_edges = edges.query(
        f"label_source == 'SUBSTANCE' \
        and label_target == 'EFFECT'"
    )

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
