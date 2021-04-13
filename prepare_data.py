import pandas as pd
import numpy as np
from itertools import compress


def load_data():
    edges = pd.read_excel(
        "data/edges_with_blocks.xlsx", sheet_name="edges", engine="openpyxl"
    )
    blocks = pd.read_excel(
        "data/edges_with_blocks.xlsx", sheet_name="blocks_0", engine="openpyxl"
    )

    remedy_block_list = blocks["remedy block"].eq("X")
    remedy_blocks = list(compress(range(len(remedy_block_list)), remedy_block_list))

    possible_block_list = blocks["possible remedy block"].eq("X")
    possible_blocks = list(
        compress(range(len(possible_block_list)), possible_block_list)
    )

    remedy_edges = edges.query(
        f"block_level_0_source in {remedy_blocks} \
        and label_source == 'SUBSTANCE' \
        and label_target == 'EFFECT'"
    )
    remedy_edges["remedy_type"] = "Remedy"

    possible_edges = edges.query(
        f"block_level_0_source in {possible_blocks} \
        and label_source == 'SUBSTANCE' \
        and label_target == 'EFFECT'"
    )
    possible_edges["remedy_type"] = "Possible Remedy"

    remedy_edges = remedy_edges.append(possible_edges)

    source_nodes = remedy_edges[
        [
            "source_text",
            "block_level_0_source",
            "block_level_1_source",
            "count_source",
            "label_source",
            "remedy_type",
        ]
    ].drop_duplicates()
    source_nodes = source_nodes.rename(
        columns={
            "source_text": "id",
            "block_level_0_source": "block_level_0",
            "block_level_1_source": "block_level_1",
            "count_source": "count",
            "label_source": "label",
        }
    )
    source_nodes["count_log"] = np.log(source_nodes["count"])

    target_nodes = remedy_edges[
        [
            "target_text",
            "block_level_0_target",
            "block_level_1_target",
            "count_target",
            "label_target",
        ]
    ].drop_duplicates()
    target_nodes = target_nodes.rename(
        columns={
            "target_text": "id",
            "block_level_0_target": "block_level_0",
            "block_level_1_target": "block_level_1",
            "count_target": "count",
            "label_target": "label",
        }
    )
    target_nodes["count_log"] = np.log(target_nodes["count"])

    remedy_edges = remedy_edges[
        [
            "source_text",
            "target_text",
            "remedy_type",
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
