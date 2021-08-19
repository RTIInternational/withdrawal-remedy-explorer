# Withdrawal Remedy Explorer

This repo contains the files needed to deploy the [Withdrawal Remedy Explorer app](https://share.streamlit.io/rtiinternational/withdrawal-remedy-explorer) on Streamlit Sharing. 

## Contents

- **prepare_data.py** was used to filter and transform the source data, `edges.parquet` and `nodes.parquet`, and generate the files read by the app: `remedy_edges.parquet`, `source_nodes.parquet`, and `target_nodes.parquet`.

- **streamlit_app.py** runs the app. 

## Instructions

General users of the Withdrawal Remedy Explorer app should use the app [on Streamlit Sharing](https://share.streamlit.io/rtiinternational/withdrawal-remedy-explorer) rather than through this repo. 

For app developers, here are some instructions to get started.

- To run the app locally, create a new Python virtual environment and install the dependencies with `pip install -r requirements.txt`. Then run `streamlit run streamlit_app.py`. 
- To update the source data, replace `edges.parquet` and `nodes.parquet` and run `python prepare_data.py`. Note that the data must be in the same format as the current version of `edges.parquet` and `nodes.parquet`. For more information on how to update these files, contact the maintainers.
- This repo only contains the files necessary to run the app. The rest of the files and data associated with this project are stored in a separate, private repo. If you think you need access to those, contact the maintainers. 
