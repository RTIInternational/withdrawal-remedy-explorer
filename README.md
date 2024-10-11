# Withdrawal Remedy Explorer

This repo contains the files needed to deploy the [Withdrawal Remedy Explorer app](https://share.streamlit.io/rtiinternational/withdrawal-remedy-explorer) on Streamlit Sharing. 

## Contents

- **prepare_data.py** was used to filter and transform the source data, `edges.parquet` and `nodes.parquet`, and generate the files read by the app: `remedy_edges.parquet`, `source_nodes.parquet`, and `target_nodes.parquet`.

- **streamlit_app.py** runs the app. 

## Instructions

General users of the Withdrawal Remedy Explorer app should use the app [on Streamlit Sharing](https://share.streamlit.io/rtiinternational/withdrawal-remedy-explorer) rather than through this repo. 

For app developers, here are some instructions to get started.

### Dependency management

This repo includes both `uv.lock` and `requirements.txt`. This is an unusual choice, motivated by the deployment on Streamlit Sharing (at the time of this decision, Streamlit did not have functionality to use `uv` for dependency management). 

This provides two options for running the app locally:

1. Create a virtual environment using `uv venv` and install the dependencies with `uv sync`.
1. Create a Python virtual environment using your tool of choice and install the dependencies with `pip install -r requirements.txt`. 

The big downside is keeping these files in sync. If you add dependencies using `uv add`, make sure to then run `uv export --format requirements-txt > requirements.txt`.

### Running the app locally

Run `streamlit run streamlit_app.py`. 

### Updating input data

- To update the input data, replace `edges.parquet` and `nodes.parquet` and run `python prepare_data.py`. Note that the data must be in the same format as the current version of `edges.parquet` and `nodes.parquet`. For more information on how to update these files, contact the maintainers.
- This repo only contains the files necessary to run the app. The rest of the files and data associated with this project are stored in a separate, private repo. If you think you need access to those, contact the maintainers. 
