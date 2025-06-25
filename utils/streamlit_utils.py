import streamlit as st
from collections import defaultdict
import pandas as pd
import hashlib
import json
import os

def load_css(file_path: str):
    """
    Load custom CSS from a file into Streamlit.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at: {file_path}")

def dict_hash(dictionary: dict) -> str:
    """
    Generate a hash from a dictionary.
    Ensures consistent ordering for hashing.
    """
    dict_str = json.dumps(dictionary, sort_keys=True)
    return hashlib.sha256(dict_str.encode()).hexdigest()

def compute_avg_metrics_per_k(eval_df):
    """
    Computes the average MRR, Recall, Precision, and nDCG for each k in the evaluation results.
    """

    metric_groups = defaultdict(dict)

    for col in eval_df.columns:
        if "@" in col:
            metric, k = col.split("@")
            metric_groups[metric][f"k{k}"] = eval_df[col].mean()

    return pd.DataFrame(metric_groups).T.sort_index()


def get_component_config(config_dict, label: str, include_provider=False):
    """
    Generic function to display a selectbox in the Streamlit sidebar for a component (embedder, reranker, reader).
    """
    items = []
    for group, models in config_dict.items():
        for model in models:
            display_label = f"{group} - {model['name']}"
            if include_provider:
                items.append((display_label, model, group))
            else:
                items.append((display_label, model))

    display_labels = [label for label, *_ in items]
    selected_label = st.sidebar.selectbox(f"Select {label}", display_labels)

    if include_provider:
        selected_config, provider = next((cfg, prov) for lbl, cfg, prov in items if lbl == selected_label)
        return selected_config, provider
    else:
        selected_config = next(cfg for lbl, cfg in items if lbl == selected_label)
        return selected_config

