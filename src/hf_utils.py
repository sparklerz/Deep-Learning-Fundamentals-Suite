from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from huggingface_hub import hf_hub_download


@st.cache_data(show_spinner=False)
def hf_download(repo_id: str, filename: str, repo_type: str = "model") -> str:
    """Download a file from Hugging Face Hub and return local cached path."""
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


@st.cache_data(show_spinner=False)
def hf_read_json(repo_id: str, filename: str, repo_type: str = "model") -> Dict[str, Any]:
    path = hf_download(repo_id, filename, repo_type=repo_type)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def hf_load_pickle(repo_id: str, filename: str, repo_type: str = "model") -> Any:
    path = hf_download(repo_id, filename, repo_type=repo_type)
    with open(path, "rb") as f:
        return pickle.load(f)
