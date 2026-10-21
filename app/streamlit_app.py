"""Whose Inference? Interactive demo.

Predict how a specific reader will interpret a statistical statement from their personal knowledge base (PKB),
and rephrase the statement so this reader does not misread it (RSA-style).
"""
import os
import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "src"))
sys.path.insert(0, str(APP))

st.set_page_config(page_title="Whose Inference?", page_icon="🧠", layout="wide")

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

import ui  # noqa: E402

pages = {
    "": [st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True)],
    "Try it": [
        st.Page("views/predict_reader.py", title="Predict a reader", icon=":material/person_search:"),
        st.Page("views/rephrase.py", title="Rephrase for a reader", icon=":material/edit_note:"),
        st.Page("views/inside_model.py", title="Inside the model", icon=":material/memory:"),
    ],
    "Guide": [
        st.Page("views/how_to_use.py", title="How to use", icon=":material/help:"),
        st.Page("views/about.py", title="About the study", icon=":material/info:"),
        st.Page("views/run_and_test.py", title="Run and test", icon=":material/terminal:"),
    ],
}
nav = st.navigation(pages, position="top")

ui.html(ui.CSS)
ui.html(ui.PORTFOLIO_BANNER)
nav.run()

st.divider()
st.caption("Whose Inference? A pilot study by Amrina · "
           "[Project page](https://sayamrina.github.io/whose-inference/) · "
           "[Code and report](https://github.com/sayamrina/whose-inference) · "
           "Tiny, self-made data: read the results as illustrations, not findings.")
