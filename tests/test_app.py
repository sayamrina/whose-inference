"""Every page of the Streamlit app opens without an exception (no model calls are made on load)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
PAGES = {"overview": "Whose Inference?", "predict_reader": "Predict a reader", "rephrase": "Rephrase for a reader",
         "inside_model": "Inside the model", "how_to_use": "How to use", "about": "About the study",
         "run_and_test": "Run and test"}


@pytest.mark.parametrize("page", list(PAGES))
def test_page_renders(page):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py")
        at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == PAGES[page]
