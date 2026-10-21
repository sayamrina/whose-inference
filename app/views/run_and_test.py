import streamlit as st

st.title("Run and test")
st.markdown("Everything in this app can be reproduced on your own computer. You need Python 3.11 or newer and Git.")

st.subheader("1. Get the code")
st.code("git clone https://github.com/sayamrina/whose-inference.git\ncd whose-inference\n"
        "python -m venv .venv && source .venv/bin/activate", language="bash")

st.subheader("2. Run this app")
st.code("pip install -r app/requirements.txt\nstreamlit run app/streamlit_app.py", language="bash")
st.markdown("The dataset examples work without an API key, because their model answers are stored in "
            "`results/llm_cache.jsonl`. For your own text, set an OpenAI key first:")
st.code("export OPENAI_API_KEY=sk-...", language="bash")

st.subheader("3. Run the tests")
st.code("pip install pytest\npytest", language="bash")
st.markdown("""
The tests need no API key and no internet connection. They check that:
- the dataset is complete (24 items, two critical and two control items per concept, valid labels),
- each reader's knowledge base is built and retrieved correctly, and the gold labels follow from it,
- every page of this app opens without an error.

The same tests run on GitHub for every change.
""")

st.subheader("4. Reproduce the experiments")
st.code("pip install -r requirements.txt\n"
        "python src/predict.py   # predicting readers (about $2 of API use in total, answers are cached)\n"
        "python src/rsa.py       # rephrasing for readers\n"
        "python src/mech.py      # inside the model, runs on a CPU in about 30 minutes\n"
        "python src/plots.py     # figures", language="bash")
st.markdown("Details and the full report: [README on GitHub](https://github.com/sayamrina/whose-inference#readme).")
