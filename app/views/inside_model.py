import streamlit as st

import ui

st.title("Inside the model")
st.markdown("A cooperative assistant must keep two things apart: **what the user believes** and **what is true**. "
            "Here a small open model, Qwen2.5-0.5B-Instruct, was given 48 knowledge-base entries in which the user "
            "believes something false, such as “the capital of France is Lyon”. These results were computed in advance.")

st.markdown("""
| Question to the model | Answer follows the user's belief |
|---|---|
| No knowledge base, “What is the capital…” | 0% |
| “What does **the user believe** is the capital…” | 100% |
| “What **is** the capital…” (belief in the knowledge base) | 17% |
""")
st.caption("In 17% of cases the user's false belief leaks into a factual answer.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Where the belief travels**")
    st.image(str(ui.ROOT / "results" / "patching.png"))
    st.caption("Activation patching: the information sits at the knowledge-base entry until about layer 15, then "
               "moves to the answer in two steps, at layer 16 and at layers 20–22.")
with c2:
    st.markdown("**Steering the model**")
    st.image(str(ui.ROOT / "results" / "steering.png"))
    st.caption("Adding a “user perspective” direction at layer 12 switches factual questions to the user's belief "
               "(up to 100%), while prompts without a knowledge base keep the true answer.")
