import json

import streamlit as st

import pkb
import ui

items = {i["id"]: i for i in pkb.load_items()}
item = items["s1"]
concept = pkb.load_concepts()[item["concept"]]

st.title("Whose Inference?")
ui.html('<p class="wi-lede">The same sentence means different things to different readers. '
        'This app predicts how one specific reader will understand a statement about statistics, using an explicit '
        'record of what that reader knows and believes, and rewrites the statement so this reader does not misread it.</p>')

st.markdown(f"**A study report says:**")
ui.html(f'<p class="wi-quote">{item["premise"]}</p>')
st.markdown(f"Does the reader conclude: *“{item['hypothesis']}”*")

cards = []
for name, r in ui.READERS.items():
    prof = pkb.load_profiles()[name]
    if concept["misconception_id"] in prof["believes"]:
        knows = f"Believes: {concept['misconception'][0].lower()}{concept['misconception'][1:]}"
    elif item["concept"] in prof["knows"]:
        knows = f"Knows: {concept['correct'].split(';')[0]}."
    else:
        knows = "Has no view on this concept."
    verdict = ui.LABEL_TEXT[pkb.gold_label(item, name)]
    cards.append(f'<div class="wi-reader" style="--c:{r["color"]}"><div class="who">{r["name"]}</div>'
                 f'<div class="knows">{knows}</div><div class="verdict">This reader <b>{verdict}</b>.</div></div>')
ui.html(f'<div class="wi-readers">{"".join(cards)}</div>')
ui.html('<p class="wi-note">Labels from the dataset of this study. The app tries to predict them from each reader\'s '
        'knowledge base.</p>')

st.subheader("What you can do here")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Predict a reader**  \nChoose a statement and a reader, and see which conclusion the model "
                "expects this reader to draw, and why.")
    st.page_link("views/predict_reader.py", label="Predict a reader", icon=":material/person_search:")
with c2:
    st.markdown("**Rephrase for a reader**  \nLet the system rewrite a statement so that this reader does not "
                "fall into their misconception.")
    st.page_link("views/rephrase.py", label="Rephrase for a reader", icon=":material/edit_note:")
with c3:
    st.markdown("**Look inside the model**  \nSee where a small language model keeps the reader's belief apart "
                "from facts about the world, and how to steer it.")
    st.page_link("views/inside_model.py", label="Inside the model", icon=":material/memory:")

st.subheader("Results so far")
summary = json.loads((ui.ROOT / "results" / "predict.json").read_text())["summary"]
s1, s2, s3 = st.columns(3)
s1.markdown(f'<div class="wi-stat">{summary["no_pkb"]["critical_binary"]:.0%} → '
            f'{summary["pkb_retrieved"]["critical_binary"]:.0%}</div>', unsafe_allow_html=True)
s1.caption("Correct predictions of a reader's inference, without and with their knowledge base (GPT-4o-mini)")
s2.markdown(f'<div class="wi-stat">{summary["pkb_retrieved_gpt4o"]["critical_binary"]:.0%}</div>', unsafe_allow_html=True)
s2.caption("The same with GPT-4o")
s3.markdown(f'<div class="wi-stat">{summary["counterfactual"]["specificity_unchanged_elsewhere"]:.1%}</div>',
            unsafe_allow_html=True)
s3.caption("Predictions that stay the same when an unrelated part of the knowledge base is edited")
st.page_link("views/about.py", label="Read about the study and its limits", icon=":material/info:")
