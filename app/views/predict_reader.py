import streamlit as st

import predict
import ui

st.title("Predict a reader")
st.markdown("Pick a statement, a conclusion someone might draw from it, and a reader. The model looks up what this "
            "reader knows and believes, then predicts whether they draw the conclusion.")

left, right = st.columns([1.1, 1], gap="large")
with left:
    item = ui.text_inputs("t1")
    name, prof = ui.reader_editor("t1")
    go = st.button("Predict how this reader understands it", type="primary", key="t1_go")
    runs_slot = st.empty()
with right:
    if go and not (item["premise"].strip() and item["hypothesis"].strip()):
        st.warning("Write a statement and a possible conclusion first.")
    elif go and ui.budget_ok():
        g = ui.build_graph(name, prof)
        ui.show_pkb(g, item, name)
        with st.spinner("Asking the model…"):
            label, dist = predict.predict(item, g, "pkb_retrieved")
            base_label, _ = predict.predict(item, g, "no_pkb")
        r = ui.READERS.get(name, {"name": "Your reader", "color": "inherit"})
        ui.html(f'<h3 style="margin-top:.6rem"><span style="color:{r["color"]}">{r["name"]}</span> '
                f'{ui.LABEL_TEXT[label]}</h3>')
        for lab in ["entailment", "neutral", "contradiction"]:
            st.progress(float(dist[lab]), text=f"{ui.LABEL_TEXT[lab].capitalize()}: {dist[lab]:.0%}")
        st.caption(f"Without the knowledge base, the model expects a typical reader to say: {ui.LABEL_TEXT[base_label]}.")
        if item.get("id") != "custom" and name != "custom":
            gold = predict.gold_for(item, prof)
            verdict = "matches" if gold == label else "does not match"
            st.caption(f"The dataset label for this reader is “{ui.LABEL_TEXT[gold]}”, so the prediction {verdict}.")
    else:
        st.info("Your result appears here. Not sure where to start? Keep the first example, choose the "
                "**Teacher who reads study news**, and press the button.")

runs_slot.caption(f"Live model runs left this visit: {ui.runs_left()}. Examples from the dataset load from a cache.")
