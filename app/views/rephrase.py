import streamlit as st

import rsa
import ui

st.title("Rephrase for a reader")
st.markdown("The conclusion below is the **misreading to avoid**. A speaker model writes several versions of the "
            "statement, a simulated reader with this knowledge base reads each one, and the version this reader is "
            "most likely to understand correctly wins (a Rational Speech Acts loop).")

left, right = st.columns([1.1, 1], gap="large")
with left:
    item = ui.text_inputs("t2", hyp_label="Misreading to avoid")
    name, prof = ui.reader_editor("t2")
    go = st.button("Rephrase for this reader", type="primary", key="t2_go")
    runs_slot = st.empty()
with right:
    if go and not (item["premise"].strip() and item["hypothesis"].strip()):
        st.warning("Write a statement and the misreading to avoid first.")
    elif go and ui.budget_ok():
        g = ui.build_graph(name, prof)
        intended = item.get("label_correct", "neutral")
        with st.spinner("Writing candidates and simulating the reader…"):
            before = rsa.p_correct(rsa.listener(item["premise"], item, g), intended)
            best, trace = rsa.rsa(item, g, intended)
        st.markdown("**Original statement**")
        st.write(item["premise"])
        st.caption(f"Chance this reader understands it correctly (simulated): {before:.0%}")
        st.markdown("**Rewritten for this reader**")
        st.success(best)
        chosen = next(t for t in trace if t["text"] == best)
        st.caption(f"Chance this reader understands it correctly (simulated): {chosen['p_sim']:.0%}")
        with st.expander(f"All {len(trace)} versions the speaker considered"):
            for t in sorted(trace, key=lambda t: -t["u"]):
                st.markdown(f"- **{t['p_sim']:.0%}** · score {t['u']:+.2f} · round {t['round'] + 1}: {t['text']}")
        st.caption("The chances come from a simulated reader (a language model), not from real people. "
                   "The study found that such simulated readers often disagree with each other.")
    else:
        st.info("Your rewritten statement appears here. Try the first example with the **Student after one "
                "statistics course**.")

runs_slot.caption(f"Live model runs left this visit: {ui.runs_left()}. This step makes several model calls.")
