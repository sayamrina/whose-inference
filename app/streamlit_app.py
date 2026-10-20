"""Whose Inference? Interactive demo.

Predict how a specific reader will interpret a statistical statement from their personal knowledge base (PKB),
and rephrase the statement so this reader does not misread it (RSA-style).
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Whose Inference?", page_icon="🧠", layout="wide")

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

import pkb  # noqa: E402
import predict  # noqa: E402
import rsa  # noqa: E402

MAX_CALLS = 15  # live model runs per visitor session, to keep API costs bounded
LABEL_TEXT = {"entailment": "draws this conclusion", "neutral": "does not draw it", "contradiction": "rejects it"}
STATE_OPTIONS = ["understands it", "holds the misconception", "no entry"]

concepts = pkb.load_concepts()
profiles = pkb.load_profiles()
items = pkb.load_items()
st.session_state.setdefault("calls", 0)


def budget_ok():
    if st.session_state.calls >= MAX_CALLS:
        st.warning("This demo allows a limited number of live model runs per visit. Reload the page later to try again.")
        return False
    st.session_state.calls += 1
    return True


def reader_editor(key):
    """Pick a preset reader, or set the reader's knowledge per concept."""
    names = {"expert": "Methods lecturer", "mixed": "Student after one statistics course",
             "misconceived": "Teacher reading study summaries in the news"}
    choice = st.radio("Reader", list(names) + ["custom"], key=f"{key}_reader", horizontal=True,
                      format_func=lambda k: names.get(k, "Custom reader"))
    if choice != "custom":
        prof = profiles[choice]
        st.caption(prof["description"])
        return choice, prof
    prof = {"description": "A custom reader.", "knows": [], "believes": []}
    cols = st.columns(2)
    for i, (cid, c) in enumerate(concepts.items()):
        with cols[i % 2]:
            s = st.selectbox(c["name"], STATE_OPTIONS, key=f"{key}_{cid}",
                             help=f"Misconception: {c['misconception']}")
            if s == STATE_OPTIONS[0]:
                prof["knows"].append(cid)
            elif s == STATE_OPTIONS[1]:
                prof["believes"].append(c["misconception_id"])
    return "custom", prof


def text_inputs(key):
    ex = st.selectbox("Example", ["Write my own"] + [i["id"] for i in items if i["type"] == "critical"], index=1, key=f"{key}_ex",
                      format_func=lambda i: i if i == "Write my own" else next(x["hypothesis"] for x in items if x["id"] == i))
    base = next((x for x in items if x["id"] == ex), {"premise": "", "hypothesis": ""})
    premise = st.text_area("Statement (premise)", base["premise"], key=f"{key}_p_{ex}", height=80)
    hyp = st.text_input("Possible conclusion (hypothesis)", base["hypothesis"], key=f"{key}_h_{ex}")
    item = dict(base) if ex != "Write my own" else {"id": "custom", "type": "critical"}
    item.update(premise=premise, hypothesis=hyp)
    return item


def show_pkb(g, item):
    hits = pkb.retrieve(g, item["premise"] + " " + item["hypothesis"])
    if hits:
        st.markdown("**Retrieved from the reader's PKB**")
        for s, r, o in hits:
            st.markdown(f"- `{r}` {o}")
    else:
        st.info("No PKB entries match the concepts in this text, so the model treats the reader like anyone else.")


# Portfolio banner: always visible, cannot be closed, opens the portfolio in a new tab
PORTFOLIO_BANNER = """
<style>
.project-popup { display: flex; align-items: center; gap: 12px; width: fit-content; max-width: 100%; margin: 4px auto 14px;
  padding: 8px 10px 8px 14px; background: #fff; color: #18212b; font-size: 14px; line-height: 1.4;
  border: 1px solid #dde3ea; border-left: 4px solid #1f5fa8; border-radius: 12px; box-shadow: 0 10px 28px rgba(0, 0, 0, .10); }
.project-popup-badge { display: grid; place-items: center; flex: none; width: 26px; height: 26px; border-radius: 50%;
  background: #1f5fa8; color: #fff; font-size: 13px; }
.project-popup a { flex: none; padding: 7px 12px; background: #173f6e; color: #fff !important; font-weight: 700;
  text-decoration: none !important; border-radius: 8px; white-space: nowrap; }
.project-popup a:hover { background: #1f5fa8; }
@media (max-width: 640px) { .project-popup { flex-wrap: wrap; } }
</style>
<aside class="project-popup" aria-label="Portfolio project">
  <span class="project-popup-badge" aria-hidden="true">★</span>
  <span>This app was developed by Amrina as part of a portfolio.</span>
  <a href="https://sayamrina.github.io/" target="_blank" rel="noopener">View the portfolio →</a>
</aside>
"""
st.markdown(PORTFOLIO_BANNER, unsafe_allow_html=True)

st.title("Whose Inference?")
st.markdown("The same sentence means different things to different readers. This demo predicts **how a specific reader** "
            "will interpret a statistical statement, from an explicit record of what they know and believe, and rewrites "
            "the statement so this reader does not misread it.")
st.markdown("[Project page](https://sayamrina.github.io/whose-inference/) · "
            "[Code and report](https://github.com/sayamrina/whose-inference) · "
            "by Amrina ([portfolio](https://sayamrina.github.io/))")

tab1, tab2, tab3 = st.tabs(["1 · Predict a reader", "2 · Rephrase for this reader", "3 · Inside the model"])

with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        item = text_inputs("t1")
        pname, prof = reader_editor("t1")
        go = st.button("Predict interpretation", type="primary", key="t1_go")
    with right:
        if go and item["premise"].strip() and item["hypothesis"].strip() and budget_ok():
            g = pkb.build_pkb(pname) if pname != "custom" else predict.build_from_profile(prof)
            show_pkb(g, item)
            with st.spinner("Asking the model…"):
                label, dist = predict.predict(item, g, "pkb_retrieved")
                base_label, _ = predict.predict(item, g, "no_pkb")
            st.markdown(f"### This reader **{LABEL_TEXT[label]}**")
            for lab in ["entailment", "neutral", "contradiction"]:
                st.progress(float(dist[lab]), text=f"{lab}: {dist[lab]:.0%}")
            st.caption(f"Without the PKB, the model predicts that a typical reader {LABEL_TEXT[base_label]}.")
            if item.get("id") != "custom" and pname != "custom":
                gold = predict.gold_for(item, prof)
                st.caption(f"Label in the dataset for this reader: {gold} ({'match' if gold == label else 'mismatch'}).")
        elif not go:
            st.info("Choose a statement and a reader, then press **Predict interpretation**.")

with tab2:
    left, right = st.columns([1.1, 1])
    with left:
        item2 = text_inputs("t2")
        st.caption("The hypothesis is the misreading to avoid.")
        pname2, prof2 = reader_editor("t2")
        go2 = st.button("Rephrase for this reader", type="primary", key="t2_go")
    with right:
        if go2 and item2["premise"].strip() and item2["hypothesis"].strip() and budget_ok():
            g2 = pkb.build_pkb(pname2) if pname2 != "custom" else predict.build_from_profile(prof2)
            intended = item2.get("label_correct", "neutral")
            with st.spinner("Generating candidates and simulating the reader…"):
                before = rsa.p_correct(rsa.listener(item2["premise"], item2, g2), intended)
                best, trace = rsa.rsa(item2, g2, intended)
            st.markdown("**Original**")
            st.write(item2["premise"])
            st.caption(f"Simulated chance this reader understands it correctly: {before:.0%}")
            st.markdown("**Chosen rephrasing**")
            st.success(best)
            chosen = next(t for t in trace if t["text"] == best)
            st.caption(f"Simulated chance of correct understanding: {chosen['p_sim']:.0%}")
            with st.expander(f"All {len(trace)} candidates the speaker considered"):
                for t in sorted(trace, key=lambda t: -t["u"]):
                    st.markdown(f"- **{t['p_sim']:.0%}** · utility {t['u']:+.2f} · round {t['round'] + 1}: {t['text']}")
            st.caption("Scores come from a simulated reader (an LLM), not from real people. See the report for why that matters.")
        elif not go2:
            st.info("Pick a statement, the misreading to avoid and a reader, then press **Rephrase for this reader**.")

with tab3:
    st.markdown("Precomputed results with Qwen2.5-0.5B-Instruct (48 facts where the PKB says the user believes something false).")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ROOT / "results" / "patching.png"),
                 caption="Activation patching: the user's belief moves from the PKB entry to the answer in layers 16 and 20–22.")
    with c2:
        st.image(str(ROOT / "results" / "steering.png"),
                 caption="Steering at layer 12 switches world questions to the user's belief (21% → 100%).")

with st.sidebar:
    st.markdown("### About")
    st.markdown("A pilot study on personal knowledge bases, cooperative rephrasing (Rational Speech Acts) and "
                "interpretability. Tiny, self-made data: treat results as illustrations, not findings.")
    st.markdown(f"Live model runs left this visit: **{MAX_CALLS - st.session_state.calls}**")
    st.caption("Examples from the dataset load instantly from a cache; your own text calls GPT-4o-mini.")
