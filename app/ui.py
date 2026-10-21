"""Shared look, reader identities and input widgets for all pages."""
from pathlib import Path

import streamlit as st

import pkb
import predict

ROOT = Path(__file__).resolve().parent.parent
MAX_CALLS = 15  # live model runs per visitor session, to keep API costs bounded

# Each reader keeps one colour on every page, so people can follow them through the app.
READERS = {
    "expert": {"name": "Methods lecturer", "color": "#0f8a83"},
    "mixed": {"name": "Student after one statistics course", "color": "#c27a0e"},
    "misconceived": {"name": "Teacher who reads study news", "color": "#c2307f"},
}
LABEL_TEXT = {"entailment": "draws this conclusion", "neutral": "does not draw it", "contradiction": "rejects it"}
STATE_OPTIONS = ["understands it", "holds the misconception", "no entry"]

CSS = """
<style>
.block-container { max-width: 1080px; padding-top: 2.2rem; }
.wi-lede { font-size: 1.15rem !important; line-height: 1.6 !important; max-width: 46rem; margin: 0 0 1.4rem; }
.wi-quote { font-size: clamp(1.35rem, 2.6vw, 1.9rem) !important; line-height: 1.35 !important; font-weight: 700; max-width: 44rem;
  margin: .4rem 0 1.6rem; padding-left: 1rem; border-left: 4px solid currentColor; }
.wi-readers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-bottom: .6rem; }
.wi-reader { border-top: 4px solid var(--c); padding: .7rem 0 0; }
.wi-reader .who { font-weight: 800; color: var(--c); }
.wi-reader .knows { font-size: .92rem; opacity: .8; margin: .25rem 0 .6rem; }
.wi-reader .verdict { font-size: 1.05rem; }
.wi-reader .verdict b { color: var(--c); }
.wi-note { font-size: .9rem; opacity: .75; }
.wi-steps { counter-reset: s; list-style: none; padding: 0; margin: .5rem 0 0; }
.wi-steps li { counter-increment: s; position: relative; padding: 0 0 1rem 2.6rem; }
.wi-steps li::before { content: counter(s); position: absolute; left: 0; top: 0; width: 1.8rem; height: 1.8rem;
  border-radius: 50%; border: 2px solid currentColor; display: grid; place-items: center; font-weight: 800; font-size: .9rem; }
.wi-chip { display: inline-block; padding: .05rem .5rem; border-radius: 4px; border: 1.5px solid var(--c); color: var(--c);
  font-weight: 700; font-size: .85rem; }
.wi-triple { font-family: 'Atkinson Hyperlegible Mono', monospace; font-size: .88rem; padding: .35rem .6rem; margin: .25rem 0;
  border-left: 3px solid var(--c, #888); background: rgba(127, 127, 127, .08); }
.wi-stat { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.project-popup { display: flex; align-items: center; gap: 12px; width: fit-content; max-width: 100%; margin: 12px auto 18px;
  padding: 8px 10px 8px 14px; background: #fff; color: #18212b; font-size: 14px; line-height: 1.4;
  border: 1px solid #dde3ea; border-left: 4px solid #2b3a8c; border-radius: 10px; box-shadow: 0 8px 24px rgba(0, 0, 0, .10); }
.project-popup-badge { display: grid; place-items: center; flex: none; width: 26px; height: 26px; border-radius: 50%;
  background: #2b3a8c; color: #fff; font-size: 13px; }
.project-popup a { flex: none; padding: 7px 12px; background: #2b3a8c; color: #fff !important; font-weight: 700;
  text-decoration: none !important; border-radius: 6px; white-space: nowrap; }
.project-popup a:hover { background: #1f2a6b; }
@media (max-width: 720px) {
  .wi-readers { grid-template-columns: minmax(0, 1fr); }
  .project-popup { flex-wrap: wrap; }
}
</style>
"""

# Portfolio banner: always visible, cannot be closed, opens the portfolio in a new tab
PORTFOLIO_BANNER = """
<aside class="project-popup" aria-label="Portfolio project">
  <span class="project-popup-badge" aria-hidden="true">★</span>
  <span>This app was developed by Amrina as part of a portfolio.</span>
  <a href="https://sayamrina.github.io/" target="_blank" rel="noopener">View the portfolio →</a>
</aside>
"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def chip(reader):
    r = READERS[reader]
    return f'<span class="wi-chip" style="--c:{r["color"]}">{r["name"]}</span>'


def budget_ok():
    st.session_state.setdefault("calls", 0)
    if st.session_state.calls >= MAX_CALLS:
        st.warning("You have used all live model runs for this visit. Reload the page later to try again, "
                   "or pick an example from the dataset, which loads from the cache.")
        return False
    st.session_state.calls += 1
    return True


def reader_editor(key):
    """Pick one of the three readers, or describe your own reader concept by concept."""
    profiles = pkb.load_profiles()
    choice = st.radio("Who is reading?", list(READERS) + ["custom"], key=f"{key}_reader",
                      format_func=lambda k: READERS[k]["name"] if k in READERS else "My own reader")
    if choice != "custom":
        prof = profiles[choice]
        st.caption(prof["description"])
        return choice, prof
    prof = {"description": "A reader described by you.", "knows": [], "believes": []}
    st.caption("For each concept, say what your reader thinks.")
    cols = st.columns(2)
    for i, (cid, c) in enumerate(pkb.load_concepts().items()):
        with cols[i % 2]:
            s = st.selectbox(c["name"], STATE_OPTIONS, key=f"{key}_{cid}", help=f"The misconception: {c['misconception']}")
            if s == STATE_OPTIONS[0]:
                prof["knows"].append(cid)
            elif s == STATE_OPTIONS[1]:
                prof["believes"].append(c["misconception_id"])
    return "custom", prof


def text_inputs(key, hyp_label="Possible conclusion"):
    items = pkb.load_items()
    critical = [i["id"] for i in items if i["type"] == "critical"]
    ex = st.selectbox("Start from an example", critical + ["mine"], key=f"{key}_ex",
                      format_func=lambda i: "Write my own text" if i == "mine"
                      else next(x["hypothesis"] for x in items if x["id"] == i))
    base = next((x for x in items if x["id"] == ex), {"premise": "", "hypothesis": ""})
    premise = st.text_area("Statement", base["premise"], key=f"{key}_p_{ex}", height=90)
    hyp = st.text_input(hyp_label, base["hypothesis"], key=f"{key}_h_{ex}")
    item = dict(base) if ex != "mine" else {"id": "custom", "type": "critical"}
    item.update(premise=premise, hypothesis=hyp)
    return item


def build_graph(name, prof):
    return pkb.build_pkb(name) if name != "custom" else predict.build_from_profile(prof)


def show_pkb(g, item, reader):
    hits = pkb.retrieve(g, item["premise"] + " " + item["hypothesis"])
    color = READERS.get(reader, {}).get("color", "#888")
    if not hits:
        st.info("Nothing in this reader's knowledge base matches the concepts in this text, "
                "so the model treats them like any other reader.")
        return
    st.markdown("**What the reader's knowledge base says about this text**")
    for _, r, o in hits:
        html(f'<div class="wi-triple" style="--c:{color}">reader <b>{r.replace("_", " ")}</b>: {o}</div>')


def runs_left():
    st.session_state.setdefault("calls", 0)
    return MAX_CALLS - st.session_state.calls
