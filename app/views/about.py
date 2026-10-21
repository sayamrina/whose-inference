import json

import streamlit as st

import ui

summary = json.loads((ui.ROOT / "results" / "predict.json").read_text())["summary"]

st.title("About the study")
st.markdown("When a tutor writes “the result was significant, p = .03”, a statistician and a student with a common "
            "misconception draw different conclusions from the same sentence. A cooperative AI system should know "
            "which reader it is talking to, predict how that reader will understand a message, and phrase it so the "
            "intended meaning gets through. It should do this from an explicit, checkable record of what the reader "
            "knows and believes, and it should really use that record.")
st.markdown("This pilot tests each step of that chain on a small scale. It was built for the doctoral topic "
            "*Cooperative User Interfaces as Rational Agents* at Saarland University.")

st.subheader("What was done")
st.markdown("""
1. **Data.** 24 statement–conclusion pairs about six statistics concepts with well-known misconceptions
   (p-values, significance vs. effect size, non-significant results, confidence intervals, correlation vs. causation,
   standard error). Each pair has a label for a reader who understands the concept and one for a reader who holds
   the misconception. Three readers are stored as small knowledge graphs.
2. **Predicting a reader.** GPT-4o-mini predicts a reader's conclusion with no knowledge base, with the whole
   knowledge base as text, and with only the retrieved entries.
3. **Rephrasing for a reader.** A speaker model writes candidates; a simulated reader scores them (Rational Speech Acts).
4. **Inside the model.** Activation patching and steering on Qwen2.5-0.5B.
""")

st.subheader("Main results")
rows = [("No knowledge base", summary["no_pkb"]), ("Whole knowledge base as text", summary["pkb_text"]),
        ("Retrieved entries", summary["pkb_retrieved"]), ("Retrieved entries, GPT-4o", summary["pkb_retrieved_gpt4o"])]
st.dataframe([{"Condition": n, "Correct (draws it or not)": f"{s['critical_binary']:.0%}",
               "Correct (three labels)": f"{s['critical']:.0%}"} for n, s in rows], hide_index=True)
cf = summary["counterfactual"]
st.markdown(f"""
- Retrieving the relevant entries works better than putting the whole knowledge base into the prompt.
- When one concept in a reader's knowledge base is changed, the prediction for that concept follows in
  {cf['sensitivity_correct_after_edit']:.0%} of cases, and unrelated predictions stay the same in
  {cf['specificity_unchanged_elsewhere']:.1%}. The model often reads the right entry but still answers from its own knowledge.
- Choosing among rewrites with a simulated reader only helped when that reader was a strong model, and two model
  evaluators disagreed widely. Whether the rewrites help people needs a study with real readers.
- In the small model, the user's belief and the true fact are kept apart in most cases and can be steered.
""")

st.subheader("Limits")
st.markdown("""
- Tiny, self-made data: 24 items, labelled by one person, no human reader data.
- The three readers are idealised; real people are less consistent.
- The rephrasing results are judged by language models, not people.
- The interpretability part uses a 0.5B model and simple facts; it may not hold for larger models.
""")

st.subheader("Who made it")
st.markdown("Amrina (M.Sc. Educational Technology, Saarland University). Study design, item labels and "
            "interpretation are Amrina's own; AI was used as a supporting tool for coding. "
            "[Full report on GitHub](https://github.com/sayamrina/whose-inference) · "
            "[Portfolio](https://sayamrina.github.io/)")
