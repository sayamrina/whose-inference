import streamlit as st

import ui

st.title("How to use")
st.markdown("You do not need to know statistics or AI to try this app. It takes about three minutes.")

ui.html("""
<ol class="wi-steps">
<li><b>Open “Predict a reader”</b> in the menu at the top (under <i>Try it</i>).</li>
<li><b>Keep the first example</b> or pick another one. Each example is a short statement from a study report and a
conclusion that some people wrongly draw from it.</li>
<li><b>Choose who is reading.</b> Start with the <i>Teacher who reads study news</i>, then try the
<i>Methods lecturer</i> and compare.</li>
<li><b>Press the button.</b> You see what the reader's knowledge base says about this text, and whether the model
expects this reader to draw the conclusion.</li>
<li><b>Open “Rephrase for a reader”.</b> Pick the same example and reader. The app rewrites the statement so this
reader does not misread it.</li>
<li><b>Describe your own reader.</b> Choose <i>My own reader</i> and say, concept by concept, what they understand
or misunderstand. Or write your own statement.</li>
</ol>
""")

st.subheader("Words used in this app")
st.markdown("""
- **Statement**: what a report or a tutor says, for example “p = .03”.
- **Conclusion**: something a reader might infer from the statement. Some conclusions are correct, others are common misreadings.
- **Knowledge base**: a small, readable list of what one reader knows and believes. Because it is explicit, you can check and correct it.
- **Draws this conclusion / does not draw it / rejects it**: the three possible reactions of a reader (in NLP terms: entailment, neutral, contradiction).
- **Simulated reader**: a language model that is told what the reader knows and asked how they would understand a text.
""")

st.subheader("Good to know")
st.markdown(f"""
- Each visit allows {ui.MAX_CALLS} live model runs. Examples from the dataset usually load instantly from a cache.
- Your own text is sent to the OpenAI API to get a prediction. Do not enter personal information.
- The results are illustrations from a small pilot. Read **About the study** before drawing conclusions.
""")
