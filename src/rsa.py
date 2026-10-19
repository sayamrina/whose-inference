"""Part 3: cooperative generation in the spirit of the Rational Speech Acts (RSA) framework.

Goal: convey a statistical result to a specific reader WITHOUT triggering a known misinterpretation.
For each critical item (premise = what the tutor wants to say, hypothesis = a misreading):

  literal   : the premise as is
  plain     : one rewrite by the LLM with a generic "explain clearly" instruction (no listener model)
  single    : the speaker's first candidate (knows the misreading, but no listener simulation)
  rsa       : speaker samples K candidates; a simulated listener (Part 2 predictor + the reader's
              retrieved PKB) scores each by P(reader does NOT draw the misreading); the speaker picks
              the best utility = log P(correct) - lambda * length. If the best candidate is still weak,
              the gaps (misconceptions in the reader's PKB relevant to the message) are made explicit
              and the loop runs again (max 2 rounds).

Evaluation uses a DIFFERENT listener model (gpt-4o) to reduce self-evaluation bias.
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkb
from llm import chat, label_distribution
from predict import SYSTEM, gold_for

ROOT = Path(__file__).resolve().parent.parent
K, LAMBDA, THRESHOLD = 5, 0.6, 0.8
SPEAKER = "gpt-4o-mini"
SIM_LISTENER = os.environ.get("SIM_LISTENER", "gpt-4o-mini")
EVAL_LISTENER = os.environ.get("EVAL_LISTENER", "gpt-4o")


def listener(message, item, g, model="gpt-4o-mini"):
    hits = pkb.retrieve(g, item["premise"] + " " + item["hypothesis"])
    kb = "\n".join(f"- {s} {r} {o}" for s, r, o in hits) or "- (no entries)"
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": f"Reader's personal knowledge base (entries retrieved for this text):\n{kb}\n\n"
                                        f"Premise: {message}\nHypothesis: {item['hypothesis']}\nPrediction:"}]
    return label_distribution(msg, model=model)


def p_correct(dist, intended):
    """Intended = the label an informed reader gives; a misreading is any move toward 'entailment' when that is wrong."""
    if intended == "entailment":
        return dist["entailment"]
    return 1 - dist["entailment"]


def speaker(item, n, gaps=None, seed=0):
    extra = ""
    if gaps:
        extra = ("\nThis reader holds these beliefs, which could cause a misunderstanding. Make the needed background "
                 "explicit, briefly and kindly, without lecturing:\n" + "\n".join(f"- {x}" for x in gaps))
    msg = [{"role": "system", "content": "You are a tutor rewriting one sentence for a specific student. Keep every fact of the "
                                         "original, add at most two short sentences, plain English. Return only the rewritten text."},
           {"role": "user", "content": f"Original: {item['premise']}\n"
                                        f"Avoid this misreading: \"{item['hypothesis']}\"{extra}\nRewrite (variant {seed}):"}]
    return chat(msg, model=SPEAKER, temperature=1.0, max_tokens=120, n=n)["texts"]


def utility(p, text):
    return math.log(max(p, 1e-6)) - LAMBDA * len(text.split()) / 50


def rsa(item, g, intended):
    log = []
    gaps = None
    best = None
    for rnd in range(2):
        cands = speaker(item, K, gaps, seed=rnd)
        for c in cands:
            p = p_correct(listener(c, item, g, model=SIM_LISTENER), intended)
            log.append({"round": rnd, "text": c, "p_sim": round(p, 3), "u": round(utility(p, c), 3)})
            if best is None or utility(p, c) > utility(best[1], best[0]):
                best = (c, p)
        if best[1] >= THRESHOLD:
            break
        gaps = [o for s, r, o in pkb.retrieve(g, item["premise"] + " " + item["hypothesis"]) if r == "believes"]
    return best[0], log


def plain(item):
    msg = [{"role": "system", "content": "Rewrite the sentence so it is clear for a student. Keep every fact. Return only the text."},
           {"role": "user", "content": item["premise"]}]
    return chat(msg, model=SPEAKER, max_tokens=120)["texts"][0]


def main():
    items = [it for it in pkb.load_items() if it["type"] == "critical"]
    profiles = pkb.load_profiles()
    concepts = pkb.load_concepts()
    rows = []
    for pname in ["misconceived", "mixed"]:
        g, prof = pkb.build_pkb(pname), profiles[pname]
        for it in items:
            if concepts[it["concept"]]["misconception_id"] not in prof["believes"]:
                continue  # only messages where this reader is at risk of misreading
            intended = it["label_correct"]
            texts = {"literal": it["premise"], "plain": plain(it)}
            texts["rsa"], trace = rsa(it, g, intended)
            texts["single"] = trace[0]["text"]  # same speaker prompt, no listener simulation / selection
            r = {"profile": pname, "item": it["id"], "intended": intended, "trace": trace}
            for k, t in texts.items():
                r[k] = {"text": t, "words": len(t.split()),
                        "p_eval": round(p_correct(listener(t, it, g, model=EVAL_LISTENER), intended), 3)}
            rows.append(r)
        print("done", pname, flush=True)
    summary = {}
    for k in ["literal", "plain", "single", "rsa"]:
        summary[k] = {"mean_p_correct_eval": round(sum(r[k]["p_eval"] for r in rows) / len(rows), 3),
                      "share_understood_eval": round(sum(r[k]["p_eval"] > 0.5 for r in rows) / len(rows), 3),
                      "mean_words": round(sum(r[k]["words"] for r in rows) / len(rows), 1)}
    summary["n"] = len(rows)
    summary["sim_listener"], summary["eval_listener"] = SIM_LISTENER, EVAL_LISTENER
    (ROOT / "results" / f"rsa_{SIM_LISTENER}_eval_{EVAL_LISTENER}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
