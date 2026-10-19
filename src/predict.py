"""Part 2: predict how a specific reader interprets a text, given (or not given) their PKB.

Conditions: no PKB / full PKB as text / PKB triples retrieved from the knowledge graph.
Plus counterfactual PKB edits: toggle one concept in the 'mixed' profile and check that
only the predictions for that concept change (sensitivity) and the others stay (specificity).
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkb
from llm import LABELS, label_distribution

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = ("You model how a specific reader interprets statistical statements. "
          "Given a premise and a hypothesis, predict whether THIS reader would conclude the hypothesis from the premise. "
          "entailment = the reader would conclude it; contradiction = the reader would conclude it is false; "
          "neutral = the reader would conclude neither. Base the prediction on the reader's knowledge and beliefs, "
          "even when they are wrong. Answer with one word: entailment, neutral or contradiction.")


def reader_block(condition, g, item):
    if condition == "no_pkb":
        return "Reader: a typical adult reader."
    if condition == "pkb_text":
        lines = [f"- {s} {r} {o}" for s, r, o in pkb.triples(g) if s == "user"]
        return "Reader's personal knowledge base (all entries):\n" + "\n".join(lines)
    hits = pkb.retrieve(g, item["premise"] + " " + item["hypothesis"])
    lines = [f"- {s} {r} {o}" for s, r, o in hits] or ["- (no entries about the concepts in this text)"]
    return "Reader's personal knowledge base (entries retrieved for this text):\n" + "\n".join(lines)


def predict(item, g, condition, model="gpt-4o-mini"):
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": f"{reader_block(condition, g, item)}\n\nPremise: {item['premise']}\nHypothesis: {item['hypothesis']}\nPrediction:"}]
    dist = label_distribution(msg, model=model)
    return max(dist, key=dist.get), dist


def gold_for(item, profile):
    concepts = pkb.load_concepts()
    mid = concepts[item["concept"]]["misconception_id"]
    return item["label_misconceived"] if mid in profile["believes"] else item["label_correct"]


def build_from_profile(profile):
    """Build a PKB graph from an in-memory profile dict (used for counterfactual edits)."""
    profiles = pkb.load_profiles()
    profiles["_tmp"] = profile
    orig = pkb.load_profiles
    pkb.load_profiles = lambda: profiles
    try:
        return pkb.build_pkb("_tmp")
    finally:
        pkb.load_profiles = orig


def main():
    items, profiles, concepts = pkb.load_items(), pkb.load_profiles(), pkb.load_concepts()
    rows = []
    for pname, prof in profiles.items():
        g = pkb.build_pkb(pname)
        for it in items:
            gold = gold_for(it, prof)
            for cond in ["no_pkb", "pkb_text", "pkb_retrieved"]:
                pred, dist = predict(it, g, cond)
                rows.append({"profile": pname, "item": it["id"], "type": it["type"], "concept": it["concept"],
                             "condition": cond, "gold": gold, "pred": pred, "p_gold": dist[gold]})
        print("done", pname, flush=True)

    def acc(rs):
        return round(sum(r["pred"] == r["gold"] for r in rs) / len(rs), 3) if rs else None

    def acc_bin(rs):
        """Does the reader draw the inference (entailment) or not? The question that matters for misunderstandings."""
        return round(sum((r["pred"] == "entailment") == (r["gold"] == "entailment") for r in rs) / len(rs), 3) if rs else None

    summary = {}
    for cond in ["no_pkb", "pkb_text", "pkb_retrieved"]:
        rs = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"all": acc(rs), "critical": acc([r for r in rs if r["type"] == "critical"]),
                         "control": acc([r for r in rs if r["type"] == "control"]),
                         "critical_binary": acc_bin([r for r in rs if r["type"] == "critical"]),
                         **{p: acc([r for r in rs if r["profile"] == p and r["type"] == "critical"]) for p in profiles}}

    # Counterfactual edits on the mixed profile: toggle each concept between "knows" and "believes misconception".
    base = profiles["mixed"]
    g0 = pkb.build_pkb("mixed")
    base_pred = {it["id"]: predict(it, g0, "pkb_retrieved")[0] for it in items}
    edits = []
    for cid, c in concepts.items():
        prof = copy.deepcopy(base)
        if c["misconception_id"] in prof["believes"]:
            prof["believes"].remove(c["misconception_id"]); prof["knows"].append(cid)
        else:
            prof["knows"] = [k for k in prof["knows"] if k != cid]; prof["believes"].append(c["misconception_id"])
        g = build_from_profile(prof)
        for it in items:
            pred = predict(it, g, "pkb_retrieved")[0]
            gold = gold_for(it, prof)
            edits.append({"edited": cid, "item": it["id"], "affected": it["concept"] == cid and it["type"] == "critical",
                          "pred": pred, "gold": gold, "before": base_pred[it["id"]]})
    aff = [e for e in edits if e["affected"]]
    unaff = [e for e in edits if not e["affected"]]
    summary["counterfactual"] = {
        "sensitivity_correct_after_edit": round(sum(e["pred"] == e["gold"] for e in aff) / len(aff), 3),
        "specificity_unchanged_elsewhere": round(sum(e["pred"] == e["before"] for e in unaff) / len(unaff), 3),
        "n_affected": len(aff), "n_unaffected": len(unaff)}
    # Model-size check: the same retrieved-PKB condition with gpt-4o on the critical items.
    big = []
    for pname, prof in profiles.items():
        g = pkb.build_pkb(pname)
        for it in items:
            if it["type"] != "critical":
                continue
            pred, dist = predict(it, g, "pkb_retrieved", model="gpt-4o")
            big.append({"profile": pname, "item": it["id"], "gold": gold_for(it, prof), "pred": pred})
    summary["pkb_retrieved_gpt4o"] = {"critical": acc(big), "critical_binary": acc_bin(big),
                                      **{p: acc([r for r in big if r["profile"] == p]) for p in profiles}}
    out = {"summary": summary, "rows": rows, "edits": edits, "rows_gpt4o": big}
    (ROOT / "results" / "predict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
