"""Part 4: does a small LLM follow the user's PKB or its own parametric knowledge?

1. Behaviour: PKB-follow rate when the PKB contradicts world knowledge.
2. Activation patching: which layers / positions carry the PKB information.
3. Activation steering: a "use the PKB" direction added to the residual stream.

Model: Qwen2.5-0.5B-Instruct on CPU. Facts: country capitals with a counterfactual PKB entry.
"""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
torch.manual_seed(0)
torch.set_num_threads(6)
random.seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
LAYERS = model.model.layers


def prompt(country, kb_city=None, ask="user", kb_style="belief"):
    """kb_city=None: no PKB. ask="user": the answer is the user's view; ask="world": the answer is a plain fact.
    kb_style="belief": PKB records a belief of the user; "fact": PKB is presented as true context."""
    if kb_city is None:
        kb = ""
    elif kb_style == "belief":
        kb = f"Personal knowledge base of the user:\n- The user believes that the capital of {country} is {kb_city}.\n\n"
    else:
        kb = f"Context (the user's personal knowledge base):\nThe capital of {country} is {kb_city}.\n\n"
    if ask == "user":
        return f"{kb}Question: What does the user believe is the capital of {country}?\nAnswer: The user believes the capital of {country} is"
    return f"{kb}Question: What is the capital of {country}?\nAnswer: The capital of {country} is"


def first_id(word):
    return tok.encode(" " + word)[0]


@torch.no_grad()
def run(text, hook_layer=None, hook_fn=None, hidden=False):
    """Forward pass; optionally edit the residual stream output of one layer. Returns last-token logits and all hidden states."""
    ids = tok(text, return_tensors="pt").input_ids
    handle = None
    if hook_layer is not None:
        handle = LAYERS[hook_layer].register_forward_hook(lambda m, i, o: (hook_fn(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else hook_fn(o))
    try:
        out = model(ids, output_hidden_states=hidden, logits_to_keep=1)
    finally:
        if handle:
            handle.remove()
    return out.logits[0, -1], (out.hidden_states if hidden else None), ids


def logit_diff(logits, f):
    """PKB answer minus parametric answer (positive = follows the PKB)."""
    return (logits[first_id(f["pkb"])] - logits[first_id(f["true"])]).item()


def behaviour(facts):
    """Share of items where the model's answer is the PKB city, per condition.
    Desired: high when asked for the user's belief, low when asked about the world (a belief is not a fact)."""
    conds = {  # name: (PKB present, ask, kb_style)
        "no_pkb": (False, "world", "belief"),
        "belief_ask_user": (True, "user", "belief"),
        "belief_ask_world": (True, "world", "belief"),
        "fact_ask_world": (True, "world", "fact"),
    }
    rows = []
    for f in facts:
        r = {"country": f["country"]}
        for name, (has_kb, ask, style) in conds.items():
            text = prompt(f["country"], f["pkb"] if has_kb else None, ask, style)
            r[name] = logit_diff(run(text)[0], f)
        rows.append(r)
    summary = {k: sum(r[k] > 0 for r in rows) / len(rows) for k in conds}
    return rows, summary


def patching(facts):
    """Clean = PKB says the counterfactual city; corrupt = PKB says the true capital (same length).
    Patch the clean residual stream into the corrupt run, one layer and position at a time.
    Recovery 1.0 = patching that activation alone restores the PKB answer."""
    n_layers = len(LAYERS)
    res = {"final": [0.0] * n_layers, "kb_token": [0.0] * n_layers}
    used = 0
    for f in facts:
        clean_txt = prompt(f["country"], f["pkb"], "user")
        corrupt_txt = prompt(f["country"], f["true"], "user")
        c_logits, c_hs, c_ids = run(clean_txt, hidden=True)
        x_logits, _, x_ids = run(corrupt_txt)
        if c_ids.shape != x_ids.shape:
            continue
        diff_pos = (c_ids[0] != x_ids[0]).nonzero().flatten().tolist()
        kb_pos, last = diff_pos[0], c_ids.shape[1] - 1
        clean_d, corrupt_d = logit_diff(c_logits, f), logit_diff(x_logits, f)
        if clean_d - corrupt_d < 1.0:
            continue
        used += 1
        for layer in range(n_layers):
            for name, pos in [("final", last), ("kb_token", kb_pos)]:
                src = c_hs[layer + 1][0, pos].clone()

                def fn(h, src=src, pos=pos):
                    h[0, pos] = src
                    return h
                p_logits = run(corrupt_txt, layer, fn)[0]
                res[name][layer] += (logit_diff(p_logits, f) - corrupt_d) / (clean_d - corrupt_d)
    return {k: [v / max(used, 1) for v in vals] for k, vals in res.items()}, used


def steering(facts, layer, alphas=(0, 4, 8, 16)):
    """Perspective direction = mean(final-token residual | ask user) - mean(... | ask world), same PKB, train half.
    Added to "ask world" prompts of the test half: does the answer switch to the user's belief?
    Control: added to prompts without any PKB (the answer should stay the true capital)."""
    random.shuffle(facts)
    train, test = facts[: len(facts) // 2], facts[len(facts) // 2:]
    vecs = []
    for f in train:
        hu = run(prompt(f["country"], f["pkb"], "user"), hidden=True)[1][layer + 1][0, -1]
        hw = run(prompt(f["country"], f["pkb"], "world"), hidden=True)[1][layer + 1][0, -1]
        vecs.append(hu - hw)
    v = torch.stack(vecs).mean(0)
    v = v / v.norm()
    out = []
    for a in alphas:
        def fn(h, a=a):
            h[0, -1] += a * v
            return h
        follow = sum(logit_diff(run(prompt(f["country"], f["pkb"], "world"), layer, fn)[0], f) > 0 for f in test) / len(test)
        control = sum(logit_diff(run(prompt(f["country"], None, "world"), layer, fn)[0], f) < 0 for f in test) / len(test)
        out.append({"alpha": a, "answer_is_user_belief": follow, "no_pkb_still_true_capital": control})
    return out, len(test)


if __name__ == "__main__":
    facts = json.loads((ROOT / "data" / "conflicts.json").read_text())
    results = {}
    rows, summary = behaviour(facts)
    results["behaviour"] = {"summary": summary, "rows": rows}
    print("behaviour", summary, flush=True)
    patch, used = patching(facts[:24])
    results["patching"] = {"recovery": patch, "n": used}
    print("patching n =", used, flush=True)
    best = max(range(len(LAYERS)), key=lambda i: patch["final"][i] - (patch["final"][i - 1] if i else 0))
    results["steering"] = {}
    for layer in sorted({best, len(LAYERS) // 2}):
        st, n = steering(list(facts), layer)
        results["steering"][str(layer)] = {"n_test": n, "results": st}
        print("steering layer", layer, st, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mech.json").write_text(json.dumps(results, indent=1))
