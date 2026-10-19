"""Personal knowledge bases (PKBs) as small knowledge graphs of (subject, relation, object) triples."""
import json
from pathlib import Path

import networkx as nx

DATA = Path(__file__).resolve().parent.parent / "data"


def load_concepts():
    return json.loads((DATA / "concepts.json").read_text())


def load_profiles():
    return json.loads((DATA / "pkb_profiles.json").read_text())


def load_items():
    return [json.loads(line) for line in (DATA / "items.jsonl").read_text().splitlines() if line.strip()]


def build_pkb(profile_name):
    """Build one user's PKB as a directed graph: user -> concept / misconception nodes."""
    concepts, profile = load_concepts(), load_profiles()[profile_name]
    by_mid = {c["misconception_id"]: cid for cid, c in concepts.items()}
    g = nx.MultiDiGraph()
    g.add_node("user", kind="user", description=profile["description"])
    for cid, c in concepts.items():
        g.add_node(cid, kind="concept", name=c["name"])
    for cid in profile["knows"]:
        g.add_node(f"def:{cid}", kind="statement", text=concepts[cid]["correct"])
        g.add_edge("user", cid, relation="understands")
        g.add_edge("user", f"def:{cid}", relation="knows_that")
        g.add_edge(f"def:{cid}", cid, relation="about")
    for mid in profile["believes"]:
        cid = by_mid[mid]
        g.add_node(mid, kind="statement", text=concepts[cid]["misconception"])
        g.add_edge("user", mid, relation="believes")
        g.add_edge(mid, cid, relation="about")
    return g


def triples(g):
    """All edges as readable triples; statement nodes are shown by their text."""
    def label(n):
        d = g.nodes[n]
        return d.get("text") or d.get("name") or n
    return [(label(u), d["relation"], label(v)) for u, v, d in g.edges(data=True)]


def link_concepts(text):
    """Very simple entity linking: which concepts does a text mention (keyword match)?"""
    t = text.lower()
    return [cid for cid, c in load_concepts().items() if any(k.lower() in t for k in c["keywords"])]


def retrieve(g, text):
    """Retrieve the PKB triples about the concepts mentioned in `text` (1 hop from each concept)."""
    hits = set(link_concepts(text))
    out = []
    for u, v, d in g.edges(data=True):
        target = v if g.nodes[v]["kind"] == "concept" else None
        if target is None:  # user -> statement edge: follow the statement to its concept
            target = next((w for _, w, dd in g.out_edges(v, data=True) if dd["relation"] == "about"), None)
        if target in hits and u == "user":
            out.append((u, d["relation"], v))
    def label(n):
        nd = g.nodes[n]
        return nd.get("text") or nd.get("name") or n
    return [(label(u), r, label(v)) for u, r, v in out]


def gold_label(item, profile_name):
    """The label a reader with this PKB would give: correct label unless they hold the item's misconception."""
    concepts, profile = load_concepts(), load_profiles()[profile_name]
    mid = concepts[item["concept"]]["misconception_id"]
    return item["label_misconceived"] if mid in profile["believes"] else item["label_correct"]
