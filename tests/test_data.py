"""Dataset and knowledge-base checks. No API key or network needed."""
from collections import Counter

import pkb

LABELS = {"entailment", "neutral", "contradiction"}


def test_items_are_complete():
    items = pkb.load_items()
    concepts = pkb.load_concepts()
    assert len(items) == 24
    per_concept = Counter((i["concept"], i["type"]) for i in items)
    for cid in concepts:
        assert per_concept[(cid, "critical")] == 2
        assert per_concept[(cid, "control")] == 2
    for i in items:
        assert {i["label_correct"], i["label_misconceived"]} <= LABELS
        assert i["premise"].strip() and i["hypothesis"].strip()


def test_critical_items_depend_on_the_reader():
    for i in pkb.load_items():
        same = i["label_correct"] == i["label_misconceived"]
        assert same == (i["type"] == "control"), i["id"]


def test_profiles_refer_to_known_concepts():
    concepts = pkb.load_concepts()
    mids = {c["misconception_id"] for c in concepts.values()}
    for p in pkb.load_profiles().values():
        assert set(p["knows"]) <= set(concepts)
        assert set(p["believes"]) <= mids


def test_retrieval_returns_only_the_relevant_concept():
    g = pkb.build_pkb("mixed")
    text = "The difference was statistically significant (p = .01)."
    hits = pkb.retrieve(g, text)
    assert hits, "the mixed reader holds the significance misconception"
    assert any(r == "believes" for _, r, _ in hits)
    assert not pkb.retrieve(g, "The weather was nice.")


def test_gold_labels_follow_the_knowledge_base():
    s1 = next(i for i in pkb.load_items() if i["id"] == "s1")
    assert pkb.gold_label(s1, "expert") == s1["label_correct"]
    assert pkb.gold_label(s1, "misconceived") == s1["label_misconceived"]
    assert pkb.gold_label(s1, "mixed") == s1["label_misconceived"]  # mixed believes M2
