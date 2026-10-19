"""Small OpenAI wrapper with an on-disk cache, so reruns cost nothing and results are reproducible."""
import hashlib
import json
import math
import os
from pathlib import Path

from openai import OpenAI

CACHE = Path(__file__).resolve().parent.parent / "results" / "llm_cache.jsonl"
LABELS = ["entailment", "neutral", "contradiction"]
_client = None
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                d = json.loads(line)
                _cache[d["key"]] = d["value"]
    return _cache


def chat(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=400, logprobs=False, n=1):
    global _client
    key = hashlib.sha256(json.dumps([messages, model, temperature, max_tokens, logprobs, n]).encode()).hexdigest()
    cache = _load()
    if key in cache:
        return cache[key]
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kw = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n)
    if logprobs:
        kw.update(logprobs=True, top_logprobs=10)
    r = _client.chat.completions.create(**kw)
    value = {"texts": [c.message.content for c in r.choices]}
    if logprobs:
        first = r.choices[0].logprobs.content[0].top_logprobs
        value["top"] = [[t.token, t.logprob] for t in first]
    cache[key] = value
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")
    return value


def label_distribution(messages, model="gpt-4o-mini"):
    """Ask for a one-word label and read P(label) from the first-token logprobs (renormalised over the 3 labels)."""
    v = chat(messages, model=model, max_tokens=3, logprobs=True)
    p = {lab: 0.0 for lab in LABELS}
    for tok, lp in v["top"]:
        t = tok.strip().lower()
        for lab in LABELS:
            if t and lab.startswith(t) and len(t) >= 3:
                p[lab] += math.exp(lp)
    z = sum(p.values())
    if z == 0:  # fall back to the generated word
        word = v["texts"][0].strip().lower()
        return {lab: float(word.startswith(lab[:4])) for lab in LABELS}
    return {lab: x / z for lab, x in p.items()}
