"""Retrieval-augmented evaluation: replace the eval prompt's baked
Context line with documents retrieved from a fact corpus at inference
time.

Corpus: one document per unique fact sentence (entity -> attribute), built
from the committed sft/eval parquets. Retrieval is BM25 over word tokens —
no external index; the corpus is a few hundred docs.

The eval contrast this creates: for `answerable` rows the corpus contains
the needed fact, so retrieval tests whether the model can *use* grounded
context. For `unanswerable` rows the corpus contains the queried entity's
*other* attributes but not the asked one — a harder abstention test than
the original eval (which supplies an unrelated entity's fact), because a
weak abstainer sees topically related context and has more pressure to
fabricate.
"""

import math
import re
from collections import Counter

import pandas as pd

TOKEN_RE = re.compile(r"\w+")
CTX_RE = re.compile(r"Context: (.*?)\nQuestion:", re.S)


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Minimal BM25 (k1=1.5, b=0.75) for a few-hundred-doc corpus."""

    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d["text"]) for d in docs]
        self.docs = docs
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}
        self.avgdl = (sum(len(t) for t in self.doc_tokens) / n) if n else 1.0

    def score(self, query_tokens: list, i: int) -> float:
        toks = self.doc_tokens[i]
        tf = Counter(toks)
        dl = len(toks) or 1
        s = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def retrieve(self, query: str, k: int = 3) -> list:
        qt = tokenize(query)
        scored = sorted(((self.score(qt, i), i) for i in range(len(self.docs))),
                        key=lambda x: -x[0])
        return [self.docs[i] for s, i in scored[:k] if s > 0]


def build_corpus(sft_parquet: str, eval_parquet: str) -> list:
    """Unique fact sentences from the committed splits.

    Doc = {"entity": e, "text": fact}. eval rows contribute their `gold`
    sentence for answerable items (the fact a deployed KB would hold).
    unanswerable rows' gold is the abstain phrase, which is not a fact and
    is skipped.
    """
    docs, seen = [], set()

    def add(entity: str, text: str):
        key = (entity, text.strip())
        if key not in seen:
            seen.add(key)
            docs.append({"entity": entity, "text": text.strip()})

    sft = pd.read_parquet(sft_parquet)
    for _, r in sft.iterrows():
        if str(r["gold"]).strip():
            add(r["entity"], r["gold"])
    ev = pd.read_parquet(eval_parquet)
    for _, r in ev[ev.kind == "answerable"].iterrows():
        add(r["entity"], r["gold"])
    return docs


def rewrite_prompt(prompt: str, docs: list) -> str:
    """Swap the prompt's Context line for the retrieved documents."""
    ctx = " ".join(d["text"] for d in docs) or "(no documents retrieved)"
    return CTX_RE.sub(f"Context: {ctx}\nQuestion:", prompt, count=1)


_Q_STOP = {"what", "is", "the", "of", "how", "primarily", "by", "a",
           "an", "does", "do", "much", "many"}


def _doc_answers(doc_text: str, entity: str, question: str) -> bool:
    """Whether a retrieved doc states the attribute the question asks for
    the queried entity: same entity mention and the question's content
    tokens all present in the doc."""
    if entity not in doc_text:
        return False
    entity_toks = set(tokenize(entity))
    q_toks = set(tokenize(question)) - entity_toks - _Q_STOP
    return bool(q_toks) and q_toks <= set(tokenize(doc_text))


def build_sft_rag_dataframe(train_df: pd.DataFrame, corpus: list,
                            k: int) -> pd.DataFrame:
    """Rebuild an SFT frame with each prompt's baked Context line replaced
    by k BM25-retrieved docs, and completions made consistent with what
    retrieval supplies: "answer iff the retrieved context answers".

    - answerable row whose gold fact was retrieved: keep the gold
      completion
    - answerable row whose gold fact was missed: flip to abstain (the
      context does not support the answer)
    - unanswerable row where retrieval surfaced a same-entity doc that
      answers the question: rescue the completion to that doc's fact
    - otherwise keep the abstain completion

    Callers should pass a train-only corpus so no held-out entity fact
    leaks into a training context."""
    from posttrain.data import ABSTAIN

    index = BM25(corpus)
    out = train_df.copy()
    prompts, completions, n_retrieved, flipped = [], [], [], []
    for _, r in out.iterrows():
        q = r["prompt"].split("Question:")[-1].replace("Answer:", "").strip()
        docs = index.retrieve(f'{r["entity"]} {q}', k)
        prompts.append(rewrite_prompt(r["prompt"], docs))
        n_retrieved.append(len(docs))
        completion = r["completion"]
        if r["kind"] == "answerable":
            hit = any(d["text"] == str(r["gold"]).strip() for d in docs)
            flipped.append(not hit)
            if not hit:
                completion = " " + ABSTAIN
        else:
            hit_doc = next((d for d in docs
                            if _doc_answers(d["text"], r["entity"], q)),
                           None)
            flipped.append(hit_doc is not None)
            if hit_doc is not None:
                completion = " " + hit_doc["text"]
        completions.append(completion)
    out["prompt"] = prompts
    out["completion"] = completions
    out["n_retrieved"] = n_retrieved
    out["completion_flipped"] = flipped
    return out.reset_index(drop=True)


def retrieval_hit(docs: list, gold_value: str) -> bool:
    """Whether any retrieved doc states the gold value for the question."""
    return any(re.search(rf"\b{re.escape(str(gold_value))}\b", d["text"])
               for d in docs)


def eval_rag(model, tok, ev: pd.DataFrame, corpus: list, k: int):
    """Same greedy decode + classify as eval_stage, but the context comes
    from retrieval. Returns (per-kind metrics, response frame)."""
    from posttrain.evaluate import classify, generate

    index = BM25(corpus)
    rows = []
    for _, r in ev.iterrows():
        q = r["prompt"].split("Question:")[-1].replace("Answer:", "").strip()
        docs = index.retrieve(f'{r["entity"]} {q}', k)
        prompt = rewrite_prompt(r["prompt"], docs)
        resp = generate(model, tok, prompt)
        rows.append({
            "kind": r["kind"],
            "label": classify(resp, r["entity"], r["gold_value"], r["kind"]),
            "response": resp[:120],
            "retrieved": " || ".join(d["text"][:80] for d in docs),
            "hit": retrieval_hit(docs, r["gold_value"]),
        })
    rdf = pd.DataFrame(rows)
    out = {}
    for kind in ("answerable", "unanswerable"):
        g = rdf[rdf.kind == kind]
        out[kind] = {
            "n": int(len(g)),
            "correct": float((g.label == "correct").mean()),
            "abstains": float((g.label == "abstains").mean()),
            "degenerate": float((g.label == "degenerate").mean()),
            "fabricates": float((g.label == "fabricates").mean()),
        }
    ans = rdf[rdf.kind == "answerable"]
    out["retrieval"] = {
        "k": k, "corpus_docs": len(corpus),
        "hit_rate_answerable": float(ans.hit.mean()) if len(ans) else None,
    }
    return out, rdf
