import pandas as pd
import pytest

from posttrain.rag import (BM25, build_corpus, build_sft_rag_dataframe,
                           rewrite_prompt, retrieval_hit)


@pytest.fixture
def corpus(tmp_path):
    sft = pd.DataFrame([
        {"entity": "capx-1", "kind": "answerable",
         "prompt": "p", "completion": " capx-1 is cleared by the kidney.",
         "gold": "capx-1 is cleared by the kidney."},
        {"entity": "capx-1", "kind": "answerable",
         "prompt": "p", "completion": " capx-1 has a 62 hour half-life.",
         "gold": "capx-1 has a 62 hour half-life."},
        {"entity": "zorb-2", "kind": "answerable",
         "prompt": "p", "completion": " zorb-2 is cleared by the liver.",
         "gold": "zorb-2 is cleared by the liver."},
        # duplicate gold dedupes
        {"entity": "capx-1", "kind": "answerable",
         "prompt": "p", "completion": " x",
         "gold": "capx-1 is cleared by the kidney."},
    ])
    ev = pd.DataFrame([
        {"entity": "capx-1", "kind": "answerable",
         "prompt": ('Say exactly "abstain" if missing.\n\n'
                    "Context: x\nQuestion: How is capx-1 cleared?\nAnswer:"),
         "gold": "capx-1 is cleared by the kidney.", "gold_value": "kidney"},
        {"entity": "zorb-2", "kind": "unanswerable",
         "prompt": ('Say exactly "abstain" if missing.\n\n'
                    "Context: y\nQuestion: What is zorb-2 half-life?\nAnswer:"),
         "gold": "I don't have enough information to answer.",
         "gold_value": "99"},
    ])
    sft.to_parquet(tmp_path / "sft.parquet")
    ev.to_parquet(tmp_path / "eval.parquet")
    return build_corpus(str(tmp_path / "sft.parquet"),
                        str(tmp_path / "eval.parquet"))


def test_corpus_dedupes_and_skips_abstain_gold(corpus):
    texts = [d["text"] for d in corpus]
    assert len(texts) == len(set(texts)) == 3
    assert all("enough information" not in t for t in texts)


def test_bm25_ranks_matching_doc_first(corpus):
    idx = BM25(corpus)
    top = idx.retrieve("capx-1 cleared", k=1)
    assert len(top) == 1 and "kidney" in top[0]["text"]
    # entity token alone should surface that entity's docs
    tops = idx.retrieve("zorb-2", k=2)
    assert all(d["entity"] == "zorb-2" for d in tops)


def test_bm25_zero_score_returns_nothing(corpus):
    assert BM25(corpus).retrieve("qqq nonexistent", k=3) == []


def test_rewrite_swaps_context_line():
    p = ('instr\n\nContext: OLD FACT\nQuestion: q?\nAnswer:')
    out = rewrite_prompt(p, [{"entity": "e", "text": "NEW FACT"}])
    assert "NEW FACT" in out and "OLD FACT" not in out
    assert "Question: q?" in out


def test_rewrite_empty_docs_notes_absence():
    p = 'Context: OLD\nQuestion: q?\nAnswer:'
    assert "(no documents retrieved)" in rewrite_prompt(p, [])


def test_retrieval_hit_matches_value(corpus):
    assert retrieval_hit([{"text": "x is cleared by the kidney.", "entity": "x"}],
                         "kidney")
    assert not retrieval_hit([{"text": "x is cleared by the liver.",
                              "entity": "x"}], "kidney")
    # word-boundary: '62' must not match '620'
    assert not retrieval_hit([{"text": "x has 620 units.", "entity": "x"}], "62")


def _prompt(ctx, q):
    return f"instr\n\nContext: {ctx}\nQuestion: {q}\nAnswer:"


def test_build_sft_rag_rewrites_context_keeps_completion():
    import pandas as pd
    train = pd.DataFrame([
        {"entity": "capx-1", "kind": "answerable",
         "prompt": _prompt("The half-life of capx-1 is 42 hours.",
                          "What is the half-life of capx-1?"),
         "completion": " The half-life of capx-1 is 42 hours.",
         "gold": "The half-life of capx-1 is 42 hours."},
        {"entity": "capx-1", "kind": "unanswerable",
         "prompt": _prompt("The half-life of zorb-2 is 9 hours.",
                          "What is the molecular weight of capx-1?"),
         "completion": " I don't have enough information to answer.",
         "gold": "I don't have enough information to answer."},
    ])
    corpus = [
        {"entity": "capx-1", "text": "The half-life of capx-1 is 42 hours."},
        {"entity": "capx-1", "text": "capx-1 is primarily cleared by the liver."},
        {"entity": "zorb-2", "text": "The half-life of zorb-2 is 9 hours."},
    ]
    out = build_sft_rag_dataframe(train, corpus, k=3)
    ans = out[out.kind == "answerable"].iloc[0]
    # answerable: retrieval surfaces the needed fact, gold kept
    assert "42 hours" in ans["prompt"]
    assert ans["completion"] == " The half-life of capx-1 is 42 hours."
    unans = out[out.kind == "unanswerable"].iloc[0]
    # unanswerable MW question: corpus lacks capx-1 MW, abstain kept
    ctx = unans["prompt"].split("Context:")[1].split("Question:")[0]
    assert "molecular weight of capx-1" not in ctx
    assert unans["completion"].strip().startswith("I don't have enough")
    assert "n_retrieved" in out.columns


def test_build_sft_rag_unanswerable_retrieves_same_entity_distractor():
    import pandas as pd
    train = pd.DataFrame([
        {"entity": "capx-1", "kind": "unanswerable",
         "prompt": _prompt("Filler.",
                          "What is the molecular weight of capx-1?"),
         "completion": " I don't have enough information to answer.",
         "gold": "x"},
    ])
    corpus = [
        {"entity": "capx-1", "text": "The half-life of capx-1 is 42 hours."},
        {"entity": "capx-1", "text": "capx-1 is primarily cleared by the liver."},
        {"entity": "zorb-2", "text": "The half-life of zorb-2 is 9 hours."},
    ]
    out = build_sft_rag_dataframe(train, corpus, k=2)
    ctx = out.iloc[0]["prompt"].split("Context:")[1].split("Question:")[0]
    # the hard case: same-entity related facts, not the asked attribute
    assert "capx-1" in ctx and "300" not in ctx


def test_build_sft_rag_flips_on_miss_and_rescues_on_hit():
    train = pd.DataFrame([
        {"entity": "capx-1", "kind": "answerable",
         "prompt": _prompt("x", "What is the half-life of capx-1?"),
         "completion": " The half-life of capx-1 is 42 hours.",
         "gold": "The half-life of capx-1 is 42 hours."},
        {"entity": "zorb-2", "kind": "unanswerable",
         "prompt": _prompt("x", "What is the half-life of zorb-2?"),
         "completion": " I don't have enough information to answer.",
         "gold": "x"},
    ])
    # corpus has zorb-2's half-life but not capx-1's
    corpus = [
        {"entity": "zorb-2", "text": "The half-life of zorb-2 is 9 hours."},
        {"entity": "capx-1", "text": "capx-1 is primarily cleared by the liver."},
    ]
    out = build_sft_rag_dataframe(train, corpus, k=3)
    ans, unans = out.iloc[0], out.iloc[1]
    # retrieval missed the gold -> answerable completion flips to abstain
    assert ans["completion_flipped"]
    assert "enough information" in ans["completion"]
    # retrieval surfaced zorb-2's asked attribute -> rescued to the fact
    assert unans["completion_flipped"]
    assert "9 hours" in unans["completion"]
