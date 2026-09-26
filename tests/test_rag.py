import pandas as pd
import pytest

from posttrain.rag import BM25, build_corpus, rewrite_prompt, retrieval_hit


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
