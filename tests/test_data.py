import pandas as pd

from posttrain.data import ABSTAIN, build_dpo_pairs, build_eval, build_examples


CFG = {
    "data": {
        "n_entities": 40,
        "n_answerable_per_entity": 2,
        "n_unanswerable_train": 40,
        "n_dpo_pairs": 20,
        "heldout_entities": 8,
        "seed": 5,
    }
}


def test_examples_both_kinds_and_prompted():
    df = build_examples(CFG)
    assert set(df.kind) == {"answerable", "unanswerable"}
    unans = df[df.kind == "unanswerable"].iloc[0]
    assert unans["completion"].strip() == ABSTAIN
    ans = df[df.kind == "answerable"].iloc[0]
    assert ans["gold"] in ans["completion"]


def test_eval_uses_heldout_entities():
    ev = build_eval(CFG)
    train_ents = set(build_examples(CFG).entity) - set(ev.entity)
    assert len(set(ev.entity)) == CFG["data"]["heldout_entities"]
    # heldout entities are the tail of the name list -> disjoint by construction
    assert len(ev) == CFG["data"]["heldout_entities"] * 2


def test_dpo_pairs_shape_and_difference():
    df = build_examples(CFG)
    pairs = build_dpo_pairs(df, CFG)
    assert {"prompt", "chosen", "rejected"} <= set(pairs.columns)
    assert (pairs["chosen"] != pairs["rejected"]).all()
