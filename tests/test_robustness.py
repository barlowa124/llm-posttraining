"""Robustness battery: classifier boundary cases, reward partial-credit
traps, advantage normalization, masking, and data-split disjointness."""

import numpy as np
import pandas as pd
import pytest
import torch

from posttrain.data import (ABSTAIN, build_dpo_pairs, build_eval,
                            build_examples, entity_names)
from posttrain.dpo import dpo_loss
from posttrain.evaluate import _is_degenerate, classify
from posttrain.grpo import group_advantages, reward
from posttrain.lm import collate, encode_pair


class ToyTok:
    pad_token_id = 0

    def __call__(self, s, add_special_tokens=True, **kw):
        return {"input_ids": [ord(c) % 50 + 1 for c in s]}


class TestClassifyEdges:
    def test_value_substring_must_not_match(self):
        # gold "62" must not pass as correct inside "620"
        assert classify("The half-life of foo-1 is 620 hours.",
                        "foo-1", "62", "answerable") != "correct"

    def test_value_without_entity_is_not_correct(self):
        assert classify("The answer is 42.", "foo-1", "42",
                        "answerable") != "correct"

    def test_entity_without_value_is_not_correct(self):
        assert classify("foo-1 is cleared by the liver.",
                        "foo-1", "kidney", "answerable") != "correct"

    def test_abstain_prefix_with_trailing_text(self):
        r = ABSTAIN + " The half-life is 42 hours."
        assert classify(r, "foo-1", "42", "unanswerable") == "abstains"

    def test_mid_sentence_abstain_is_not_abstain(self):
        r = "The half-life is 42. " + ABSTAIN
        assert classify(r, "foo-1", "42", "unanswerable") != "abstains"

    def test_empty_and_whitespace_responses(self):
        for r in ("", "   ", "\n"):
            assert classify(r, "e", "1", "answerable") == "fabricates"

    def test_degenerate_detects_repetition(self):
        assert _is_degenerate("the the the the the the")
        assert _is_degenerate("ab cd " * 5)
        assert not _is_degenerate("short")
        assert not _is_degenerate("The half-life of foo-1 is 42 hours.")

    def test_degenerate_checked_before_fabricate(self):
        r = "zz " * 10
        assert classify(r, "foo-1", "42", "unanswerable") == "degenerate"


class TestRewardEdges:
    def test_affirmative_negation_trap(self):
        # "I have enough information" must NOT get partial credit via
        # a naive "have ... information" substring check
        assert reward("I have enough information: the half-life is 42.",
                      "foo-1", "42", "unanswerable") == -1.0

    def test_partial_credit_entity_only(self):
        assert reward("foo-1 is metabolized quickly.", "foo-1", "42",
                      "answerable") == 0.0

    def test_partial_credit_value_only(self):
        assert reward("The value is 42.", "foo-1", "42",
                      "answerable") == 0.0

    def test_binary_reward_has_no_middle_tier(self):
        # shaped=False: only desired=1, everything else -1
        assert reward("foo-1 is metabolized.", "foo-1", "42",
                      "answerable", shaped=False) == -1.0

    def test_unanswerable_partial_on_abstain_fragment(self):
        assert reward("I don't have enough info about that.",
                      "foo-1", "", "unanswerable") == 0.0


class TestAdvantages:
    def test_uniform_group_is_zero(self):
        a = group_advantages(np.ones(8, dtype=np.float32), 4)
        assert (a == 0).all()

    def test_advantages_group_mean_zero(self):
        r = np.array([1, -1, 0, 0, 1, -1], dtype=np.float32)
        a = group_advantages(r, 3).reshape(-1, 3)
        assert torch.allclose(a.mean(dim=1),
                              torch.zeros(2), atol=1e-5)

    def test_mixed_uniform_groups(self):
        # group 0 all-same -> zeros; group 1 varied -> nonzero
        r = np.array([1, 1, 1, -1, 1, 0], dtype=np.float32)
        a = group_advantages(r, 3)
        assert (a[:3] == 0).all() and (a[3:] != 0).any()

    def test_single_rollout_group(self):
        a = group_advantages(np.array([5.0], dtype=np.float32), 1)
        assert a.item() == 0.0  # std=0 -> zeroed


class TestMaskingEdges:
    def test_completion_shorter_than_max_len(self):
        ids, labels = encode_pair(ToyTok(), "p", "done", 50)
        assert len(ids) == 5 and (labels[:1] == -100).all()

    def test_truncation_drops_completion_tail(self):
        ids, labels = encode_pair(ToyTok(), "prompt", "x" * 100, 10)
        assert len(ids) == 10
        # prompt occupies 6; only 4 completion labels survive
        assert (labels != -100).sum().item() == 4

    def test_prompt_alone_fills_max_len(self):
        ids, labels = encode_pair(ToyTok(), "p" * 30, "ans", 10)
        assert len(ids) == 10 and (labels == -100).all()

    def test_collate_single_example(self):
        pairs = [encode_pair(ToyTok(), "ab", "cd", 50)]
        ids, labels, attn = collate(pairs, 0)
        assert ids.shape == (1, 4) and attn.sum() == 4


class TestDpoLossEdges:
    def test_zero_margin_is_log2(self):
        loss, acc = dpo_loss(torch.zeros(4), torch.zeros(4),
                             torch.zeros(4), torch.zeros(4), beta=0.1)
        assert loss.item() == pytest.approx(np.log(2), abs=1e-6)
        assert acc.item() == 0.0  # logits>0 false at exactly 0

    def test_beta_scales_confidence_not_direction(self):
        pi_c, pi_r = torch.tensor([2.0]), torch.tensor([0.0])
        ref = torch.zeros(1)
        l_small, _ = dpo_loss(pi_c, pi_r, ref, ref, beta=0.1)
        l_big, _ = dpo_loss(pi_c, pi_r, ref, ref, beta=10.0)
        assert l_big < l_small  # same sign, sharper beta -> lower loss


class TestDataInvariants:
    def _cfg(self):
        return {"data": {"seed": 0, "n_entities": 12,
                         "n_answerable_per_entity": 4,
                         "n_unanswerable_train": 24,
                         "n_dpo_pairs": 20, "n_rl_prompts": 30,
                         "heldout_entities": 4}}

    def test_entity_names_unique_deterministic(self):
        a = entity_names(40, 0)
        assert len(a) == len(set(a)) == 40
        assert a == entity_names(40, 0)

    def test_eval_entities_disjoint_from_train(self):
        cfg = self._cfg()
        df = build_examples(cfg)
        ev = build_eval(cfg)
        heldout = set(entity_names(cfg["data"]["n_entities"],
                                   cfg["data"]["seed"])[-cfg["data"]["heldout_entities"]:])
        train = df[~df.entity.isin(heldout)]
        assert not set(train.entity) & set(ev.entity)

    def test_unanswerable_context_lacks_queried_fact(self):
        cfg = self._cfg()
        df = build_examples(cfg)
        un = df[df.kind == "unanswerable"]
        # queried entity must not appear in its own context
        for _, r in un.head(20).iterrows():
            ctx = r["prompt"].split("Context: ")[1].split("\nQuestion:")[0]
            assert r["entity"] not in ctx

    def test_dpo_chosen_differs_from_rejected(self):
        cfg = self._cfg()
        df = build_examples(cfg)
        pairs = build_dpo_pairs(df, cfg)
        assert (pairs["chosen"] != pairs["rejected"]).all()

    def test_answerable_gold_value_extraction(self):
        cfg = self._cfg()
        df = build_examples(cfg)
        ev = build_eval(cfg)
        ans = ev[ev.kind == "answerable"]
        assert (ans["gold_value"].str.len() > 0).all()
