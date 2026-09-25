import pytest
import torch

from posttrain.dpo import dpo_loss, seq_logp_batch
from posttrain.evaluate import classify
from posttrain.data import ABSTAIN
from posttrain.lm import collate, encode_pair


class ToyTok:
    pad_token_id = 0

    def __call__(self, s, add_special_tokens=True, **kw):
        ids = [ord(c) % 50 + 1 for c in s]
        return {"input_ids": ids}


def test_encode_pair_masks_prompt():
    tok = ToyTok()
    ids, labels = encode_pair(tok, "PROMPT", "done", 50)
    assert len(ids) == len(labels)
    n_prompt = len(tok("PROMPT")["input_ids"])
    assert (labels[:n_prompt] == -100).all()
    assert (labels[n_prompt:] != -100).all()


def test_collate_padding():
    tok = ToyTok()
    pairs = [encode_pair(tok, "a", "bc", 50), encode_pair(tok, "longer", "d", 50)]
    ids, labels, attn = collate(pairs, 0)
    assert ids.shape[0] == 2 and attn[0].sum() < attn[1].sum()
    assert (labels[0][attn[0] == 0] == -100).all()


def test_dpo_loss_prefers_chosen():
    # logits = beta * margin = 0.1 * 10 = 1 -> loss = -log sigmoid(1) ~ 0.313
    loss, acc = dpo_loss(
        torch.tensor([5.0]), torch.tensor([-5.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1,
    )
    assert loss.item() == pytest.approx(0.3133, abs=1e-3)
    assert acc.item() == 1.0
    # same margin at beta=1 -> loss ~ 0
    loss_b, _ = dpo_loss(
        torch.tensor([5.0]), torch.tensor([-5.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0,
    )
    assert loss_b.item() < 0.001
    # reversed preference -> high loss, acc 0
    loss2, acc2 = dpo_loss(
        torch.tensor([-5.0]), torch.tensor([5.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1,
    )
    assert loss2.item() == pytest.approx(1.3133, abs=1e-3)
    assert acc2.item() == 0.0


def test_classify():
    assert classify(ABSTAIN, "x-1", "12", "unanswerable") == "abstains"
    # right entity + right value, degenerate tail -> still correct
    assert classify(
        "The half-life of x-1 is 62 don don don", "x-1", "62", "answerable"
    ) == "correct"
    # wrong value -> fabricates
    assert classify(
        "The half-life of x-1 is 63 hours.", "x-1", "62", "answerable"
    ) == "fabricates"
    # word-boundary: '62' must not match inside '620'
    assert classify(
        "The half-life of x-1 is 620 hours.", "x-1", "62", "answerable"
    ) == "fabricates"
    # right value but wrong entity -> fabricates
    assert classify(
        "The half-life of y-9 is 62 hours.", "x-1", "62", "answerable"
    ) == "fabricates"
    # repetition collapse -> degenerate, not fabrication
    assert classify("I I I I I I I I I I", "x-1", "", "unanswerable") == "degenerate"
    assert classify("388-0 is 388-0 is 388-0 is", "x-1", "62", "answerable") == "degenerate"
    assert classify("I made this up.", "x-1", "", "unanswerable") == "fabricates"
