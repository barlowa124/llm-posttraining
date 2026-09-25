import numpy as np
import torch

from posttrain.grpo import group_advantages, reward


def test_reward_desired_labels():
    from posttrain.data import ABSTAIN

    assert reward("The half-life of foo-1 is 42 hours.", "foo-1", "42",
                  "answerable") == 1.0
    assert reward(ABSTAIN, "foo-1", "42", "answerable") == -1.0
    assert reward(ABSTAIN, "foo-1", "", "unanswerable") == 1.0
    assert reward("The half-life of foo-1 is 99 hours.", "foo-1", "42",
                  "unanswerable") == -1.0


def test_reward_penalizes_degenerate():
    r = reward("foo foo foo foo foo foo foo", "foo-1", "42", "answerable")
    assert r == -1.0


def test_reward_no_credit_for_confident_fabrication():
    # "I have enough information" + fabricated value on an unanswerable
    # prompt must score -1, not partial credit for the abstain fragment
    r = reward("I have enough information: the half-life of foo-1 is 99 hours.",
               "foo-1", "", "unanswerable")
    assert r == -1.0
    # the actual hedge fragment still earns the shaped middle tier
    r2 = reward("I don't have enough information, but it might be 99 hours.",
                "foo-1", "", "unanswerable")
    assert r2 == 0.0


def test_group_advantages_normalized():
    r = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
    a = group_advantages(r, k=3)
    assert a.shape == (6,)
    for g in (a[:3], a[3:]):
        assert abs(float(g.mean())) < 1e-6


def test_uniform_group_gives_zero_signal():
    r = np.ones(6, dtype=np.float32)
    a = group_advantages(r, k=3)
    assert torch.equal(a, torch.zeros(6))


def test_reward_sets_sign_correctly():
    # mixed group: the +1 rollout should get positive advantage, -1 negative
    r = np.array([1.0, -1.0], dtype=np.float32)
    a = group_advantages(r, k=2)
    assert a[0] > 0 > a[1]
