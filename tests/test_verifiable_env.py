from posttrain.verifiable_env import TEMPLATES, make_pool, reward


def test_reward_tiers():
    assert reward("42", 42) == 1.0
    assert reward("The answer is 42.", 42) == 1.0
    assert reward("I think 7 or maybe 9", 9) == 1.0      # last number wins
    assert reward("the answer is 7", 8) == 0.0           # wrong number
    assert reward("no digits here", 5) == -1.0           # no attempt
    assert reward("4 2", 42) == 0.0                      # "2" != 42


def test_make_pool_deterministic_and_consistent():
    a, b = make_pool(30, seed=5), make_pool(30, seed=5)
    assert a.equals(b)
    for _, row in a.iterrows():
        # the prompt's integers recompute to the stored answer
        import re
        nums = [int(x) for x in re.findall(r"\d+", row["prompt"])]
        assert len(nums) == 2
        for tpl, fn in TEMPLATES:
            if tpl.format(a=nums[0], b=nums[1]) in row["prompt"]:
                assert row["answer"] == fn(nums[0], nums[1])
                break
        else:
            raise AssertionError(f"no template matched {row['prompt']!r}")


def test_division_template_is_exact():
    p = make_pool(200, seed=1)
    div = p[p["prompt"].str.contains("split evenly")]
    for _, row in div.iterrows():
        import re
        a, b = [int(x) for x in re.findall(r"\d+", row["prompt"])]
        assert a % b == 0 and row["answer"] == a // b
