"""Scientific checks for causal indexing, paired identity and finite head effects."""


import numpy as np

from test_path_conflict import WordTokenizer
from experiments.path_conflict.paired_inputs import compile_pair, phase_positions, source_groups


def test_phases_and_claim_history_never_include_unconsumed_target():
    roles = dict(scope=np.array([0]), supported_value=np.array([1]), value_source=np.array([2]))
    groups = source_groups(3, 6, roles, (2, 5))
    np.testing.assert_array_equal(groups["history"], [3, 4, 5])
    np.testing.assert_array_equal(groups["query_self"], [5])
    np.testing.assert_array_equal(groups["prior_history"], [3, 4])
    np.testing.assert_array_equal(groups["claim_history"], [5])
    assert 6 not in groups["head_total"]
    phases = phase_positions((0, 1), 2, np.array([False, True]))
    assert phases == {"onset": 0}  # No invented before/back-half or EOS recovery.


def test_saved_tokens_define_both_sides_without_inventing_a_missing_phase(tmp_path):
    tokenizer = WordTokenizer()
    prompt_ids = tokenizer.encode("Only ruler. Ordinary caps. Royal gold. Output:")
    samples = []
    for seed, response in [(0, "They wear caps with cloth. End"), (1, "They instead wear gold")]:
        ids = prompt_ids + tokenizer.encode(response)
        length = len(ids) - len(prompt_ids)
        name = f"{seed}.npz"
        np.savez(tmp_path / name, token_ids=ids, prompt_length=len(prompt_ids),
                 token_text=[tokenizer.decode([token]) for token in ids],
                 chosen_logit=np.zeros(length), log_normalizer=np.zeros(length),
                 top_ids=np.zeros((length, 5), dtype=int), top_logits=np.zeros((length, 5)),
                 special_mask=np.zeros(len(ids), dtype=bool))
        samples.append(dict(source_id="one", seed=seed, trace=name, response=response))
    case = dict(case_id="claim", source_id="one", supported=dict(seed=0, target="caps with cloth"),
                unsupported=dict(seed=1, target="gold"), candidates=[" caps with cloth", " gold"],
                evidence=["Only ruler.", "Ordinary caps."], value_source=["Royal gold."],
                source_roles=dict(scope=["Only ruler."], supported_value=["Ordinary caps."],
                                  value_source=["Royal gold."]))
    pair = compile_pair(tmp_path, tokenizer, samples, case)
    assert set(pair["supported"]) == {"before_claim", "onset", "back_half", "post_claim"}
    assert set(pair["unsupported"]) == {"before_claim", "onset"}
    for phases in pair.values():
        assert tokenizer.decode(phases["onset"]["prefix_ids"]).endswith("wear")
        assert phases["onset"]["groups"]["claim_history"].size == 0
        for probe in phases.values():
            assert probe["prefix_ids"][:len(prompt_ids)] == prompt_ids
            assert probe["groups"]["head_total"][-1] == len(probe["prefix_ids"]) - 1
