from route_graph.audit_pools import origin_controls, origin_pool_id, raw_origin_pool


def test_one_token_origin_has_frozen_same_role_windows_without_sentence_length_bias():
    row = {"prompt": "a b c d e f", "source_span": [0, 11], "response": "g h"}
    alignment = {
        "source_offsets": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
        "response_offsets": [(0, 1), (2, 3)],
        "prompt_length": 6,
    }
    group = {"role": "source", "keys": [4, 5]}
    pool = raw_origin_pool([1], group, list(range(6)), [1.0] * 8, row, alignment)
    assert len(pool) >= 2
    assert all(p["role"] == "source" and 1 not in p["keys"] for p in pool)
    assert all(0.5 <= p["matching"]["key_ratio"] <= 2 for p in pool)
    assert all(abs(p["matching"]["causal_reach_difference"]) <= 0.1 for p in pool)
    frozen = {"origin_control_pools": {origin_pool_id([1], group): pool}}
    labels = {p["id"]: {"relation": "uncertain"} for p in pool}
    assert origin_controls([1], group, frozen, labels, "source") == []
    for p in pool[-2:]:
        labels[p["id"]] = {"relation": "unrelated"}
    chosen = origin_controls([1], group, frozen, labels, "source")
    assert [c["id"] for c in chosen] == [c["id"] for c in pool[-2:]]
    assert len(pool) <= 16
