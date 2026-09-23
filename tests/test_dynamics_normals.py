"""Normal high-score runs must respect labels, missingness and answer borders."""

from experiments.native_support.dynamics_normals import normal_tables


def tokens():
    return [{"response_id": "a", "source_id": "s", "target": i, "label": int(i == 2), "token": str(i),
             "state_dynamics": .99, "future_query_count": max(4 - i, 0), "state_log_odds": 4.,
             "entropy": 1., "prompt_routing_imbalance": .5, "emission_log_ratio": 3.,
             "mean_head_source_read": .1, "mean_head_history_read": .3} for i in range(6)]


def test_normal_runs_do_not_cross_labels_missingness_or_answers():
    rows = tokens()
    rows.append({**rows[0], "response_id": "b", "target": 4, "token": "new"})
    tables = normal_tables(rows)
    assert [row["text"] for row in tables["normal_high_runs"]] == ["01", "3", "new"]
    assert sum(row["tokens"] for row in tables["normal_high_counts"]) == 7
    assert tables["normal_high_answer_groups"] == [
        {"evaluated_answer_has_error": False, "label": 0, "tokens": 1, "responses": 1},
        {"evaluated_answer_has_error": True, "label": 0, "tokens": 3, "responses": 1},
        {"evaluated_answer_has_error": True, "label": 1, "tokens": 1, "responses": 1},
    ]


def test_normal_runs_do_not_bridge_excluded_tokens():
    rows = [row for row in tokens() if row["target"] in (0, 3)]
    tables = normal_tables(rows)
    assert [(row["start"], row["end"]) for row in tables["normal_high_runs"]] == [(0, 0), (3, 3)]
