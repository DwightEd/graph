import numpy as np
import pandas as pd

from experiments.path_conflict.flow import attach_identity
from experiments.path_conflict.flow_inputs import add_flow_groups
from experiments.path_conflict.flow_plan import opposition, select_heads, sign_role
from experiments.path_conflict.flow_report import role_table
from experiments.charm_structure_audit.supervised_head_roles import head_statistics


def test_flow_groups_are_exact_aggregates():
    probe = dict(prefix_ids=list(range(9)), groups=dict(
        scope=np.array([0, 1]),
        supported_value=np.array([2]),
        value_source=np.array([3]),
        other_prompt=np.array([4]),
        remote_history=np.array([5]),
        recent_history=np.array([6, 7]),
        query_self=np.array([8]),
    ))
    result = add_flow_groups(probe)
    np.testing.assert_array_equal(result["groups"]["evidence"], [0, 1, 2])
    np.testing.assert_array_equal(result["groups"]["history"], [5, 6, 7, 8])
    np.testing.assert_array_equal(result["groups"]["all_context"], np.arange(9))


def test_head_selection_uses_baseline_local_effect():
    rows = []
    for head, value in enumerate([.1, 2.0, -.8]):
        for group in ("all_context", "evidence", "wrong_source", "history"):
            rows.append(dict(side="unsupported", panel="natural", layer=1, head=head,
                             source_group=group, local_lens_support=value))
    selected = select_heads(pd.DataFrame(rows), top_k=1, max_heads=2)
    assert selected[0] == (1, 1)


def test_roles_and_opposition_are_directional():
    assert sign_role(.2) == "supports_correct"
    assert sign_role(-.2) == "supports_wrong"
    assert sign_role(0) == "neutral"
    assert opposition(2, -2) == 1
    assert opposition(2, 3) == 0


def test_role_table_detects_downstream_reversal(tmp_path):
    rows = []
    for group, local in [
        ("all_context", 1.0),
        ("evidence", .8),
        ("wrong_source", -.2),
        ("history", -.4),
    ]:
        rows.append(dict(case_id="c", side="unsupported", panel="natural",
                         layer=2, head=3, source_group=group,
                         attention_mass=.2, local_lens_support=local))
    pd.DataFrame(rows).to_csv(tmp_path / "baseline_head_sources.csv.gz", index=False)

    final = []
    for group, support in [
        ("all_context", -.5),
        ("evidence", .3),
        ("wrong_source", -.1),
        ("history", -.2),
    ]:
        final.append(dict(case_id="c", side="unsupported", panel="natural",
                          layer=2, head=3, source_group=group,
                          final_support=support))
    pd.DataFrame(final).to_csv(tmp_path / "head_interventions.csv", index=False)

    result = role_table(tmp_path).iloc[0]
    assert result.local_role == "supports_correct"
    assert result.final_role == "supports_wrong"
    assert bool(result.downstream_reversal)
    assert result.local_source_opposition > 0


def test_supervised_head_statistics_separates_self_and_prompt_roles():
    values = np.array([[.1, .6], [.2, .5], [.8, .2], [.9, .1]])
    prompt = np.array([[.8, .2], [.7, .3], [.2, .7], [.1, .8]])
    labels = np.array([0, 0, 1, 1])
    frame = head_statistics(values, prompt, labels, np.ones(2), np.ones(2), heads=2)
    assert frame.loc[0, "routing_pattern"] == "self_up_prompt_down"
    assert frame.loc[1, "routing_pattern"] == "self_down_prompt_up"


def test_write_identity_ignores_candidate_list():
    frame = pd.DataFrame(dict(layer=[0, 1], head=[2, 3]))
    identity = dict(
        case_id="c",
        source_id="s",
        side="supported",
        panel="natural",
        seed=0,
        trace="00000.npz",
        query=42,
        candidates=[" correct", " wrong"],
    )
    result = attach_identity(frame, identity)
    assert "candidates" not in result
    assert result.case_id.tolist() == ["c", "c"]
    assert result["query"].tolist() == [42, 42]
