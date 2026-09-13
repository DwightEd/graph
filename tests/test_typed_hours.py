"""Typed interpretation boundary checks; fixtures are not empirical results."""

import copy

import pytest

from next_iteration.typed_hours import DAYS, assess_base, source_schedule


def source(**changes):
    return repr({"name": "Cafe", "hours": dict.fromkeys(DAYS, "8:0-16:0"),
        "attributes": {"WiFi": "free"}, **changes})


def fact(text="The business hours are from 8 AM to 6 PM every day, and they offer free WiFi.", source_text=None):
    schedule = source_schedule(source_text or source(), source_id="fixture")
    row = {"id": "fixture", "source_id": "fixture", "response": text}
    base = {"text": text, "span": [0, len(text)]}
    return assess_base(row, base, schedule), schedule


def test_changes_one_endpoint_preserves_day_scope_wifi_and_every_other_byte():
    result, schedule = fact()
    assert result["status"] == "typed_contrast_available"
    assert result["replacement"] == "4 PM"
    a, b = result["answer_span"]
    original = result["original_base"]
    assert original[a:b] == "6 PM"
    assert result["edited_base"] == original[:a] + "4 PM" + original[b:]
    assert len(result["source_value_spans"]) == 7
    assert result["mismatched_days"] == list(DAYS)
    for a, b in result["source_value_spans"]:
        assert schedule["source_graph"]["text"][a:b] == "16:0"


def test_one_different_day_cannot_be_hidden_by_six_matching_days():
    hours = dict.fromkeys(DAYS, "8:0-16:0")
    hours["Sunday"] = "8:0-18:0"
    result, _ = fact(source_text=source(hours=hours))
    assert result["typed_relation"] == "conflict"
    assert result["status"] == "schedule_varies_no_single_range_repair"
    assert result["mismatched_days"] == list(DAYS[:-1])


def test_missing_day_is_unknown_not_all_days_supported():
    hours = dict.fromkeys(DAYS[:-1], "8:0-16:0")
    with pytest.raises(ValueError, match="missing or unknown"):
        source_schedule(source(hours=hours), source_id="fixture")


@pytest.mark.parametrize("wifi", [None, False, "paid", "no"])
def test_non_target_wifi_assertion_must_be_independently_supported(wifi):
    result, _ = fact(source_text=source(attributes={"WiFi": wifi}))
    assert result["typed_relation"] == "conflict"
    assert result["status"] == "non_target_wifi_not_supported"


def test_supported_clock_does_not_create_an_error_contrast():
    result, _ = fact("The business hours are from 8 AM to 4 PM every day.")
    assert result["status"] == "typed_supported_no_error_contrast"


@pytest.mark.parametrize("text", [
    "The business is not open from 8 AM to 6 PM every day.",
    'He said: "The business hours are from 8 AM to 6 PM every day."',
    "The other business hours are from 8 AM to 6 PM every day.",
    "It operates from 8 AM to 6 PM every day.",
    "The business hours are from 8 to 6 PM every day.",
    "The business hours are from 8 AM to 6 PM on some days."])
def test_unsupported_owner_polarity_quote_or_quantifier_never_enters_contrast(text):
    result, _ = fact(text)
    assert result["status"] == "response_relation_or_scope_outside_grammar"


def test_clock_rendering_preserves_minute_and_case_style():
    result, _ = fact("The business hours are from 8:00 am to 6:00 pm daily.")
    assert result["replacement"] == "4:00 pm"


def test_two_endpoint_errors_are_not_a_single_value_contrast():
    result, _ = fact("The business hours are from 9 AM to 6 PM every day.")
    assert result["status"] == "multiple_endpoint_changes_not_single_value_contrast"


def test_overnight_source_does_not_get_a_guessed_meridiem():
    with pytest.raises(ValueError, match="overnight"):
        source_schedule(source(hours=dict.fromkeys(DAYS, "5:0-2:0")), source_id="fixture")


def test_mutated_schedule_is_rejected():
    _, schedule = fact()
    schedule = copy.deepcopy(schedule)
    schedule["days"][0]["minutes"] = [480, 1080]
    with pytest.raises(ValueError):
        assess_base({"id": "fixture", "source_id": "fixture", "response": "x"},
            {"span": [0, 1], "text": "x"}, schedule)
