from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.jsonreply import NoJsonFound, extract_json


def test_plain_object_and_array() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("[1, 2]") == [1, 2]


def test_fenced_block_with_prose_around_it() -> None:
    text = 'Sure! Here you go:\n```json\n{"a": {"b": [1, 2]}}\n```\nHope that helps {really}.'
    assert extract_json(text) == {"a": {"b": [1, 2]}}


def test_fence_beats_earlier_braces_in_prose() -> None:
    assert extract_json('Use {braces} like so:\n```\n{"ok": true}\n```') == {"ok": True}


def test_think_block_is_ignored_even_when_it_contains_json() -> None:
    assert extract_json('<think>maybe {"a": 0}</think>{"a": 1}') == {"a": 1}


def test_prose_before_and_after_without_fence() -> None:
    assert extract_json('Result: {"n": "}"} done') == {"n": "}"}


def test_cjk_and_nested_strings() -> None:
    assert extract_json('{"text": "他说：\\"你好\\""}') == {"text": '他说："你好"'}


@pytest.mark.parametrize("text", ["", "no json here", "{broken", "<think>{}</think>"])
def test_no_json(text: str) -> None:
    with pytest.raises(NoJsonFound):
        extract_json(text)


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(st.integers(), st.text(max_size=10), st.lists(st.integers(), max_size=3)),
        min_size=1,
        max_size=4,
    )
)
def test_roundtrip_through_noise(obj: dict[str, object]) -> None:
    import json

    wrapped = f"Here:\n```json\n{json.dumps(obj, ensure_ascii=False)}\n```\nbye"
    assert extract_json(wrapped) == obj
