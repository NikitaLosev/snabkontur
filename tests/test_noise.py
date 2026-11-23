"""Шумный срез dev: детерминизм, дословные цитаты, схема оценщика и неизменные значения gold."""

import json
import random

import pytest

from eval.noise import (
    DEV_PATH,
    NOISY_PATH,
    SEED,
    CaseEdit,
    EventEdit,
    asr_text,
    build_noisy,
    insert_fillers,
    is_filler_slot,
    make_noisy,
)
from snab.contracts import SCORER

REQUIRED_PARENTS = {"E09", "E10", "E15", "D-WOR-01"}
REQUIRED_CATEGORIES = {"extraction", "decisions", "entities", "trust", "workflow"}


def read_jsonl(path) -> list[dict]:
    """Строки JSONL -> объекты."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def parents() -> dict[str, dict]:
    return {case["id"]: case for case in read_jsonl(DEV_PATH)}


@pytest.fixture(scope="module")
def noisy() -> list[dict]:
    return read_jsonl(NOISY_PATH)


def events_by_id(case: dict) -> dict[str, dict]:
    data = case["input"]
    steps = data.get("timeline") or [{"events": data.get("events", [])}]
    return {event["id"]: event for step in steps for event in step["events"]}


def without_quotes(value):
    """Gold без текстов цитат: остаются значения, ссылки на события и проверки."""
    if isinstance(value, dict):
        return {key: without_quotes(item) for key, item in value.items() if key != "quote"}
    if isinstance(value, list):
        return [without_quotes(item) for item in value]
    return value


def test_same_seed_rebuilds_committed_file(parents: dict[str, dict], noisy: list[dict]) -> None:
    dev = list(parents.values())

    assert build_noisy(dev, SEED) == build_noisy(dev, SEED) == noisy


def test_other_seed_moves_fillers_but_keeps_gold_values(parents: dict[str, dict], noisy: list[dict]) -> None:
    other = build_noisy(list(parents.values()), SEED + 1)

    assert [case["input"] for case in other] != [case["input"] for case in noisy]
    assert [without_quotes(case["gold"]) for case in other] == [without_quotes(case["gold"]) for case in noisy]


def test_every_gold_quote_is_literal_span_of_new_text(noisy: list[dict]) -> None:
    for case in noisy:
        events = events_by_id(case)
        stack = [case["gold"]]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "quote" in node and "event_id" in node:
                    assert node["quote"] in events[node["event_id"]]["text"], (case["id"], node)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)


def test_cases_pass_scorer_schema(noisy: list[dict]) -> None:
    assert {case["id"]: SCORER.validation_errors(case, SCORER.DatasetCase) for case in noisy} == {
        case["id"]: [] for case in noisy
    }


def test_ids_unique_and_parents_exist(parents: dict[str, dict], noisy: list[dict]) -> None:
    ids = [case["id"] for case in noisy]

    assert len(ids) == len(set(ids)) == 15
    for case in noisy:
        parent = parents[case["origin"]["parent_id"]]
        assert case["id"] == parent["id"] + "-N"
        assert case["origin"]["kind"] == "synthetic_noise"
        assert case["scenario_family_id"] == parent["scenario_family_id"]
        assert case["split"] == "dev"
        assert case["dataset_version"] == parent["dataset_version"] + "-noisy.1"
        assert case["origin"]["gold_note"]


def test_slice_covers_required_parents_and_categories(noisy: list[dict]) -> None:
    assert REQUIRED_PARENTS <= {case["origin"]["parent_id"] for case in noisy}
    assert REQUIRED_CATEGORIES <= {case["category"] for case in noisy}


def test_gold_values_match_parent(parents: dict[str, dict], noisy: list[dict]) -> None:
    # Ни одна правка среза не меняет смысл, поэтому меняются только тексты цитат
    for case in noisy:
        assert without_quotes(case["gold"]) == without_quotes(parents[case["origin"]["parent_id"]]["gold"]), case["id"]


def test_texts_differ_from_parent(parents: dict[str, dict], noisy: list[dict]) -> None:
    for case in noisy:
        before = events_by_id(parents[case["origin"]["parent_id"]])
        changed = [
            event_id for event_id, event in events_by_id(case).items() if event["text"] != before[event_id]["text"]
        ]
        assert changed, case["id"]


def test_answer_revealing_tail_removed(noisy: list[dict]) -> None:
    text = events_by_id(next(case for case in noisy if case["id"] == "E09-N"))["e1"]["text"]

    assert "не согласовано" not in text
    assert "10 октября" not in text


def test_meeting_speech_is_asr_with_speaker_label(noisy: list[dict]) -> None:
    event = events_by_id(next(case for case in noisy if case["id"] == "D-WOR-01-N"))["D-WOR-01-e4"]

    assert event["text"].startswith("SPEAKER_01: ")
    assert event["speaker_id"] == "SPEAKER_01"
    assert not any(char in event["text"].removeprefix("SPEAKER_01: ") for char in ".,:;")
    body = event["text"].removeprefix("SPEAKER_01: ")
    assert body == body.lower()


def test_asr_text_drops_punctuation_and_keeps_ids() -> None:
    assert asr_text("Так, по P-17 — насосы: 12.5 тысяч. Да!") == "так по P-17 насосы 12.5 тысяч да"


class FixedSlot(random.Random):
    """Фейк генератора: вставка одного "ну" перед заданным словом."""

    def __init__(self, position: int) -> None:
        super().__init__(0)
        self.position = position

    def sample(self, population, k):
        return [self.position]

    def choice(self, seq):
        return "ну"


@pytest.mark.parametrize(
    ("before_word", "expected_quote"),
    [
        ("двадцать", "берём ну двадцать"),  # внутри цитаты: граница конца сдвигается
        ("берём", "берём двадцать"),  # перед цитатой: сдвигаются обе границы
        ("штук", "берём двадцать"),  # после цитаты: границы на месте
    ],
)
def test_insert_fillers_keeps_quote_literal(before_word: str, expected_quote: str) -> None:
    text = "по P-1 берём двадцать штук"
    start = text.index("берём")
    span = [start, start + len("берём двадцать")]

    result = insert_fillers(text, [span], asr=True, rng=FixedSlot(text.index(before_word)))

    assert result[span[0] : span[1]] == expected_quote


@pytest.mark.parametrize(
    ("previous", "allowed"),
    [("по", False), ("октября", False), ("двенадцать", False), ("P-17,", False), ("согласовали", True)],
)
def test_filler_never_splits_number_date_or_preposition(previous: str, allowed: bool) -> None:
    assert is_filler_slot(previous) is allowed


def test_rewrite_without_quote_mapping_fails(parents: dict[str, dict]) -> None:
    edit = CaseEdit(("vague_tail",), "test", {"e1": EventEdit("по P-17 цена другая")})

    with pytest.raises(ValueError, match="is not in"):
        make_noisy(parents["E10"], edit, SEED)
