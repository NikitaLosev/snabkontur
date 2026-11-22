"""Правила сведения частичных диффов, производная сумма и полномочия авторов."""

from datetime import datetime

import pytest

from snab.merge import Candidate, authority_problem, derived_total, gate_authority, merge_candidates, to_operations

T1 = (datetime.fromisoformat("2026-11-03T10:15:00+03:00"), 900, 2)
T2 = (datetime.fromisoformat("2026-11-03T11:10:00+03:00"), 4200, 8)
# То же время сообщения, но более поздний кусок записи
T1_LATER_CHUNK = (T1[0], T1[1], 8)


def cand(field: str, value: object, status: str, order: tuple, event: str = "e1", committed: bool = False) -> Candidate:
    """Кандидат P-1 с одной цитатой из события event."""
    return Candidate(
        purchase_id="P-1",
        field=field,
        value=value,
        status=status,
        evidence=({"event_id": event, "quote": f"{field}={value}"},),
        order=order,
        event_id=event,
        committed=committed,
    )


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        # Позднее согласование заменяет раннее
        (
            [cand("unit_price", "12500.00", "agreed", T1), cand("unit_price", "12000.00", "agreed", T2)],
            {"unit_price": "12000.00"},
        ),
        # Порядок задает время события, а не порядок ответа модели
        (
            [cand("unit_price", "12000.00", "agreed", T2), cand("unit_price", "12500.00", "agreed", T1)],
            {"unit_price": "12000.00"},
        ),
        # При одном времени решает номер куска
        (
            [
                cand("delivery_date", "2026-11-20", "agreed", T1_LATER_CHUNK),
                cand("delivery_date", "2026-11-18", "agreed", T1),
            ],
            {"delivery_date": "2026-11-20"},
        ),
        # Предложение и условие не заменяют согласованное значение
        (
            [cand("unit_price", "12500.00", "agreed", T1), cand("unit_price", "12000.00", "offer", T2)],
            {"unit_price": "12500.00"},
        ),
        (
            [cand("unit_price", "12500.00", "agreed", T1), cand("unit_price", "12000.00", "conditional", T2)],
            {"unit_price": "12500.00"},
        ),
        # Одно предложение без принятия ничего не меняет
        ([cand("unit_price", "4500.00", "conditional", T1), cand("prepayment_percent", 100, "conditional", T1)], {}),
        # Новая цена снимает прежнюю группу цена + предоплата целиком
        (
            [
                cand("unit_price", "12500.00", "agreed", T1),
                cand("prepayment_percent", 0, "agreed", T1),
                cand("unit_price", "12000.00", "agreed", T2),
            ],
            {"unit_price": "12000.00"},
        ),
        # Принятая условная цена принимает и ее условие оплаты
        (
            [
                cand("unit_price", "12000.00", "conditional", T1),
                cand("prepayment_percent", 100, "conditional", T1),
                cand("unit_price", "12000", "agreed", T2, "e2"),
            ],
            {"unit_price": "12000", "prepayment_percent": 100},
        ),
        # Явное согласование поля важнее условия из предложения
        (
            [
                cand("unit_price", "12000.00", "conditional", T1),
                cand("prepayment_percent", 100, "conditional", T1),
                cand("unit_price", "12000.00", "agreed", T2),
                cand("prepayment_percent", 50, "agreed", T2),
            ],
            {"unit_price": "12000.00", "prepayment_percent": 50},
        ),
        # Старое письмо, загруженное позже, не перезаписывает записанное решение
        (
            [
                cand("delivery_date", "2026-11-24", "agreed", T2, committed=True),
                cand("delivery_date", "2026-11-10", "agreed", T1),
            ],
            {"delivery_date": "2026-11-24"},
        ),
        # Поля вне группы сводятся независимо
        (
            [cand("budget", "400000.00", "agreed", T1), cand("unit_price", "12000.00", "agreed", T2)],
            {"budget": "400000.00", "unit_price": "12000.00"},
        ),
    ],
)
def test_merge_keeps_one_value_per_key(candidates: list[Candidate], expected: dict) -> None:
    merged = merge_candidates(candidates)

    assert {field: candidate.value for (_, field), candidate in merged.final.items()} == expected
    assert merged.issues == []


def test_same_moment_contradiction_becomes_issue() -> None:
    candidates = [
        cand("delivery_date", "2026-11-18", "agreed", T1, "e3"),
        cand("delivery_date", "2026-11-19", "agreed", T1, "e4"),
    ]

    merged = merge_candidates(candidates)

    assert merged.final == {}
    assert merged.issues == [
        {"code": "unresolved_terms", "purchase_id": "P-1", "field": "delivery_date", "event_ids": ["e3", "e4"]}
    ]


def test_same_number_in_other_format_is_not_contradiction() -> None:
    merged = merge_candidates(
        [cand("unit_price", "12000.00", "agreed", T1), cand("unit_price", 12000, "agreed", T1, "e2")]
    )

    assert merged.issues == []
    assert list(merged.final) == [("P-1", "unit_price")]


def test_superseded_quotes_stay_in_chain() -> None:
    first, offer, second = (
        cand("unit_price", "12500.00", "agreed", T1, "e1"),
        cand("unit_price", "11000.00", "offer", T1_LATER_CHUNK, "e2"),
        cand("unit_price", "12000.00", "agreed", T2, "e3"),
    )

    winner = merge_candidates([first, offer, second]).final[("P-1", "unit_price")]

    assert winner.evidence == second.evidence
    assert [entry["value"] for entry in winner.superseded] == ["12500.00", "11000.00"]
    assert winner.superseded[0]["evidence"] == list(first.evidence)


def test_pending_offer_after_agreement_stays_pending() -> None:
    offer = cand("unit_price", "11000.00", "offer", T2, "e2")

    merged = merge_candidates([cand("unit_price", "12500.00", "agreed", T1), offer])

    assert merged.pending == [offer]


@pytest.mark.parametrize("accepted_at", [T1, T2], ids=["same_moment", "later"])
def test_accepted_offer_moves_to_chain_whenever_accepted(accepted_at: tuple) -> None:
    offer = cand("unit_price", "12000.00", "conditional", T1)
    condition = cand("prepayment_percent", 100, "conditional", T1)

    merged = merge_candidates([offer, condition, cand("unit_price", "12000", "agreed", accepted_at, "e2")])

    assert {field: candidate.value for (_, field), candidate in merged.final.items()} == {
        "unit_price": "12000",
        "prepayment_percent": 100,
    }
    assert merged.pending == []
    chains = {
        field: [entry["status"] for entry in candidate.superseded] for (_, field), candidate in merged.final.items()
    }
    assert chains == {"unit_price": ["conditional"], "prepayment_percent": ["conditional"]}


def test_operations_skip_committed_and_derive_total() -> None:
    final = merge_candidates(
        [
            cand("delivery_date", "2026-11-24", "agreed", T1, committed=True),
            cand("unit_price", "12000.00", "agreed", T2, "e4"),
        ]
    ).final
    cards = {"P-1": {"revision": 1, "fields": {"unit_price": "13000.00", "quantity": 10, "total_amount": "130000.00"}}}

    operations = to_operations(final, cards)

    assert [(op["field"], op["value"]) for op in operations] == [
        ("unit_price", "12000.00"),
        ("total_amount", "120000.00"),
    ]
    assert operations[1]["evidence"] == [{"event_id": "e4", "quote": "unit_price=12000.00"}]


PRICE = {"op": "set_field", "purchase_id": "P-1", "field": "unit_price", "value": "100.00", "evidence": []}
QUANTITY = {"op": "set_field", "purchase_id": "P-1", "field": "quantity", "value": 12, "evidence": []}


@pytest.mark.parametrize(
    ("operations", "fields", "expected"),
    [
        ([PRICE], {"quantity": 3, "total_amount": "390.00"}, "300.00"),
        ([QUANTITY], {"unit_price": "12100.00", "total_amount": "121000.00"}, "145200.00"),
        # Цена без НДС: сумма с НДС сверху, 100 * 3 * 1.2
        ([PRICE], {"quantity": 3, "total_amount": "1.00", "price_includes_vat": False, "vat_rate": 20}, "360.00"),
        # Суммы в карточке нет - не придумываем
        ([PRICE], {"quantity": 3}, None),
        # Сумму назвали в источнике - она важнее расчета
        ([PRICE, {**PRICE, "field": "total_amount", "value": "1.00"}], {"quantity": 3, "total_amount": "5.00"}, None),
        # Неразборчивое значение модели не роняет сборку
        ([{**PRICE, "value": "двенадцать"}], {"quantity": 3, "total_amount": "5.00"}, None),
    ],
)
def test_derived_total(operations: list[dict], fields: dict, expected: str | None) -> None:
    total = derived_total(operations, fields)

    assert (total and total["value"]) == expected


REGISTRY = {"Марина": {"can_confirm": ["budget"]}, "Стажёр": {"can_confirm": []}}


@pytest.mark.parametrize(
    ("event", "problem"),
    [
        ({"speaker_id": "SPEAKER_01"}, "unknown_speaker_authority"),
        ({"author": "SPEAKER_02"}, "unknown_speaker_authority"),
        ({"author": "Стажёр"}, "conflicting_unconfirmed_claim"),
        ({"author": "Марина"}, None),
        # Автор вне реестра прав не теряет: дифф подтверждает закупщик
        ({"author": "Ирина"}, None),
        # Кусок встречи без разметки автора
        ({}, None),
    ],
)
def test_authority_problem_for_budget(event: dict, problem: str | None) -> None:
    assert authority_problem(event, "budget", REGISTRY) == problem


def test_speaker_label_without_registry_needs_no_authority_check() -> None:
    # Как в gold E15-N и E21-N: реестра нет, решение подтверждает закупщик
    assert authority_problem({"speaker_id": "SPEAKER_01"}, "unit_price", {}) is None


def budget(value: str, event_id: str) -> Candidate:
    """Кандидат нового лимита из события event_id."""
    return Candidate("P-1", "budget", value, "agreed", ({"event_id": event_id, "quote": value},), T1, event_id)


def test_unknown_speaker_change_becomes_issue_and_named_author_passes() -> None:
    events = {"e3": {"id": "e3", "speaker_id": "SPEAKER_02"}, "e4": {"id": "e4", "author": "Марина"}}
    cards = {"P-1": {"revision": 1, "fields": {"budget": "600000.00"}}}
    voice, letter = budget("300000.00", "e3"), budget("400000.00", "e4")

    allowed, issues = gate_authority([voice, letter], events, REGISTRY, cards)

    assert allowed == [letter]
    assert issues == [{"code": "unknown_speaker_authority", "event_id": "e3", "purchase_id": "P-1"}]


def test_unknown_speaker_repeating_card_value_asks_nothing() -> None:
    events = {"e2": {"id": "e2", "speaker_id": "SPEAKER_01"}}
    cards = {"P-1": {"revision": 1, "fields": {"budget": "1000000.00"}}}

    allowed, issues = gate_authority([budget("1000000", "e2")], events, REGISTRY, cards)

    assert (allowed, issues) == ([], [])
