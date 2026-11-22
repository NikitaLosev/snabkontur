"""Детерминированная сборка частичных диффов в один пакет на закупку.

Модель по каждому куску возвращает кандидатов: закупка, поле, значение, статус и цитаты.
Код сводит их по ключу (purchase_id, field):

- позднее согласованное значение заменяет раннее; порядок - время события, таймкод, номер куска
- предложение или условие никогда не заменяет согласованное значение
- цена, предоплата и сумма заказа меняются вместе: новое согласование одной из них
  снимает прежнюю группу целиком, а принятие условной цены принимает и ее условие
- два разных согласованных значения в один момент - вопрос закупщику, а не выбор кода
- в пакете остается не больше одной операции на ключ, замененные цитаты хранятся в цепочке
  `superseded` у победившей операции

Здесь же проверка полномочий автора: метка SPEAKER_xx не дает прав, права названного
автора берутся только из authority_registry.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import groupby

AGREED = "agreed"
STATUSES = ("agreed", "offer", "conditional")
LINKED_FIELDS = frozenset({"unit_price", "prepayment_percent", "total_amount"})
TOTAL_INPUTS = ("unit_price", "quantity")
EARLIEST = datetime.min.replace(tzinfo=UTC)
SPEAKER_LABEL = re.compile(r"^SPEAKER_\d+$")
MEDIA_TIME = re.compile(r"^(\d+):(\d{2}):(\d{2})$")
CENTS = Decimal("0.01")

Order = tuple[datetime, int, int]


@dataclass(frozen=True)
class Candidate:
    """Значение поля из одного куска; order - момент последней цитаты.

    committed отмечает уже предложенное и записанное в CRM значение: оно участвует в сведении,
    чтобы старое письмо не перезаписало более новое решение, но второй раз не предлагается.
    grounds - цитаты, в сообщении которых есть само значение: None, если код не может это
    проверить (значение не число), пусто, если числа нет ни в одном сообщении цитат.
    """

    purchase_id: str
    field: str
    value: str | int | float | bool | None
    status: str
    evidence: tuple[dict, ...]
    order: Order
    event_id: str
    superseded: tuple[dict, ...] = ()
    committed: bool = False
    grounds: tuple[dict, ...] | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Ключ сведения."""
        return self.purchase_id, self.field


@dataclass(frozen=True)
class Merge:
    """Итог сведения: одно согласованное значение на ключ, непринятые предложения, вопросы."""

    final: dict[tuple[str, str], Candidate]
    pending: list[Candidate]
    issues: list[dict]


def media_seconds(media_time: str | None) -> int:
    """'01:10:00' -> 4200; без таймкода 0."""
    match = MEDIA_TIME.match(media_time or "")
    if match is None:
        return 0
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def event_order(event: dict) -> Order:
    """Момент события: время, затем таймкод внутри записи, затем номер куска."""
    occurred = event.get("occurred_at")
    moment = datetime.fromisoformat(occurred) if occurred else EARLIEST
    return moment, media_seconds(event.get("media_time")), event.get("chunk_index") or 0


def same_value(left: object, right: object) -> bool:
    """Числа сравниваются по значению ("12000.00" и 12000), остальное как есть."""
    left_number, right_number = to_decimal(left), to_decimal(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return left == right


def chain_entry(candidate: Candidate) -> dict:
    """Запись цепочки замененных значений: что было, с каким статусом и откуда."""
    return {"value": candidate.value, "status": candidate.status, "evidence": list(candidate.evidence)}


def supersede(winner: Candidate, losers: Iterable[Candidate]) -> Candidate:
    """Победитель забирает цепочки и цитаты проигравших в superseded."""
    chain = list(winner.superseded)
    for loser in losers:
        chain.extend(loser.superseded)
        chain.append(chain_entry(loser))
    return replace(winner, superseded=tuple(chain))


def contradiction_issue(candidates: list[Candidate]) -> dict:
    """Разные согласованные значения одного поля в один момент -> вопрос закупщику."""
    first = candidates[0]
    return {
        "code": "unresolved_terms",
        "purchase_id": first.purchase_id,
        "field": first.field,
        "event_ids": sorted({candidate.event_id for candidate in candidates}),
    }


def accepted_conditions(agreed: list[Candidate], pending: list[Candidate]) -> list[Candidate]:
    """Принятая условная цена принимает и связанные с ней условия того же предложения.

    Предложение - кандидаты одной закупки из одного момента. Явно согласованное поле
    в этом же куске важнее условия из предложения.
    """
    stated = {candidate.key for candidate in agreed}
    promoted = []
    for accepted in agreed:
        if accepted.field not in LINKED_FIELDS:
            continue
        offers = [
            offer
            for offer in pending
            if offer.key == accepted.key and same_value(offer.value, accepted.value) and offer.order <= accepted.order
        ]
        for offer in offers:
            for condition in pending:
                same_offer = condition.purchase_id == offer.purchase_id and condition.order == offer.order
                if same_offer and condition.field in LINKED_FIELDS and condition.key not in stated:
                    stated.add(condition.key)
                    promoted.append(
                        replace(
                            condition,
                            status=AGREED,
                            evidence=condition.evidence + accepted.evidence,
                            order=accepted.order,
                            event_id=accepted.event_id,
                        )
                    )
    return promoted


def apply_moment(final: dict, pending: list[Candidate], moment: list[Candidate], issues: list[dict]) -> list[Candidate]:
    """Кандидаты одного момента меняют итог; возвращает непринятые предложения после момента."""
    agreed = [candidate for candidate in moment if candidate.status == AGREED]
    offered = pending + [candidate for candidate in moment if candidate.status != AGREED]
    agreed += accepted_conditions(agreed, offered)
    by_key = {}
    for candidate in agreed:
        by_key.setdefault(candidate.key, []).append(candidate)
    winners = {}
    for key, same_key in by_key.items():
        if any(not same_value(candidate.value, same_key[0].value) for candidate in same_key):
            issues.append(contradiction_issue(same_key))
            final.pop(key, None)
            continue
        winners[key] = same_key[-1]
    for purchase_id in {candidate.purchase_id for candidate in winners.values() if candidate.field in LINKED_FIELDS}:
        group = [key for key in final if key[0] == purchase_id and key[1] in LINKED_FIELDS and key not in winners]
        dropped = [final.pop(key) for key in group]
        if dropped:
            anchor = min((key for key in winners if key[0] == purchase_id and key[1] in LINKED_FIELDS), key=str)
            winners[anchor] = supersede(winners[anchor], dropped)
    for key, winner in winners.items():
        # Предложения этого же момента тоже проигравшие: принятие в том же куске, что и
        # предложение, дает ту же цепочку, что и принятие позже
        losers = [offer for offer in offered if offer.key == key]
        if key in final:
            losers.append(final[key])
        final[key] = supersede(winner, sorted(losers, key=lambda loser: loser.order))
    return [offer for offer in offered if offer.key not in winners]


def merge_candidates(candidates: Iterable[Candidate]) -> Merge:
    """Кандидаты из кусков в любом порядке -> итог по ключу (purchase_id, field)."""
    final, pending, issues = {}, [], []
    ordered = sorted(candidates, key=lambda candidate: candidate.order)
    for _, moment in groupby(ordered, key=lambda candidate: candidate.order):
        pending = apply_moment(final, pending, list(moment), issues)
    return Merge(final=final, pending=pending, issues=issues)


def to_decimal(value: object) -> Decimal | None:
    """Значение модели -> Decimal или None, если это не число."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def derived_total(operations: list[dict], fields: dict) -> dict | None:
    """Новая цена или количество при известной сумме заказа -> операция новой суммы.

    Сумма считается как у оценщика: количество на цену, с НДС сверху, если цена без НДС.
    Модель на 4B часто ошибается в умножении, поэтому сумму считает код.
    """
    changed = {operation["field"]: operation for operation in operations if operation["op"] == "set_field"}
    if "total_amount" in changed or fields.get("total_amount") is None:
        return None
    inputs = [changed[name] for name in TOTAL_INPUTS if name in changed]
    if not inputs:
        return None
    values = {name: changed[name]["value"] if name in changed else fields.get(name) for name in TOTAL_INPUTS}
    price, quantity = to_decimal(values["unit_price"]), to_decimal(values["quantity"])
    if price is None or quantity is None:
        return None
    total = price * quantity
    if fields.get("price_includes_vat") is False:
        rate = to_decimal(fields.get("vat_rate"))
        if rate is None:
            return None
        total *= 1 + rate / 100
    evidence = [quote for operation in inputs for quote in operation["evidence"]]
    unique = [quote for index, quote in enumerate(evidence) if quote not in evidence[:index]]
    return {
        "op": "set_field",
        "purchase_id": inputs[0]["purchase_id"],
        "field": "total_amount",
        "value": str(total.quantize(CENTS, rounding=ROUND_HALF_UP)),
        "evidence": unique,
    }


def derived_status(operations: list[dict], fields: dict) -> dict | None:
    """Новое принятое количество меньше заказанного -> операция статуса partially_delivered.

    Модель на 4B ставила этот статус и на сообщения без приемки, а на приемку забывала,
    поэтому статус считает код. Для полной поставки отдельного статуса в CRM нет.
    Совпадение с текущим статусом убирает build_proposals.
    """
    changed = {operation["field"]: operation for operation in operations if operation["op"] == "set_field"}
    if "delivered_quantity" not in changed:
        return None
    delivered = to_decimal(changed["delivered_quantity"]["value"])
    quantity = to_decimal(changed["quantity"]["value"] if "quantity" in changed else fields.get("quantity"))
    if delivered is None or quantity is None or not 0 < delivered < quantity:
        return None
    return {
        "op": "set_field",
        "purchase_id": changed["delivered_quantity"]["purchase_id"],
        "field": "status",
        "value": "partially_delivered",
        "evidence": list(changed["delivered_quantity"]["evidence"]),
    }


def to_operations(final: dict[tuple[str, str], Candidate], cards: dict) -> list[dict]:
    """Новые согласованные значения -> операции set_field с цепочкой, производной суммой и статусом."""
    by_purchase = {}
    for candidate in sorted(final.values(), key=lambda item: item.key):
        if candidate.committed:
            continue
        operation = {
            "op": "set_field",
            "purchase_id": candidate.purchase_id,
            "field": candidate.field,
            "value": candidate.value,
            "evidence": list(candidate.evidence),
        }
        if candidate.superseded:
            operation["superseded"] = list(candidate.superseded)
        by_purchase.setdefault(candidate.purchase_id, []).append(operation)
    operations = []
    for purchase_id, purchase_operations in by_purchase.items():
        operations.extend(purchase_operations)
        fields = cards.get(purchase_id, {}).get("fields", {})
        derived = [derived_total(purchase_operations, fields), derived_status(purchase_operations, fields)]
        operations.extend(operation for operation in derived if operation is not None)
    return operations


def authority_problem(event: dict, field: str, registry: dict) -> str | None:
    """Событие -> код проблемы полномочий для поля или None, если реплика может обосновать изменение.

    Метка диаризации не устанавливает личность. Права проверяются только у автора из реестра;
    автор вне реестра права не получает и не теряет: дифф все равно подтверждает закупщик.
    Без реестра проверять нечего: решение и так подтверждает закупщик, как в gold E15-N и E21-N.
    """
    if not registry:
        return None
    author = event.get("author")
    if author is None or SPEAKER_LABEL.match(author):
        labelled = author is not None or event.get("speaker_id") is not None
        return "unknown_speaker_authority" if labelled else None
    rights = registry.get(author)
    if rights is not None and field not in rights.get("can_confirm", []):
        return "conflicting_unconfirmed_claim"
    return None


def gate_authority(
    candidates: list[Candidate], events: dict[str, dict], registry: dict, cards: dict
) -> tuple[list[Candidate], list[dict]]:
    """Кандидаты -> допущенные и вопросы закупщику о полномочиях.

    Кандидат допущен, если хотя бы одна цитата, из которой взято значение, из реплики без
    проблемы полномочий: посторонняя цитата уполномоченного автора чужое число не пропускает.
    Число, которого нет ни в одном сообщении цитат (4B пишет его словами), могло прийти
    из любой цитаты, поэтому такой кандидат допущен, только если проблем нет ни у одной.
    Вопрос задается только о значении, которое отличается от карточки.
    """
    allowed, issues = [], []
    for candidate in candidates:
        quotes = candidate.grounds or candidate.evidence
        problems = [authority_problem(events[quote["event_id"]], candidate.field, registry) for quote in quotes]
        passed = all(problem is None for problem in problems) if candidate.grounds == () else None in problems
        if passed:
            allowed.append(candidate)
            continue
        current = cards.get(candidate.purchase_id, {}).get("fields", {}).get(candidate.field)
        if same_value(current, candidate.value):
            continue
        code, event_id = next(
            (problem, quote["event_id"]) for problem, quote in zip(problems, quotes, strict=True) if problem is not None
        )
        issue = {"code": code, "event_id": event_id, "purchase_id": candidate.purchase_id}
        if code == "conflicting_unconfirmed_claim":
            issue["field"] = candidate.field
        if not any(
            known["code"] == issue["code"]
            and known["event_id"] == issue["event_id"]
            and known["purchase_id"] == issue["purchase_id"]
            for known in issues
        ):
            issues.append(issue)
    return allowed, issues
