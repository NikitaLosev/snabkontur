"""Gold-предложения через CRM и логику действий раннера дают снимки gold.checks.

Это проверка семантики CRM и раннера, а не качества системы: вместо модели здесь
оракул, который возвращает предложения gold в эпизоде, где они доступны. Test закрыт
до КТ-4, поэтому сбой записи посреди пакета проверяется отдельным синтетическим кейсом.
"""

import json

import pytest

from snab.contracts import SCORER, SystemOutput
from snab.run import DATASETS, CaseReplay, read_cases

# Ограничения харнесса, а не ошибки gold: оракул такие кейсы не проходит.
KNOWN_HARNESS_GAPS = {
    "PZ03": "rebuild_summary берет основания только из журнала CRM; бюджет из initial_state "
    "подтверждает источник PZ03-E0, но фрагмент цитаты код без модели не выбирает",
}
INITIAL_STATE = {"purchases": {"P-1": {"revision": 1, "fields": {"budget": "5000000.00"}}}, "tasks": {}}
APPLIED_STATE = {
    "purchases": {"P-1": {"revision": 2, "fields": {"budget": "4000000.00", "currency": "RUB"}}},
    "tasks": {},
}
APPROVE = {"kind": "approve", "proposal_id": "p1", "expected_revision": {"P-1": 1}, "change_id": "c1"}
EVIDENCE = [{"event_id": "e1", "quote": "бюджет 4 млн рублей"}]
COMMIT_FAULT_CASE = {
    "id": "SYN-COMMIT",
    "schema_version": "0.1",
    "dataset_version": "synthetic",
    "input": {
        "initial_state": INITIAL_STATE,
        "events": [{"id": "e1", "occurred_at": "2026-10-01T10:00:00+03:00", "text": "По P-1 бюджет 4 млн рублей."}],
        "user_actions": [dict(APPROVE, id="a1"), dict(APPROVE, id="a2")],
        "fault_injections": [{"at": "a1", "operation": "commit", "fault": "fail_after_first_operation"}],
    },
    "gold": {
        "proposals": [
            {
                "id": "p1",
                "based_on": {"P-1": 1},
                "operations": [
                    {
                        "op": "set_field",
                        "purchase_id": "P-1",
                        "field": "budget",
                        "value": "4000000.00",
                        "evidence": EVIDENCE,
                    },
                    {
                        "op": "set_field",
                        "purchase_id": "P-1",
                        "field": "currency",
                        "value": "RUB",
                        "evidence": EVIDENCE,
                    },
                ],
            }
        ],
        "checks": [
            {"after": "a1", "expected_state": INITIAL_STATE},
            {"after": "a2", "expected_state": APPLIED_STATE},
        ],
        "expected_issues": [{"code": "write_failed", "after": "a1"}],
        "forbidden_actions": [],
        "answers": [],
    },
}


class GoldOracle:
    """Возвращает предложения gold в их эпизоде; ответы на запросы не дает."""

    name = "gold_oracle"

    def __init__(self, case: dict) -> None:
        self.case = case

    def describe(self) -> dict:
        """Оракулу нечего описывать."""
        return {"system": self.name}

    def process_episode(self, context, tools) -> SystemOutput:
        """Предложения gold, доступные после этого эпизода."""
        proposals = [
            proposal
            for proposal in self.case["gold"]["proposals"]
            if SCORER.phase_for(self.case, proposal) == context.episode_id
        ]
        return SystemOutput(proposals=json.loads(json.dumps(proposals)))

    def answer_query(self, context, query, tools) -> SystemOutput:
        """Запросы в этой проверке не оцениваются."""
        return SystemOutput()


def replay(case: dict) -> dict:
    """Строка predictions прогона gold-оракула."""
    return CaseReplay(case, GoldOracle(case), run_id="golden").run()


def replay_mismatches(case: dict) -> list[str]:
    """Контрольные точки, где снимок раннера не совпал с gold."""
    actual = {check["after"]: check for check in replay(case)["gold"]["checks"]}
    mismatches = []
    for check in case["gold"]["checks"]:
        found = actual.get(check["after"], {})
        same = SCORER.equal(check["expected_state"], found.get("expected_state"))
        if "expected_derived_views" in check:
            same &= SCORER.equal(check["expected_derived_views"], found.get("expected_derived_views"))
        if not same:
            mismatches.append(check["after"])
    return mismatches


DEV_CASES = read_cases(DATASETS["dev"])


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case["id"],
            marks=[pytest.mark.xfail(reason=KNOWN_HARNESS_GAPS[case["id"]], strict=True)]
            if case["id"] in KNOWN_HARNESS_GAPS
            else [],
        )
        for case in DEV_CASES
    ],
)
def test_gold_replay_matches_checkpoints(case: dict) -> None:
    assert replay_mismatches(case) == []


def test_commit_fault_leaves_crm_unchanged_until_next_approve() -> None:
    prediction = replay(COMMIT_FAULT_CASE)

    assert replay_mismatches(COMMIT_FAULT_CASE) == []
    assert prediction["gold"]["expected_issues"] == [{"code": "write_failed", "after": "a1"}]
