"""Шумные варианты dev-кейсов: python -m eval.noise -> eval/dev_noisy.jsonl.

Ручная правка задает смысловой шум: числа словами, размытые хвосты вместо готового ответа,
опечатку в поставщике, ссылку на товар вместо номера. Поверх нее с фиксированным seed
добавляются слова-паразиты, а речь приводится к виду ASR: строчные буквы без пунктуации
и метка SPEAKER_NN. Цитаты gold переносятся на новые дословные фрагменты, значения gold
не меняются; почему gold остался верным, записано в origin.gold_note.
"""

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from snab.contracts import REPO_ROOT, SCORER

SEED = 42
DEV_PATH = REPO_ROOT / "eval" / "dev.jsonl"
NOISY_PATH = REPO_ROOT / "eval" / "dev_noisy.jsonl"
ID_SUFFIX = "-N"
VERSION_SUFFIX = "-noisy.1"
ASR_MODALITIES = frozenset({"speech_text", "meeting_transcript"})
FILLERS = ("ну", "короче", "значит", "э")
WORDS_PER_ASR_FILLER = 8
MIN_CHAT_WORDS_FOR_FILLER = 6
# Номера закупок и метки спикеров ASR со словарем терминов пишет как есть
KEEP_CASE_RE = re.compile(r"\bP-[A-Z0-9]+\b|\bSPEAKER_\d+\b")
PUNCTUATION_RE = re.compile(r"(?<!\d)[.,]|[.,](?!\d)|[;:!?…—–«»\"()]")
# После этих слов паразит рвет число, дату или предлог с существительным
PREPOSITIONS = {"по", "за", "в", "у", "к", "на", "до", "с", "через", "при"}
MONTHS = {"января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября"}
BOUND_WORDS = frozenset(PREPOSITIONS | MONTHS | {"декабря", "сто"})
NUMBER_STEMS = (
    "один",
    "одн",
    "два",
    "две",
    "двад",
    "двен",
    "три",
    "четыр",
    "пят",
    "шест",
    "сем",
    "восем",
    "девят",
    "десят",
    "сорок",
    "тысяч",
    "миллион",
)


@dataclass(frozen=True)
class EventEdit:
    """Ручная правка события: новый текст, перенос цитат, метаданные."""

    text: str | None = None
    quotes: dict[str, str] = field(default_factory=dict)
    speaker: str | None = None
    modality: str | None = None
    drop: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseEdit:
    """Правки кейса и объяснение, почему gold остался верным."""

    perturbations: tuple[str, ...]
    gold_note: str
    events: dict[str, EventEdit] = field(default_factory=dict)


EDITS: dict[str, CaseEdit] = {
    "E01": CaseEdit(
        ("number_words", "supplier_typo", "fillers", "lowercase"),
        "Значения те же: двенадцать тысяч пятьсот = 12500. Название задачи оставлено с верным "
        "именем поставщика, опечатка Дэльты относится к тексту сообщения",
        {
            "e1": EventEdit(
                "по закупке P-17 согласовали 20 насосов по двенадцать тысяч пятьсот рублей за штуку. "
                "поставка 20 октября 2026 года. Анна, запроси счёт у Дэльты до 6 октября 2026 года",
                {
                    "по 12500 рублей за штуку": "по двенадцать тысяч пятьсот рублей за штуку",
                    "12500 рублей": "двенадцать тысяч пятьсот рублей",
                    "Поставка 20 октября 2026 года": "поставка 20 октября 2026 года",
                    "Анна, запроси счёт у Дельты до 6 октября 2026 года": (
                        "Анна, запроси счёт у Дэльты до 6 октября 2026 года"
                    ),
                },
            )
        },
    ),
    "E09": CaseEdit(
        ("vague_tail", "fillers", "lowercase"),
        "Хвост 'Пока не согласовано; план 10 октября остаётся' заменен на 'там посмотрим': "
        "'постараемся' и 'посмотрим' по-прежнему не подтверждают срок, карточка не меняется",
        {"e1": EventEdit("по P-17 постараемся 12 октября привезти, там посмотрим")},
    ),
    "E10": CaseEdit(
        ("number_words", "fillers", "lowercase", "no_self_introduction"),
        "Самопредставление 'Ирина, ответственная за P-17' убрано из текста; полномочия по-прежнему "
        "заданы метаданными author и trusted_for, поэтому предложение и значения gold не меняются",
        {
            "e1": EventEdit(
                "по P-17 старые условия больше не действуют, переподписали. "
                "подтверждаю срок 15 октября 2026 года и цену двенадцать тысяч рублей за штуку",
                {
                    "Подтверждаю срок 15 октября 2026 года": "подтверждаю срок 15 октября 2026 года",
                    "цену 12000 рублей за штуку": "цену двенадцать тысяч рублей за штуку",
                },
            )
        },
    ),
    "E15": CaseEdit(
        ("asr", "speaker_labels", "vague_tail", "implicit_year", "fillers"),
        "Имена Анна и Борис заменены метками спикеров, хвост 'окончательного согласования ещё нет' "
        "убран. Противоречие осталось: два подтверждения разных дат по P-17 без разрешения. "
        "Год 2026 следует из occurred_at. В кейсе нет authority_registry, как и у родителя, "
        "поэтому отдельный unknown_speaker_authority не добавлен: спорную дату и так решает закупщик",
        {
            "e1": EventEdit(
                "По P-17 подтверждаем поставку 12 октября.",
                speaker="SPEAKER_01",
                modality="speech_text",
                drop=("author",),
            ),
            "e2": EventEdit(
                "Не, по P-17 у нас 13 октября стоит, подтверждаем тринадцатое.",
                speaker="SPEAKER_02",
                modality="speech_text",
                drop=("author",),
            ),
        },
    ),
    "E21": CaseEdit(
        ("asr", "speaker_labels", "vague_tail", "fillers"),
        "Хвост 'Вариант с оплатой после получения тогда убираем' заменен на 'оформляйте': решение "
        "по цене и полной предоплате осталось явным, значения gold те же",
        {
            "e1": EventEdit(speaker="SPEAKER_01"),
            "e2": EventEdit(speaker="SPEAKER_02"),
            "e3": EventEdit(
                "Да, по P-17 берём по двенадцать тысяч рублей за штуку, всю сумму платим до отгрузки. "
                "Ну всё, оформляйте.",
                speaker="SPEAKER_01",
            ),
        },
    ),
    "D-EXT-01": CaseEdit(
        ("number_words", "fillers", "lowercase"),
        "двенадцать тысяч четыреста = 12400, доставка 6000 рублей сверху = 6000.00",
        {
            "D-EXT-01-e1": EventEdit(
                "по P-D101 берём восемнадцать… а нет, двадцать штук. "
                "по двенадцать тысяч четыреста рублей за штуку, доставка ещё 6000 рублей сверху"
            )
        },
    ),
    "D-EXT-03": CaseEdit(
        ("implicit_year", "implicit_month", "vague_tail", "fillers", "lowercase"),
        "'21-го' после '18 ноября' означает 21 ноября, год 2026 следует из occurred_at 2026-11-05; "
        "'логист ок' сохраняет согласие логиста",
        {"D-EXT-03-e1": EventEdit("по P-D103 отгрузка 18 ноября, к нам приедет 21-го, логист ок")},
    ),
    "D-DEC-01": CaseEdit(
        ("number_words", "vague_tail", "fillers", "lowercase"),
        "Ответ на скидку стал размытым 'с авансом там посмотрим': аванс по-прежнему не принят, цена не меняется",
        {
            "D-DEC-01-e1": EventEdit("по P-D201 до двенадцати тысяч за штуку опустимся, если всю сумму вперёд"),
            "D-DEC-01-e2": EventEdit("с авансом там посмотрим"),
        },
    ),
    "D-ENT-01": CaseEdit(
        ("supplier_typo", "implicit_year", "fillers", "lowercase"),
        "Опечатка Паурса вместо Паруса; товар 'кабель' по-прежнему однозначно указывает на P-D301, "
        "год 2026 следует из occurred_at",
        {"D-ENT-01-e1": EventEdit("у Паурса по кабелю сдвинули на 17 ноября, согласовали")},
    ),
    "D-ENT-03": CaseEdit(
        ("implicit_year", "fillers", "lowercase"),
        "Поставщик написан со строчной буквы, номера и товара по-прежнему нет: обе закупки Ритма остаются кандидатами",
        {"D-ENT-03-e1": EventEdit("по ритму машину согласовали на 20 ноября")},
    ),
    "D-TRU-01": CaseEdit(
        ("asr", "speaker_labels", "number_words", "fillers"),
        "Голос по-прежнему неизвестен (SPEAKER_01, нет в реестре), лимит Марины миллион = 1000000.00",
        {
            "D-TRU-01-e1": EventEdit("по P-D401 финотдел лимит подтвердил, миллион рублей"),
            "D-TRU-01-e2": EventEdit(
                "По P-D401 лимит теперь восемьсот тысяч рублей, на совещании так решили.", speaker="SPEAKER_01"
            ),
        },
    ),
    "D-TRU-03": CaseEdit(
        ("number_words", "fillers", "lowercase"),
        "Подтвержденный Мариной лимит 'в миллион рублей' = 1000000.00; инструкция в документе не тронута",
        {"D-TRU-03-e1": EventEdit("по P-D403 финотдел подтвердил лимит в миллион рублей")},
    ),
    "D-WOR-01": CaseEdit(
        ("item_reference", "asr", "speaker_labels", "number_words", "fillers"),
        "В деле одна закупка насосов у Дельты, ссылка по товару однозначна; двенадцать тысяч = 12000. "
        "Реестра полномочий нет, как и у родителя, метки спикеров решение не меняют",
        {
            "D-WOR-01-e1": EventEdit(
                "насосы у Дельты: 10 штук по 13000 рублей, всего 130000 рублей. склад ждёт 15 ноября 2026 года"
            ),
            "D-WOR-01-e2": EventEdit("по насосам склад просит паспорт изделия, в папке пока только счёт"),
            "D-WOR-01-e3": EventEdit("При полной предоплате могу по двенадцать тысяч отдать.", speaker="SPEAKER_02"),
            "D-WOR-01-e4": EventEdit(
                "Финансы подтвердили, платим всё до отгрузки. По двенадцать тысяч рублей берём.", speaker="SPEAKER_01"
            ),
        },
    ),
    "D-WOR-02": CaseEdit(
        ("item_reference", "asr", "speaker_labels", "number_words", "vague_tail", "fillers"),
        "Ссылка 'клапаны у Паруса' однозначна; 'аванс не пропустили, всё как было' по-прежнему "
        "отклоняет скидку, цена и оплата не меняются",
        {
            "D-WOR-02-e1": EventEdit(
                "клапаны у Паруса: 10 штук по 13000 рублей, всего 130000 рублей. склад ждёт 15 ноября 2026 года"
            ),
            "D-WOR-02-e3": EventEdit("С авансом будет по четыре пятьсот за клапан.", speaker="SPEAKER_02"),
            "D-WOR-02-e4": EventEdit("Аванс финансы не пропустили, так что всё как было.", speaker="SPEAKER_01"),
        },
    ),
    "D-WOR-14": CaseEdit(
        ("item_reference", "asr", "speaker_labels", "number_words", "fillers", "lowercase"),
        "Голос SPEAKER_02 по-прежнему вне реестра; Марина подтверждает лимит 400 тысяч = 400000.00",
        {
            "D-WOR-14-e1": EventEdit(
                "редукторы у Оси: 10 штук по 13000 рублей, всего 130000 рублей. склад ждёт 15 ноября 2026 года"
            ),
            "D-WOR-14-e3": EventEdit("Лимит по этому заказу теперь триста тысяч рублей.", speaker="SPEAKER_02"),
            "D-WOR-14-e4": EventEdit("глянула решение финансов, лимит 400 тысяч рублей, письмо скину"),
        },
    ),
}


def asr_text(text: str) -> str:
    """Текст -> вид расшифровки ASR: строчные буквы, без пунктуации, кроме номеров закупок."""
    kept = {match.start(): match.group() for match in KEEP_CASE_RE.finditer(text)}
    lowered = text.lower()
    for start, token in kept.items():
        lowered = lowered[:start] + token + lowered[start + len(token) :]
    return " ".join(PUNCTUATION_RE.sub("", lowered).split())


def is_filler_slot(previous: str) -> bool:
    """Можно ли вставить паразит после этого слова, не разорвав число или предлог."""
    word = previous.lower().strip(".,;:!?…")
    return word not in BOUND_WORDS and not any(char.isdigit() for char in word) and not word.startswith(NUMBER_STEMS)


def insert_fillers(text: str, spans: list[list[int]], asr: bool, rng: random.Random) -> str:
    """Вставляет слова-паразиты перед словами и сдвигает границы цитат [start, end)."""
    words = list(re.finditer(r"\S+", text))
    word_starts = [word.start() for prev, word in zip(words, words[1:], strict=False) if is_filler_slot(prev.group())]
    count = 1 + len(words) // WORDS_PER_ASR_FILLER if asr else int(len(words) >= MIN_CHAT_WORDS_FOR_FILLER)
    for position in sorted(rng.sample(word_starts, min(count, len(word_starts))), reverse=True):
        filler = rng.choice(FILLERS) + (" " if asr else ", ")
        text = text[:position] + filler + text[position:]
        for span in spans:
            span[0] += len(filler) if position <= span[0] else 0
            span[1] += len(filler) if position < span[1] else 0
    return text


def event_quotes(gold: dict) -> list[dict]:
    """Все ссылки gold вида {event_id, quote} на любой глубине."""
    found = []
    stack = [gold]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "event_id" in node and "quote" in node:
                found.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def perturb_event(event: dict, edit: EventEdit, quotes: list[str], rng: random.Random) -> dict[str, str]:
    """Меняет событие на месте; возвращает перенос цитат старый текст -> новый."""
    original = event["text"]
    text = edit.text or original
    moved = {quote: edit.quotes.get(quote, text if quote == original else quote) for quote in quotes}
    if edit.modality:
        event["modality"] = edit.modality
    asr = event.get("modality") in ASR_MODALITIES
    if asr:
        text = asr_text(text)
        moved = {old: asr_text(new) for old, new in moved.items()}
    spans = {}
    for old, new in moved.items():
        start = text.find(new)
        if start < 0:
            raise ValueError(f"{event['id']}: quote {new!r} is not in {text!r}")
        spans[old] = [start, start + len(new)]
    if event.get("modality") != "document_text":
        text = insert_fillers(text, list(spans.values()), asr, rng)
    if edit.speaker:
        event["speaker_id"] = edit.speaker
        label = f"{edit.speaker}: "
        text = label + text
        spans = {old: [start + len(label), end + len(label)] for old, (start, end) in spans.items()}
    for key in edit.drop:
        del event[key]
    event["text"] = text
    return {old: text[start:end] for old, (start, end) in spans.items()}


def case_events(case: dict) -> dict[str, dict]:
    """Все события кейса по id в форматах 0.1 и 0.2."""
    data = case["input"]
    steps = data.get("timeline") or [{"events": data.get("events", [])}]
    return {event["id"]: event for step in steps for event in step.get("events", [])}


def make_noisy(parent: dict, edit: CaseEdit, seed: int) -> dict:
    """Родительский кейс -> шумный вариант с перенесенными цитатами gold."""
    case = copy.deepcopy(parent)
    references = event_quotes(case["gold"])
    for event_id, event in case_events(case).items():
        quotes = sorted({ref["quote"] for ref in references if ref["event_id"] == event_id})
        rng = random.Random(f"{seed}:{parent['id']}:{event_id}")
        moved = perturb_event(event, edit.events.get(event_id, EventEdit()), quotes, rng)
        for ref in references:
            if ref["event_id"] == event_id:
                ref["quote"] = moved[ref["quote"]]
    unknown = set(edit.events) - set(case_events(case))
    if unknown:
        raise ValueError(f"{parent['id']}: edits for unknown events {sorted(unknown)}")
    case["id"] = parent["id"] + ID_SUFFIX
    case["dataset_version"] = parent["dataset_version"] + VERSION_SUFFIX
    case["tags"] = [*parent.get("tags", []), "noisy"]
    case["origin"] = {
        "kind": "synthetic_noise",
        "parent_id": parent["id"],
        "seed": seed,
        "perturbations": list(edit.perturbations),
        "gold_note": edit.gold_note,
    }
    return case


def check_case(case: dict) -> None:
    """Схема оценщика и дословность цитат; ошибка -> ValueError."""
    errors = SCORER.validation_errors(case, SCORER.DatasetCase)
    if errors:
        raise ValueError(f"{case['id']}: schema errors {errors}")
    events = case_events(case)
    for ref in event_quotes(case["gold"]):
        if ref["quote"] not in events[ref["event_id"]]["text"]:
            raise ValueError(f"{case['id']}: quote {ref['quote']!r} is not in {ref['event_id']}")


def build_noisy(parents: list[dict], seed: int = SEED) -> list[dict]:
    """Кейсы dev -> шумные варианты для EDITS в порядке dev, каждый проверен."""
    by_id = {case["id"]: case for case in parents}
    missing = set(EDITS) - set(by_id)
    if missing:
        raise ValueError(f"parents not in dev: {sorted(missing)}")
    cases = [make_noisy(case, EDITS[case["id"]], seed) for case in parents if case["id"] in EDITS]
    for case in cases:
        check_case(case)
    return cases


def main() -> None:
    """Пишет eval/dev_noisy.jsonl из eval/dev.jsonl."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=NOISY_PATH)
    args = parser.parse_args()
    parents = [json.loads(line) for line in DEV_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = build_noisy(parents, args.seed)
    args.out.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
