"""Точное число токенов Qwen3 в историях memory: python -m eval.history_tokens -> eval/history_tokens.json.

Текст истории собирается как для нижней границы в history_measurement набора: NFC, уникальные
source_id, тексты через два перевода строки. Токены считает токенизатор из GGUF модели в Ollama:
сырой промпт без шаблона, prompt_eval_count. Набор не меняется, он заморожен на КТ-2.
"""

import argparse
import json
import unicodedata
import urllib.request
from pathlib import Path

from snab.contracts import REPO_ROOT
from snab.systems.llm import MODEL, OLLAMA_URL

DATASETS = (REPO_ROOT / "eval" / "dev.jsonl", REPO_ROOT / "eval" / "test.jsonl")
OUT_PATH = REPO_ROOT / "eval" / "history_tokens.json"
# Самая длинная история - около 26 тыс. токенов; окно с запасом, чтобы Ollama не обрезал промпт
COUNT_CTX = 40960
# Ollama берет из кэша общее начало промпта и не считает его в prompt_eval_count.
# Короткий промпт с другим первым токеном между подсчетами сбрасывает это общее начало
CACHE_BREAKER = "§"
TIMEOUT_S = 900


def history_text(case: dict) -> str:
    """Кейс memory -> текст уникальных источников через два перевода строки."""
    texts = {}
    for episode in case["input"]["timeline"]:
        for event in episode["events"]:
            texts.setdefault(event["source_id"], event["text"])
    return unicodedata.normalize("NFC", "\n\n".join(texts.values()))


def count_tokens(text: str) -> int:
    """Текст -> число токенов Qwen3 по prompt_eval_count Ollama."""
    body = {
        "model": MODEL,
        "prompt": text,
        "raw": True,
        "stream": False,
        "options": {"num_predict": 1, "num_ctx": COUNT_CTX, "temperature": 0},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())["prompt_eval_count"]


def measure(case: dict) -> dict:
    """Кейс memory -> точное число токенов рядом с нижней границей из набора."""
    count_tokens(CACHE_BREAKER)
    tokens = count_tokens(history_text(case))
    if tokens >= COUNT_CTX:
        raise ValueError(f"{case['id']}: {tokens} tokens hit num_ctx {COUNT_CTX}, the prompt may be truncated")
    bound = case["input"]["history_measurement"].get("unicode_letter_runs_lower_bound")
    return {"id": case["id"], "exact_tokens": tokens, "lower_bound": bound, "exceeds_8192": tokens > 8192}


def read_memory_cases() -> list[dict]:
    """Кейсы dev и test с history_measurement."""
    cases = []
    for path in DATASETS:
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            if "history_measurement" in case["input"]:
                cases.append(case)
    return cases


def main() -> None:
    """Подсчет по всем историям memory и запись результата."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    rows = [measure(case) for case in read_memory_cases()]
    report = {"model": MODEL, "method": "ollama /api/generate raw, prompt_eval_count", "histories": rows}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['id']:10} {row['exact_tokens']:>6} (нижняя граница {row['lower_bound']})")


if __name__ == "__main__":
    main()
