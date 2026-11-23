# СнабКонтур

Проект по курсу ["Инженерия LLM-систем"](https://wiki.cs.hse.ru/%D0%98%D0%BD%D0%B6%D0%B5%D0%BD%D0%B5%D1%80%D0%B8%D1%8F_LLM-%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC)

Тема: "СнабКонтур" - система формирования и актуализации досье закупок из переписок, документов и встреч

Локальный агент разбирает записи встреч, переписку и документы и предлагает изменения карточек закупок в CRM: цены, сроки, количества и задачи. Каждая правка показана как дифф с прежним значением, новым и цитатой из источника

Закупщик, который ведет закупку, проверяет и подтверждает дифф. После этого код записывает изменения в CRM. Такая проверка нужна, поскольку ошибка модели в цене или сроке может стоить денег. Повторная обработка не создает дублей, изменения можно откатить. Каждое утро по сохраненным карточкам и задачам собирается сводка сроков и поставок

## Команда

- Ефимов Павел Дмитриевич, 249
- Лосев Никита Эдуардович, 245
- Фадеев Павел Викторович, 243

## Что где лежит

- [`design-doc/`](design-doc/README.md) - дизайн-документ: контракт, бюджеты, схема, память, eval, риски
- [`adr/`](adr/001-search-or-fixed-pipeline.md) - ADR, записанные архитектурные решения: что выбрали, из каких вариантов и когда пересмотрим
- [`eval/`](eval/README.md) - наборы и методика оценки
- [`snab/`](snab/) - харнесс: раннер, mock-CRM, инструменты, системы, отчет и страница хронометража
- [`TODO.md`](TODO.md) - задачи и контрольные точки

## Запуск харнесса

Нужен Python 3.12 или новее. Для LLM-систем нужен [Ollama](https://ollama.com) на `localhost:11434` с моделью:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Системы лестницы из раздела 8 design doc:

| `--system` | Ступень | Что делает |
|---|---|---|
| `rules` | B0 | Регулярные выражения, без модели |
| `llm_single` | B1, B1.1 | Одно извлечение моделью на кусок, без инструментов. Текущий код - B1.1, выбранная по ADR-001; B1 снята на теге `kt3-s1` |
| `agent` | S1, S1.1 | То же извлечение, поиск закупки и прежних условий решает модель, не больше 2 вызовов инструментов на кусок |
| `hybrid` | S1.2 | Поиск, привязку и прежние условия решает код, модель получает кандидатов фактов от правил B0 |

```bash
make venv                             # .venv с requirements.txt и requirements-dev.txt
make lint                             # ruff check и ruff format --check
make test                             # CRM, инструменты, системы, golden replay и смоук раннера
make baseline                         # B0 по всему dev -> runs/repro/kt3-ref, сверка с runs/kt3-ref
make dev SYSTEM=llm_single            # 90 кейсов dev без memory -> runs/repro/dev-llm_single
make memory SYSTEM=hybrid             # 10 историй memory -> runs/repro/dev-hybrid-memory
make scenarios REPEAT=3               # 7 сценариев КТ-3 для hybrid, agent и llm_single по 3 повтора
make noisy SYSTEM=hybrid              # 15 шумных вариантов кейсов dev
make report RUNS="runs/dev-agent-s11 runs/dev-llm_single-s11"  # runs/repro/REPORT.md: срезы, 95% ДИ, разница пар
python -m snab.run --system rules --split dev --category workflow --out runs/rules-workflow
```

Хронометраж человека: `python -m snab.timing --predictions runs/kt3-ref/predictions.jsonl` открывает локальную страницу с 6 кейсами dev. Сводку замеров КТ-3 пересчитывает `python -m snab.timing --summary --out-dir runs/kt3-ref/timing`

В папке прогона: `predictions.jsonl` в формате набора, `trace.jsonl` с шагами, `meta.json` с версиями, хэшами кода и промптов, `summary.json` со временем, вызовами и стоимостью, `eval-report.json` от `eval/score.py --judge off`. Кейс, до инъекции сбоя которого система не дошла, отмечен в отчете как `unsupported`. Папка `--out` должна быть пустой, поэтому цели make пишут в `runs/repro/` (меняется через `OUT=...`), а сохраненные прогоны КТ-3 лежат в `runs/` рядом для сверки. В git эталонный прогон `runs/kt3-ref` с замерами хронометража хранится целиком, у остальных прогонов КТ-3 - только `meta.json`, `summary.json` и `eval-report*.json`. Test без флага `--i-know-this-is-heldout` не запускается
