PYTHON ?= .venv/bin/python
SYSTEM ?= rules
SPLIT ?= dev
REPEAT ?= 3
# Сохраненные прогоны КТ-3 лежат в runs/, новый прогон пишется рядом, чтобы его можно было сверить
OUT ?= runs/repro
# Многошаговые сценарии КТ-3 по одному на обязательный тип project.md: второй шаг зависит
# от первого (D-WOR-01), сбой поиска (D-WOR-16), лимит шагов (D-WOR-17), старое письмо
# после нового (D-WOR-04), повтор (D-WOR-07), откат (D-WOR-06), неизвестный голос (D-WOR-14)
SCENARIOS_FILE := eval/scenarios_kt3.txt
# 90 кейсов dev без memory: длинные истории memory идут отдельной целью
NONMEMORY_IDS := eval/ids_dev_nonmemory.txt
# S1.2, S1 и B1 по умолчанию; make scenarios SYSTEM=rules гоняет одну систему
SCENARIO_SYSTEMS ?= hybrid agent llm_single
ifeq ($(origin SYSTEM),command line)
SCENARIO_SYSTEMS := $(SYSTEM)
endif

.PHONY: venv lint test baseline eval dev memory scenarios noisy report

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check snab tests
	$(PYTHON) -m ruff format --check snab tests

test:
	$(PYTHON) -m pytest

baseline:
	$(PYTHON) -m snab.run --system rules --split dev --out $(OUT)/kt3-ref

eval:
	$(PYTHON) -m snab.run --system $(SYSTEM) --split $(SPLIT) --out $(OUT)/eval-$(SYSTEM)-$(SPLIT)

dev:
	$(PYTHON) -m snab.run --system $(SYSTEM) --split dev --ids $(NONMEMORY_IDS) --out $(OUT)/dev-$(SYSTEM)

memory:
	$(PYTHON) -m snab.run --system $(SYSTEM) --split dev --category memory --out $(OUT)/dev-$(SYSTEM)-memory

scenarios:
	for system in $(SCENARIO_SYSTEMS); do \
		$(PYTHON) -m snab.run --system $$system --split dev --ids $(SCENARIOS_FILE) --repeat $(REPEAT) \
			--out $(OUT)/scenarios-$$system || exit 1; \
	done

noisy:
	$(PYTHON) -m snab.run --system $(SYSTEM) --cases eval/dev_noisy.jsonl --out $(OUT)/noisy-$(SYSTEM)

# make report RUNS="runs/kt3-ref runs/eval-agent-dev"
report:
	$(PYTHON) -m snab.report $(RUNS) --out $(OUT)/REPORT.md
