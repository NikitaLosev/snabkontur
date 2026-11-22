PYTHON ?= .venv/bin/python
SYSTEM ?= rules
SPLIT ?= dev
REPEAT ?= 3
# Многошаговые сценарии КТ-3: зависимый шаг, сбой поиска, лимит шагов, старое письмо,
# повтор, исправление и откат, неизвестный голос, ручная правка поверх диффа
SCENARIOS := D-WOR-01 D-WOR-16 D-WOR-17 D-WOR-04 D-WOR-07 D-WOR-06 D-WOR-14 D-WOR-18

.PHONY: venv lint test baseline eval scenarios

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check snab tests
	$(PYTHON) -m ruff format --check snab tests

test:
	$(PYTHON) -m pytest

baseline:
	$(PYTHON) -m snab.run --system rules --split dev --out runs/kt3-ref

eval:
	$(PYTHON) -m snab.run --system $(SYSTEM) --split $(SPLIT) --out runs/eval-$(SYSTEM)-$(SPLIT)

scenarios:
	mkdir -p runs
	printf '%s\n' $(SCENARIOS) > runs/scenario-ids.txt
	$(PYTHON) -m snab.run --system $(SYSTEM) --split dev --ids runs/scenario-ids.txt --repeat $(REPEAT) \
		--out runs/scenarios-$(SYSTEM)
