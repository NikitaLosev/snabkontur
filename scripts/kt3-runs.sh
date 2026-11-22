#!/bin/zsh
# Долгие прогоны КТ-3 по очереди: одна модель в памяти, без параллели
set -u
cd /Users/nikitalosev/Documents/snabkontur
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> runs/kt3-chain.log; }
log start
make scenarios >> runs/kt3-chain.log 2>&1; log "scenarios done rc=$?"
$PY -m snab.run --system agent --split dev --ids eval/ids_dev_nonmemory.txt --out runs/dev-agent >> runs/kt3-chain.log 2>&1; log "dev-agent done rc=$?"
$PY -m snab.run --system llm_single --split dev --ids eval/ids_dev_nonmemory.txt --out runs/dev-llm_single >> runs/kt3-chain.log 2>&1; log "dev-llm_single done rc=$?"
$PY -m snab.run --system agent --cases eval/dev_noisy.jsonl --out runs/noisy-agent >> runs/kt3-chain.log 2>&1; log "noisy-agent done rc=$?"
$PY -m snab.run --system agent --split dev --category memory --out runs/dev-agent-memory >> runs/kt3-chain.log 2>&1; log "memory-agent done rc=$?"
log finish
