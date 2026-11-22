#!/bin/zsh
# Прогоны ступени S1.2 (hybrid) после окончания прогонов S1.1: одна модель в памяти, без параллели
set -u
cd /Users/nikitalosev/Documents/snabkontur
PY=.venv/bin/python
LOG=runs/kt3-s12.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q finish runs/kt3-s11.log 2>/dev/null; do sleep 30; done
log start
$PY -m snab.run --system hybrid --split dev --ids eval/ids_dev_nonmemory.txt --out runs/dev-hybrid >> $LOG 2>&1; log "dev-hybrid done rc=$?"
$PY -m snab.run --system hybrid --split dev --ids eval/scenarios_kt3.txt --repeat 3 --out runs/scenarios-hybrid >> $LOG 2>&1; log "scenarios-hybrid done rc=$?"
$PY -m snab.run --system hybrid --cases eval/dev_noisy.jsonl --out runs/noisy-hybrid >> $LOG 2>&1; log "noisy-hybrid done rc=$?"
$PY -m snab.run --system hybrid --split dev --category memory --out runs/dev-hybrid-memory >> $LOG 2>&1; log "memory-hybrid done rc=$?"
log finish
