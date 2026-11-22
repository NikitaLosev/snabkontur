#!/bin/zsh
# Прогоны ступени S1.1: агент на 90 dev-кейсах без памяти и 7 сценариев по 3 повтора
set -u
cd /Users/nikitalosev/Documents/snabkontur
PY=.venv/bin/python
LOG=runs/kt3-s11.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log start
$PY -m snab.run --system agent --split dev --ids eval/ids_dev_nonmemory.txt --out runs/dev-agent-s11 >> $LOG 2>&1; log "dev-agent-s11 done rc=$?"
$PY -m snab.run --system agent --split dev --ids eval/scenarios_kt3.txt --repeat 3 --out runs/scenarios-agent-s11 >> $LOG 2>&1; log "scenarios-agent-s11 done rc=$?"
log finish
