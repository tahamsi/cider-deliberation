#!/usr/bin/env bash
set -u

OUT_DIR="${1:-outputs/a100_full_v10}"
PID_FILE="$OUT_DIR/run.pid"
LOG_FILE="$OUT_DIR/run.log"

echo "=== CIDeR A100 Campaign Status ==="
echo "Time: $(date)"
echo "Out dir: $OUT_DIR"
echo

if [ ! -d "$OUT_DIR" ]; then
  echo "STATUS: NOT FOUND"
  echo "Output directory does not exist: $OUT_DIR"
  exit 2
fi

echo "=== Process ==="
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  echo "PID file: $PID"

  if ps -p "$PID" > /dev/null 2>&1; then
    echo "STATUS: RUNNING"
    ps -p "$PID" -o pid,ppid,stat,wchan:32,etime,time,%cpu,%mem,cmd
  else
    echo "Process from PID file is not running."
  fi
else
  echo "No PID file found: $PID_FILE"
  PID=""
fi

echo
echo "=== Log health ==="
if [ -f "$LOG_FILE" ]; then
  stat -c "run.log size=%s bytes modified=%y" "$LOG_FILE"

  CALLS="$(grep -c '\[ollama_call\]' "$LOG_FILE" || true)"
  ERRORS="$(grep -c '\[ollama_error\]' "$LOG_FILE" || true)"
  TRACEBACKS="$(grep -c 'Traceback\|RuntimeError\|Exception\|Error' "$LOG_FILE" || true)"

  echo "ollama_call lines:  $CALLS"
  echo "ollama_error lines: $ERRORS"
  echo "error-like lines:   $TRACEBACKS"

  echo
  echo "--- Last 30 log lines ---"
  tail -n 30 "$LOG_FILE"
else
  echo "No log file found: $LOG_FILE"
fi

echo
echo "=== Output files ==="
find "$OUT_DIR" -maxdepth 8 -type f \
  -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | sort | tail -40

echo
echo "=== Raw JSONL counts ==="
find "$OUT_DIR/main" -path "*/raw/*.jsonl" -type f -print0 2>/dev/null | xargs -0 -r wc -l || true

echo
echo "=== Final artifacts ==="
SUMMARY="$OUT_DIR/a100_campaign_summary.md"
RESULTS="$OUT_DIR/a100_campaign_results.csv"
MANIFEST="$OUT_DIR/a100_campaign_manifest.json"

FOUND_FINAL=0

for f in "$SUMMARY" "$RESULTS" "$MANIFEST"; do
  if [ -f "$f" ]; then
    echo "FOUND: $f"
    FOUND_FINAL=$((FOUND_FINAL + 1))
  else
    echo "MISSING: $f"
  fi
done

echo
echo "=== Ollama ==="
if command -v ollama >/dev/null 2>&1; then
  ollama ps || true
else
  echo "ollama command not found"
fi

echo
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "=== Verdict ==="

RUNNING=0
if [ -n "${PID:-}" ] && ps -p "$PID" > /dev/null 2>&1; then
  RUNNING=1
fi

if [ "$RUNNING" -eq 1 ]; then
  echo "RUNNING: campaign process is alive."
  exit 0
fi

if [ "$FOUND_FINAL" -eq 3 ]; then
  if [ -f "$LOG_FILE" ] && grep -q '\[ollama_error\]\|Traceback\|RuntimeError' "$LOG_FILE"; then
    echo "COMPLETED WITH WARNINGS: final files exist, but log contains errors. Inspect run.log."
    exit 1
  else
    echo "COMPLETED SUCCESSFULLY: final summary/results/manifest exist and process has exited."
    exit 0
  fi
fi

if [ -f "$LOG_FILE" ] && grep -q 'Traceback\|RuntimeError\|Ollama request failed\|\[ollama_error\]' "$LOG_FILE"; then
  echo "FAILED OR INTERRUPTED: process is not running and log contains errors."
  exit 1
fi

echo "UNKNOWN / POSSIBLY INTERRUPTED: process is not running, but final artifacts are incomplete."
exit 3
