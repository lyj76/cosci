#!/usr/bin/env bash
# Quick progress inspector for Co-Scientist OA Pipeline

LOG_FILE="/mnt/c/Users/27456/Desktop/cosci/cosci/logs/current_run.log"

echo "================================================================"
echo " CO-SCIENTIST OA DISCOVERY PIPELINE - STATUS MONITOR"
echo "================================================================"

PID=$(pgrep -f "co_scientist.orchestrator.oa_pipeline" | head -n 1)
if [ -n "$PID" ]; then
    echo " [PROCESS] Status: RUNNING (PID: $PID)"
    ps -p "$PID" -o %cpu,%mem,etime,cmd --no-headers | awk '{print "           CPU: "$1"%, MEM: "$2"%, Elapsed: "$3}'
else
    echo " [PROCESS] Status: COMPLETED or STOPPED"
fi

echo ""
echo " [OUTPUT FILES] in data/runs/oa_discovery:"
ls -lh /mnt/c/Users/27456/Desktop/cosci/cosci/data/runs/oa_discovery/ 2>/dev/null || echo "   (No output files yet)"

echo ""
echo " [RECENT LOG ENTRIES] (Last 20 lines):"
echo "----------------------------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -n 20 "$LOG_FILE"
else
    echo "Log file not found at $LOG_FILE"
fi
echo "----------------------------------------------------------------"
echo "Tip: Run 'tail -f logs/current_run.log' to watch live stream."
echo "================================================================"
