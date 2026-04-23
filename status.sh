#!/bin/bash
# Check status of arXiv Paper Tracker services

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$PROJECT_ROOT/.services.pid"
LOG_DIR="$PROJECT_ROOT/logs"

echo "=== arXiv Paper Tracker Status ==="
echo

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Services are not running."
    exit 0
fi

read APP_PID SCHED_PID < "$PID_FILE"

echo "PIDs from file: Streamlit=$APP_PID, Scheduler=$SCHED_PID"
echo

# Check status
if kill -0 "$APP_PID" 2>/dev/null; then
    echo "✓ Streamlit dashboard is running (PID $APP_PID)"
else
    echo "✗ Streamlit dashboard is NOT running"
fi

if kill -0 "$SCHED_PID" 2>/dev/null; then
    echo "✓ Scheduler is running (PID $SCHED_PID)"
else
    echo "✗ Scheduler is NOT running"
fi

echo
echo "Recent logs:"
if [ -f "$LOG_DIR/app.log" ]; then
    echo "--- Last 10 lines of app.log ---"
    tail -n 10 "$LOG_DIR/app.log"
    echo
fi
if [ -f "$LOG_DIR/scheduler.log" ]; then
    echo "--- Last 10 lines of scheduler.log ---"
    tail -n 10 "$LOG_DIR/scheduler.log"
fi
