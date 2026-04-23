#!/bin/bash
# Stop arXiv Paper Tracker services

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$PROJECT_ROOT/.services.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Services are not running."
    exit 0
fi

read APP_PID SCHED_PID < "$PID_FILE"

echo "Stopping arXiv Paper Tracker services..."
echo "  Streamlit dashboard (PID $APP_PID)..."
kill -TERM "$APP_PID" 2>/dev/null
echo "  Scheduler (PID $SCHED_PID)..."
kill -TERM "$SCHED_PID" 2>/dev/null

# Wait a moment
sleep 1

# Check if any are still running and force kill if needed
ALIVE=0
if kill -0 "$APP_PID" 2>/dev/null; then
    echo "  Force killing Streamlit..."
    kill -9 "$APP_PID" 2>/dev/null
    ALIVE=1
fi
if kill -0 "$SCHED_PID" 2>/dev/null; then
    echo "  Force killing Scheduler..."
    kill -9 "$SCHED_PID" 2>/dev/null
    ALIVE=1
fi

# Remove PID file
rm -f "$PID_FILE"

echo
echo "✓ All services stopped."
