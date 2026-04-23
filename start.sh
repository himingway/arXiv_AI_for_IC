#!/bin/bash
# Start arXiv Paper Tracker services
# - Web dashboard (Streamlit) in background
# - Daily sync scheduler in background

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.services.pid"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Starting arXiv Paper Tracker services..."
echo "Project root: $PROJECT_ROOT"

# Check if already running
if [ -f "$PID_FILE" ]; then
    read APP_PID SCHED_PID < "$PID_FILE"
    if kill -0 "$APP_PID" 2>/dev/null || kill -0 "$SCHED_PID" 2>/dev/null; then
        echo "Services are already running!"
        echo "  Streamlit dashboard PID: $APP_PID"
        echo "  Scheduler PID: $SCHED_PID"
        echo "Use ./stop.sh to stop them first."
        exit 1
    else
        echo "Found stale PID file, removing..."
        rm -f "$PID_FILE"
    fi
fi

# Auto detect virtual environment
if [ -d "$PROJECT_ROOT/.venv" ]; then
    VENV_DIR="$PROJECT_ROOT/.venv"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    VENV_DIR="$PROJECT_ROOT/venv"
else
    VENV_DIR=""
fi

# Activate virtual environment if found
if [ -n "$VENV_DIR" ]; then
    echo "✓ Found virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Auto detect: use uv if available (and no venv activated already)
if command -v uv >/dev/null 2>&1 && [ -z "$VENV_DIR" ]; then
    RUN_CMD="uv run"
    echo "✓ Using uv for runtime..."
else
    RUN_CMD=""
    if [ -n "$VENV_DIR" ]; then
        echo "✓ Using activated virtual environment..."
    else
        echo "✓ No venv/uv found, using system Python..."
    fi
fi

# Start Streamlit web dashboard
echo "Starting Streamlit web dashboard..."
nohup $RUN_CMD streamlit run "$PROJECT_ROOT/app.py" \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false \
    >> "$LOG_DIR/app.log" 2>&1 &
APP_PID=$!

# Start daily sync scheduler
echo "Starting daily sync scheduler..."
nohup $RUN_CMD python "$PROJECT_ROOT/main.py" scheduler \
    >> "$LOG_DIR/scheduler.log" 2>&1 &
SCHED_PID=$!

# Save PIDs
echo "$APP_PID $SCHED_PID" > "$PID_FILE"

echo
echo "✓ Services started successfully!"
echo
echo "  Local URL:  http://localhost:8501"
echo "  Network URL: http://$(hostname -I | awk '{print $1}'):8501"
echo "  Next daily sync: Check logs for details"
echo
echo "  Log files:"
echo "    - Dashboard: $LOG_DIR/app.log"
echo "    - Scheduler:  $LOG_DIR/scheduler.log"
echo
echo "  PIDs saved to: $PID_FILE"
echo "  Use ./stop.sh to stop services"
