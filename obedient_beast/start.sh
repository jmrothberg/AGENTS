#!/bin/bash
# =============================================================================
# Obedient Beast - Startup Script
# =============================================================================
# Usage:
#   ./start.sh              - Open 4 Terminal windows (server, whatsapp, heartbeat, CLI)
#   ./start.sh server       - Start only the Python server
#   ./start.sh whatsapp     - Start only the WhatsApp bridge
#   ./start.sh heartbeat    - Start only the heartbeat
#   ./start.sh cli          - Start only the CLI (direct terminal chat)
#   ./start.sh stop         - Stop all Beast processes
#   ./start.sh status       - Check if processes are running
#   ./start.sh clear-history - Clear all conversation history
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Figure out the venv activate path
ACTIVATE=""
if [ -f "$PARENT_DIR/.venv/bin/activate" ]; then
    ACTIVATE="source $PARENT_DIR/.venv/bin/activate"
    source "$PARENT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    ACTIVATE="source $SCRIPT_DIR/.venv/bin/activate"
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "🐺 Obedient Beast"
echo "============================================================"

# ---------------------------------------------------------------------------
# Open a new terminal window and run a command (cross-platform)
# ---------------------------------------------------------------------------
open_terminal() {
    local title="$1"
    local cmd="$2"

    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS: open a new Terminal.app window
        osascript <<EOF
tell application "Terminal"
    activate
    set newTab to do script "echo '━━━ $title ━━━' && $cmd"
    set custom title of front window to "$title"
end tell
EOF
    elif command -v gnome-terminal &> /dev/null; then
        # Ubuntu/GNOME
        gnome-terminal --title="$title" -- bash -c "echo '━━━ $title ━━━' && $cmd; exec bash"
    elif command -v xterm &> /dev/null; then
        # Fallback: xterm
        xterm -T "$title" -e "echo '━━━ $title ━━━' && $cmd; bash" &
    else
        echo -e "${RED}No supported terminal found. Run each process manually:${NC}"
        echo "  python3 server.py"
        echo "  node whatsapp/bridge.js"
        echo "  python3 heartbeat.py"
        echo "  python3 beast.py"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Start all: 4 separate Terminal windows
# ---------------------------------------------------------------------------
start_all() {
    echo -e "${GREEN}Opening 4 Terminal windows...${NC}"

    # 1. Server
    open_terminal "🖥  Beast Server" "$ACTIVATE && cd $SCRIPT_DIR && python3 server.py"
    sleep 1

    # 2. WhatsApp
    open_terminal "📱 WhatsApp Bridge" "cd $SCRIPT_DIR/whatsapp && node bridge.js"
    sleep 1

    # 3. Heartbeat
    open_terminal "🫀 Heartbeat" "$ACTIVATE && cd $SCRIPT_DIR && python3 heartbeat.py"
    sleep 1

    # 4. CLI
    open_terminal "🐺 Beast CLI" "$ACTIVATE && cd $SCRIPT_DIR && python3 beast.py"

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🐺 Obedient Beast is running in 4 Terminal windows!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "  🖥  Server        — API for WhatsApp"
    echo "  📱 WhatsApp       — Bridge to your phone"
    echo "  🫀 Heartbeat      — Autonomous task processor"
    echo "  🐺 CLI            — Type directly to Beast"
    echo ""
    echo "  Stop all: ./start.sh stop"
    echo ""
}

# ---------------------------------------------------------------------------
# Individual starters (run in current terminal)
# ---------------------------------------------------------------------------
start_server() {
    echo -e "${GREEN}Starting Python server...${NC}"
    cd "$SCRIPT_DIR"
    python3 server.py
}

start_heartbeat() {
    echo -e "${GREEN}Starting heartbeat...${NC}"
    cd "$SCRIPT_DIR"
    python3 heartbeat.py
}

start_whatsapp() {
    echo -e "${GREEN}Starting WhatsApp bridge...${NC}"
    cd "$SCRIPT_DIR/whatsapp"
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not installed! Install with: brew install node${NC}"
        exit 1
    fi
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing Node dependencies...${NC}"
        npm install
    fi
    node bridge.js
}

start_cli() {
    cd "$SCRIPT_DIR"
    python3 beast.py
}

# ---------------------------------------------------------------------------
# Stop / Status
# ---------------------------------------------------------------------------
stop_all() {
    echo -e "${YELLOW}Stopping Obedient Beast...${NC}"
    pkill -f "python.*server.py" 2>/dev/null && echo "  Server stopped"
    pkill -f "node.*bridge.js" 2>/dev/null && echo "  WhatsApp stopped"
    pkill -f "python.*heartbeat.py" 2>/dev/null && echo "  Heartbeat stopped"
    pkill -f "python.*beast.py" 2>/dev/null && echo "  CLI stopped"
    # Clean up pid files
    rm -f "$SCRIPT_DIR/.server.pid" "$SCRIPT_DIR/.whatsapp.pid" "$SCRIPT_DIR/.heartbeat.pid"
    echo -e "${GREEN}Stopped.${NC}"
}

check_status() {
    echo "Checking status..."
    for proc in "python.*server.py:Server" "node.*bridge.js:WhatsApp" "python.*heartbeat.py:Heartbeat" "python.*beast.py:CLI"; do
        pattern="${proc%%:*}"
        name="${proc##*:}"
        if pgrep -f "$pattern" > /dev/null; then
            echo -e "  ${GREEN}✓ $name is running${NC}"
        else
            echo -e "  ${RED}✗ $name is NOT running${NC}"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-all}" in
    server)     start_server ;;
    whatsapp)   start_whatsapp ;;
    heartbeat)  start_heartbeat ;;
    cli)        start_cli ;;
    stop)       stop_all ;;
    status)     check_status ;;
    clear-history)
        echo -e "${YELLOW}Clearing conversation history...${NC}"
        rm -f "$SCRIPT_DIR/sessions/*.jsonl"
        echo -e "${GREEN}History cleared!${NC}"
        ;;
    all|"")     start_all ;;
    *)
        echo "Usage: $0 {server|whatsapp|heartbeat|cli|stop|status|clear-history}"
        exit 1
        ;;
esac
