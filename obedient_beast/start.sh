#!/bin/bash
# =============================================================================
# Obedient Beast - Startup Script
# =============================================================================
# Usage:
#   ./start.sh              - Open 5 Terminal windows (lfm, server, whatsapp,
#                             heartbeat, CLI). Skips any already running.
#   ./start.sh pm2          - Start server/whatsapp/heartbeat via pm2 (background,
#                             auto-restart) + terminal windows for lfm and CLI.
#                             Skips pm2 services that are already online.
#   ./start.sh server       - Start only the Python server (terminal window)
#   ./start.sh whatsapp     - Start only the WhatsApp bridge (terminal window)
#   ./start.sh heartbeat    - Start only the heartbeat (terminal window)
#   ./start.sh lfm          - Open terminal window for lfm_thinking.py (interactive)
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
# Check if a process pattern is already running
# ---------------------------------------------------------------------------
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Start all: 5 separate Terminal windows (default, no pm2)
# ---------------------------------------------------------------------------
start_all() {
    echo -e "${GREEN}Opening Terminal windows (skipping any already running)...${NC}"

    # 1. LFM Thinking (interactive model picker)
    if is_running "python.*lfm_thinking.py"; then
        echo -e "  ${YELLOW}⚠ LFM Thinking already running — skipping${NC}"
    else
        open_terminal "🧠 LFM Thinking" "cd $PARENT_DIR && $ACTIVATE && python3 lfm_thinking.py"
        sleep 1
    fi

    # 2. Server
    if is_running "python.*server.py"; then
        echo -e "  ${YELLOW}⚠ Server already running — skipping${NC}"
    else
        open_terminal "🖥  Beast Server" "$ACTIVATE && cd $SCRIPT_DIR && python3 server.py"
        sleep 1
    fi

    # 3. WhatsApp
    if is_running "node.*bridge.js"; then
        echo -e "  ${YELLOW}⚠ WhatsApp Bridge already running — skipping${NC}"
    else
        open_terminal "📱 WhatsApp Bridge" "cd $SCRIPT_DIR/whatsapp && node bridge.js"
        sleep 1
    fi

    # 4. Heartbeat
    if is_running "python.*heartbeat.py"; then
        echo -e "  ${YELLOW}⚠ Heartbeat already running — skipping${NC}"
    else
        open_terminal "🫀 Heartbeat" "$ACTIVATE && cd $SCRIPT_DIR && python3 heartbeat.py"
        sleep 1
    fi

    # 5. CLI
    if is_running "python.*beast.py"; then
        echo -e "  ${YELLOW}⚠ Beast CLI already running — skipping${NC}"
    else
        open_terminal "🐺 Beast CLI" "$ACTIVATE && cd $SCRIPT_DIR && python3 beast.py"
    fi

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🐺 Obedient Beast is running in Terminal windows!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "  🧠 LFM Thinking   — Pick model & serve"
    echo "  🖥  Server         — API for WhatsApp"
    echo "  📱 WhatsApp        — Bridge to your phone"
    echo "  🫀 Heartbeat       — Autonomous task processor"
    echo "  🐺 CLI             — Type directly to Beast"
    echo ""
    echo "  Stop all: ./start.sh stop"
    echo ""
}

# ---------------------------------------------------------------------------
# Check pm2 is installed
# ---------------------------------------------------------------------------
check_pm2() {
    if ! command -v pm2 &> /dev/null; then
        echo -e "${RED}Error: pm2 not found. Install with: sudo npm install -g pm2${NC}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# pm2 mode: background services via pm2 + terminal windows for lfm + CLI
# ---------------------------------------------------------------------------
start_pm2() {
    check_pm2

    echo -e "${GREEN}Starting background services via pm2 (skipping already-online)...${NC}"

    # Beast HTTP server
    if pm2 describe beast-server 2>/dev/null | grep -q "online"; then
        echo -e "  ${YELLOW}⚠ beast-server already online — skipping${NC}"
    else
        pm2 start "$SCRIPT_DIR/server.py" \
            --name beast-server \
            --interpreter "$PARENT_DIR/.venv/bin/python3" \
            --cwd "$SCRIPT_DIR"
    fi

    # WhatsApp bridge
    if pm2 describe whatsapp-bridge 2>/dev/null | grep -q "online"; then
        echo -e "  ${YELLOW}⚠ whatsapp-bridge already online — skipping${NC}"
    else
        pm2 start "$SCRIPT_DIR/whatsapp/bridge.js" \
            --name whatsapp-bridge \
            --cwd "$SCRIPT_DIR/whatsapp"
    fi

    # Heartbeat
    if pm2 describe beast-heartbeat 2>/dev/null | grep -q "online"; then
        echo -e "  ${YELLOW}⚠ beast-heartbeat already online — skipping${NC}"
    else
        pm2 start "$SCRIPT_DIR/heartbeat.py" \
            --name beast-heartbeat \
            --interpreter "$PARENT_DIR/.venv/bin/python3" \
            --cwd "$SCRIPT_DIR"
    fi

    pm2 save > /dev/null 2>&1
    sleep 1

    # LFM Thinking — always a terminal window (interactive model picker)
    if is_running "python.*lfm_thinking.py"; then
        echo -e "  ${YELLOW}⚠ LFM Thinking already running — skipping${NC}"
    else
        open_terminal "🧠 LFM Thinking" "cd $PARENT_DIR && $ACTIVATE && python3 lfm_thinking.py"
        sleep 1
    fi

    # Beast CLI — always a terminal window (interactive)
    if is_running "python.*beast.py"; then
        echo -e "  ${YELLOW}⚠ Beast CLI already running — skipping${NC}"
    else
        open_terminal "🐺 Beast CLI" "$ACTIVATE && cd $SCRIPT_DIR && python3 beast.py"
    fi

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🐺 Obedient Beast is running (pm2 mode)!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "  pm2 background services:"
    echo "  🖥  beast-server     — API for WhatsApp   (pm2 logs beast-server)"
    echo "  📱 whatsapp-bridge  — Bridge to your phone (pm2 logs whatsapp-bridge)"
    echo "  🫀 beast-heartbeat  — Autonomous tasks     (pm2 logs beast-heartbeat)"
    echo ""
    echo "  Terminal windows:"
    echo "  🧠 LFM Thinking     — Pick model & serve"
    echo "  🐺 Beast CLI        — Type directly to Beast"
    echo ""
    echo "  pm2 status:  pm2 status"
    echo "  pm2 logs:    pm2 logs"
    echo "  Stop all:    ./start.sh stop"
    echo ""
}

# ---------------------------------------------------------------------------
# Individual starters (all open a terminal window)
# ---------------------------------------------------------------------------

start_lfm() {
    if is_running "python.*lfm_thinking.py"; then
        echo -e "${YELLOW}⚠ LFM Thinking already running${NC}"
    else
        open_terminal "🧠 LFM Thinking" "cd $PARENT_DIR && $ACTIVATE && python3 lfm_thinking.py"
    fi
}

start_server() {
    if is_running "python.*server.py"; then
        echo -e "${YELLOW}⚠ Server already running${NC}"
    else
        open_terminal "🖥  Beast Server" "$ACTIVATE && cd $SCRIPT_DIR && python3 server.py"
    fi
}

start_heartbeat() {
    if is_running "python.*heartbeat.py"; then
        echo -e "${YELLOW}⚠ Heartbeat already running${NC}"
    else
        open_terminal "🫀 Heartbeat" "$ACTIVATE && cd $SCRIPT_DIR && python3 heartbeat.py"
    fi
}

start_whatsapp() {
    if is_running "node.*bridge.js"; then
        echo -e "${YELLOW}⚠ WhatsApp Bridge already running${NC}"
    else
        if ! command -v node &> /dev/null; then
            echo -e "${RED}Error: Node.js not installed! Install with: brew install node${NC}"
            exit 1
        fi
        if [ ! -d "$SCRIPT_DIR/whatsapp/node_modules" ]; then
            echo -e "${YELLOW}Installing Node dependencies...${NC}"
            cd "$SCRIPT_DIR/whatsapp" && npm install
        fi
        open_terminal "📱 WhatsApp Bridge" "cd $SCRIPT_DIR/whatsapp && node bridge.js"
    fi
}

start_cli() {
    if is_running "python.*beast.py"; then
        echo -e "${YELLOW}⚠ Beast CLI already running${NC}"
    else
        open_terminal "🐺 Beast CLI" "$ACTIVATE && cd $SCRIPT_DIR && python3 beast.py"
    fi
}

# ---------------------------------------------------------------------------
# Stop / Status
# ---------------------------------------------------------------------------
stop_all() {
    echo -e "${YELLOW}Stopping Obedient Beast...${NC}"

    pkill -f "python.*lfm_thinking.py" 2>/dev/null && echo "  LFM Thinking stopped"
    pkill -f "python.*server.py"       2>/dev/null && echo "  Server stopped"
    pkill -f "node.*bridge.js"         2>/dev/null && echo "  WhatsApp stopped"
    pkill -f "python.*heartbeat.py"    2>/dev/null && echo "  Heartbeat stopped"
    pkill -f "python.*beast.py"        2>/dev/null && echo "  Beast CLI stopped"

    # Also stop pm2 services if pm2 is present
    if command -v pm2 &> /dev/null; then
        pm2 stop beast-server whatsapp-bridge beast-heartbeat 2>/dev/null \
            && echo "  pm2 services stopped"
    fi

    rm -f "$SCRIPT_DIR/.server.pid" "$SCRIPT_DIR/.whatsapp.pid" "$SCRIPT_DIR/.heartbeat.pid"
    echo -e "${GREEN}Stopped.${NC}"
}

check_status() {
    echo "Checking status..."
    for proc in \
        "python.*lfm_thinking.py:LFM Thinking" \
        "python.*server.py:Server" \
        "node.*bridge.js:WhatsApp" \
        "python.*heartbeat.py:Heartbeat" \
        "python.*beast.py:Beast CLI"
    do
        pattern="${proc%%:*}"
        name="${proc##*:}"
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ $name is running${NC}"
        else
            # Also check pm2 for background services
            pm2_name=""
            [[ "$name" == "Server" ]]   && pm2_name="beast-server"
            [[ "$name" == "WhatsApp" ]] && pm2_name="whatsapp-bridge"
            [[ "$name" == "Heartbeat" ]] && pm2_name="beast-heartbeat"
            if [[ -n "$pm2_name" ]] && command -v pm2 &> /dev/null \
               && pm2 describe "$pm2_name" 2>/dev/null | grep -q "online"; then
                echo -e "  ${GREEN}✓ $name is running (pm2)${NC}"
            else
                echo -e "  ${RED}✗ $name is NOT running${NC}"
            fi
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-all}" in
    pm2)        start_pm2 ;;
    server)     start_server ;;
    whatsapp)   start_whatsapp ;;
    heartbeat)  start_heartbeat ;;
    lfm)        start_lfm ;;
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
        echo "Usage: $0 {pm2|server|whatsapp|heartbeat|lfm|cli|stop|status|clear-history}"
        exit 1
        ;;
esac
