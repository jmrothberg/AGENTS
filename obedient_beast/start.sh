#!/bin/bash
# =============================================================================
# Obedient Beast - Startup Script
# =============================================================================
# Usage:
#   ./start.sh              - Start both server and WhatsApp bridge
#   ./start.sh server       - Start only the Python server
#   ./start.sh whatsapp     - Start only the WhatsApp bridge
#   ./start.sh stop         - Stop all Beast processes
#   ./start.sh status       - Check if processes are running
#   ./start.sh clear-history - Clear all conversation history
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "🐺 Obedient Beast"
echo "============================================================"

start_server() {
    echo -e "${GREEN}Starting Python server...${NC}"
    cd "$SCRIPT_DIR"
    python3 server.py &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo $SERVER_PID > "$SCRIPT_DIR/.server.pid"
    sleep 2
}

start_whatsapp() {
    echo -e "${GREEN}Starting WhatsApp bridge...${NC}"
    cd "$SCRIPT_DIR/whatsapp"
    
    # Check if node is installed
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not installed!${NC}"
        echo "Install with: brew install node"
        exit 1
    fi
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing Node dependencies...${NC}"
        npm install
    fi
    
    node bridge.js &
    WA_PID=$!
    echo "WhatsApp bridge PID: $WA_PID"
    echo $WA_PID > "$SCRIPT_DIR/.whatsapp.pid"
}

stop_all() {
    echo -e "${YELLOW}Stopping Obedient Beast...${NC}"
    
    if [ -f "$SCRIPT_DIR/.server.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.server.pid") 2>/dev/null
        rm "$SCRIPT_DIR/.server.pid"
        echo "Server stopped"
    fi
    
    if [ -f "$SCRIPT_DIR/.whatsapp.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.whatsapp.pid") 2>/dev/null
        rm "$SCRIPT_DIR/.whatsapp.pid"
        echo "WhatsApp bridge stopped"
    fi
    
    # Also kill any remaining processes
    pkill -f "python.*server.py" 2>/dev/null
    pkill -f "node.*bridge.js" 2>/dev/null
    
    echo -e "${GREEN}Stopped.${NC}"
}

check_status() {
    echo "Checking status..."
    
    if pgrep -f "python.*server.py" > /dev/null; then
        echo -e "${GREEN}✓ Python server is running${NC}"
    else
        echo -e "${RED}✗ Python server is NOT running${NC}"
    fi
    
    if pgrep -f "node.*bridge.js" > /dev/null; then
        echo -e "${GREEN}✓ WhatsApp bridge is running${NC}"
    else
        echo -e "${RED}✗ WhatsApp bridge is NOT running${NC}"
    fi
}

case "${1:-all}" in
    server)
        start_server
        echo -e "${GREEN}Server started. Press Ctrl+C to stop.${NC}"
        wait
        ;;
    whatsapp)
        start_whatsapp
        echo -e "${GREEN}WhatsApp bridge started. Press Ctrl+C to stop.${NC}"
        wait
        ;;
    stop)
        stop_all
        ;;
    status)
        check_status
        ;;
    clear-history)
        echo -e "${YELLOW}Clearing conversation history...${NC}"
        rm -f "$SCRIPT_DIR/sessions/*.jsonl"
        echo -e "${GREEN}History cleared!${NC}"
        ;;
    all|"")
        # Start both
        start_server
        start_whatsapp
        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}🐺 Obedient Beast is running!${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo "Server:   http://localhost:5001"
        echo "WhatsApp: Check terminal for QR code (first time only)"
        echo ""
        echo "Commands:"
        echo "  ./start.sh stop          - Stop everything"
        echo "  ./start.sh status        - Check if running"
        echo "  ./start.sh clear-history - Clear conversation history"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop both processes${NC}"
        echo ""
        
        # Wait for both processes
        trap "stop_all; exit 0" INT TERM
        wait
        ;;
    *)
        echo "Usage: $0 {server|whatsapp|stop|status|clear-history|all}"
        exit 1
        ;;
esac
