#!/bin/bash
# =============================================================================
# Obedient Beast - One-Command Setup Script
# =============================================================================
# This script sets up everything needed to run Obedient Beast on a fresh system.
#
# Usage:
#   ./setup.sh              - Full setup (Python + Node.js dependencies)
#   ./setup.sh --no-node    - Python only (skip WhatsApp bridge)
#   ./setup.sh --check      - Check if everything is installed
#
# After setup, run:
#   ./start.sh              - Start the agent
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "============================================================"
echo -e "${BLUE}🐺 Obedient Beast - Setup Script${NC}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Check System Requirements
# ---------------------------------------------------------------------------

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found: $(command -v "$1")"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    echo ""
    
    MISSING=0
    
    # Python 3
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        echo "   Version: $PYTHON_VERSION"
    else
        echo "   Install with: brew install python3 (macOS) or apt install python3 (Ubuntu)"
        MISSING=1
    fi
    
    # pip
    if check_command pip3; then
        :
    else
        echo "   Install with: python3 -m ensurepip"
        MISSING=1
    fi
    
    # Node.js (optional for WhatsApp)
    if [[ "$1" != "--no-node" ]]; then
        if check_command node; then
            NODE_VERSION=$(node --version 2>&1)
            echo "   Version: $NODE_VERSION"
        else
            echo "   Install with: brew install node (macOS) or apt install nodejs npm (Ubuntu)"
            echo "   Or run ./setup.sh --no-node to skip WhatsApp"
            MISSING=1
        fi
        
        if check_command npm; then
            :
        else
            MISSING=1
        fi
    fi
    
    echo ""
    
    if [ $MISSING -eq 1 ]; then
        echo -e "${RED}Missing requirements. Please install them first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All requirements met!${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Create Virtual Environment
# ---------------------------------------------------------------------------

setup_python() {
    echo -e "${BLUE}Setting up Python environment...${NC}"
    
    # Create venv if it doesn't exist
    if [ ! -d "$PARENT_DIR/.venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$PARENT_DIR/.venv"
    else
        echo "Virtual environment already exists"
    fi
    
    # Activate venv
    source "$PARENT_DIR/.venv/bin/activate"
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    # Install server deps (MLX/FastAPI) then Beast deps
    echo "Installing Python dependencies (local LLM server)..."
    pip install -r "$PARENT_DIR/requirements.txt"
    echo "Installing Python dependencies (Beast agent)..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Setup Node.js Dependencies
# ---------------------------------------------------------------------------

setup_node() {
    echo -e "${BLUE}Setting up Node.js dependencies...${NC}"
    
    cd "$SCRIPT_DIR/whatsapp"
    
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    else
        echo "Node modules already installed"
    fi
    
    cd "$SCRIPT_DIR"
    
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Create .env Configuration
# ---------------------------------------------------------------------------

setup_env() {
    echo -e "${BLUE}Setting up configuration...${NC}"
    
    ENV_FILE="$PARENT_DIR/.env"
    
    if [ -f "$ENV_FILE" ]; then
        echo ".env file already exists at $ENV_FILE"
        echo "Skipping to preserve your settings"
    else
        echo "Creating .env from template..."
        cat > "$ENV_FILE" << 'EOF'
# =============================================================================
# Obedient Beast - Configuration (repo-root .env is canonical)
# =============================================================================
# LLM Backend: "lfm" (local model), "openai", or "claude"
# "lfm" is a legacy name — it means any model on the local server (Qwen, etc.)
LLM_BACKEND=lfm

# Local model server (no /v1 suffix — the client appends /v1/chat/completions)
LFM_URL=http://localhost:8000
# LFM_URL_REMOTE=http://192.168.1.100:8000
LFM_MODEL=Qwen3.8-27B-mxfp8

# Qwen3.8 thinking knobs as chat_template_kwargs (ignored by Claude/OpenAI;
# other local models ignore unknown fields). reasoning_effort: low|medium|xhigh
QWEN_REASONING_EFFORT=medium
QWEN_ENABLE_THINKING=true
QWEN_PRESERVE_THINKING=true

# Tool groups for local 27B: core,browser,art,desktop,mcp_mgmt,spawn or "all"
# BEAST_TOOL_GROUPS=core,browser,art

# Cloud keys (only needed if you switch with /claude or /openai)
# ANTHROPIC_API_KEY=your-anthropic-key-here
# OPENAI_API_KEY=your-openai-key-here

# WhatsApp — comma-separated numbers with country code
ALLOWED_NUMBERS=+12025551234

# MCP tool servers (filesystem, brave-search, …)
MCP_ENABLED=false
# BRAVE_API_KEY=
EOF
        echo -e "${YELLOW}⚠ Created .env at repo root — local backend is the default.${NC}"
        echo "   Location: $ENV_FILE"
    fi
    
    echo ""
}

# ---------------------------------------------------------------------------
# Create Necessary Directories
# ---------------------------------------------------------------------------

setup_directories() {
    echo -e "${BLUE}Creating directories...${NC}"
    
    mkdir -p "$SCRIPT_DIR/sessions"
    mkdir -p "$SCRIPT_DIR/workspace/screenshots"
    mkdir -p "$SCRIPT_DIR/config"
    
    echo -e "${GREEN}✓ Directories created${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Verify Installation
# ---------------------------------------------------------------------------

verify_installation() {
    echo -e "${BLUE}Verifying installation...${NC}"
    
    # Activate venv
    source "$PARENT_DIR/.venv/bin/activate"
    
    # Test Python imports
    echo "Testing Python imports..."
    python3 -c "
import sys
try:
    import flask
    import anthropic
    import openai
    from dotenv import load_dotenv
    print('  ✓ Core dependencies OK')
except ImportError as e:
    print(f'  ✗ Missing: {e}')
    sys.exit(1)

try:
    import pyautogui
    import mss
    print('  ✓ Computer control (pyautogui, mss) OK')
except ImportError as e:
    print(f'  ⚠ Computer control not available: {e}')
    print('    (Optional - run: pip install pyautogui mss pillow)')
"
    
    # Test beast.py loads
    echo "Testing beast.py..."
    cd "$SCRIPT_DIR"
    python3 -c "import beast; print(f'  ✓ beast.py loads OK ({len(beast.TOOLS)} tools)')"
    
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Print Next Steps
# ---------------------------------------------------------------------------

print_next_steps() {
    echo "Next steps:"
    echo ""
    echo "  1. (Optional) Edit $PARENT_DIR/.env — default is local via LLM_BACKEND=lfm (LFM_MODEL)"
    echo ""
    echo "  2. Typical Mac night (WhatsApp, no extra CLI):"
    echo -e "     ${YELLOW}cd $SCRIPT_DIR && ./start.sh phone${NC}"
    echo "     Then, to type in Terminal: ./start.sh you"
    echo ""
    echo "  3. Chat only (local model + CLI, no WhatsApp):"
    echo -e "     ${YELLOW}cd $SCRIPT_DIR && ./start.sh cli${NC}"
    echo ""
    echo "  4. Full five-window stack (WhatsApp + heartbeat + CLI):"
    echo -e "     ${YELLOW}cd $SCRIPT_DIR && ./start.sh${NC}"
    echo "     Scan the QR code in the WhatsApp window"
    echo ""
    echo "  5. Switch brains later: /claude  /openai  /lfm  (or LLM_BACKEND in .env)"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-}" in
    --check)
        check_requirements "$@"
        exit 0
        ;;
    --no-node)
        check_requirements "--no-node"
        setup_python
        setup_env
        setup_directories
        verify_installation
        print_next_steps
        ;;
    --help|-h)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (none)      Full setup (Python + Node.js for WhatsApp)"
        echo "  --no-node   Skip Node.js setup (no WhatsApp)"
        echo "  --check     Check system requirements only"
        echo "  --help      Show this help"
        exit 0
        ;;
    *)
        check_requirements
        setup_python
        setup_node
        setup_env
        setup_directories
        verify_installation
        print_next_steps
        ;;
esac
