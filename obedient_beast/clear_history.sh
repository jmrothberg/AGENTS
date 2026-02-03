#!/bin/bash
# =============================================================================
# Obedient Beast - Clear Conversation History
# =============================================================================
# Usage: ./clear_history.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Clearing Obedient Beast conversation history...${NC}"

# Remove all session files
rm -f sessions/*.jsonl

echo -e "${GREEN}✅ History cleared!${NC}"
echo "All conversations have been forgotten."