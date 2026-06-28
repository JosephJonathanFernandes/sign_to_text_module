#!/usr/bin/env bash
# Repository setup script for Unix/Linux/macOS
# Run from the project root:
#
#     bash scripts/setup.sh

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}=====================================================${NC}"
echo -e "${CYAN}  ISL Sign-to-Text — Developer Setup (Unix/macOS)${NC}"
echo -e "${CYAN}=====================================================\n${NC}"

# 1. Create venv if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[1/4] Creating virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${GREEN}[1/4] Virtual environment already exists${NC}"
fi

# 2. Install production deps
echo -e "${YELLOW}[2/4] Installing production dependencies...${NC}"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 3. Install dev deps
echo -e "${YELLOW}[3/4] Installing development dependencies...${NC}"
./venv/bin/pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
echo -e "${YELLOW}[4/4] Installing pre-commit hooks...${NC}"
./venv/bin/pre-commit install

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "  Activate:  source venv/bin/activate"
echo -e "  Run API:   python run_api.py"
echo -e "  Run tests: pytest tests/unit/ tests/api/"
echo -e "${GREEN}====================================================\n${NC}"
