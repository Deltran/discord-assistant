#!/bin/bash
set -euo pipefail

echo "Setting up Discord AI Assistant..."

# Create runtime directories
mkdir -p ~/.assistant/{memory,skills,config,data,logs}

# Copy default configs if not present
cp -n config/channels.yaml ~/.assistant/config/channels.yaml 2>/dev/null || true
cp -n config/schedule.yaml ~/.assistant/config/schedule.yaml 2>/dev/null || true

echo "Setup complete. Configure .env and start with: python -m src.main"
