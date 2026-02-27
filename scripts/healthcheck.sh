#!/bin/bash
# External health probe
pgrep -f "src.main" > /dev/null || exit 1
