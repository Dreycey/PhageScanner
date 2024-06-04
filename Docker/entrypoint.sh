#!/bin/bash
# Entry point script for PhageScanner

# Default to showing help if no arguments are provided
if [ "$#" -eq 0 ]; then
    set -- "--help"
fi

# Run PhageScanner
exec python3 /PhageScanner/phagescanner.py "$@"
