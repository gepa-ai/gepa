#!/bin/bash
# Simple test runner script

echo "Running MultiTraceEnv bug tests..."
python -m unittest tests.test_env_bugs -v

echo ""
echo "Running other critical tests..."
# Run multi-region billing tests
python -m unittest tests.test_multi_region_billing -v

# Run restart and migration tests  
python -m unittest tests.test_restart_and_migration -v

echo ""
echo "All tests completed!"