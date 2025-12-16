#!/bin/bash
# Continuous testing script for Isolation Forest Wafer Defect Detection

set -e  # Exit on error

echo "======================================================================"
echo "Running Isolation Forest Test Suite"
echo "======================================================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run 'python3 -m venv venv' first."
    exit 1
fi

# Run linting
echo ""
echo "1. Running code quality checks (flake8)..."
echo "----------------------------------------------------------------------"
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,archive || true
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv,archive || true

# Run unit tests
echo ""
echo "2. Running unit tests..."
echo "----------------------------------------------------------------------"
pytest tests/test_feature_extraction.py -v -m "not slow"

# Run integration tests
echo ""
echo "3. Running integration tests..."
echo "----------------------------------------------------------------------"
pytest tests/test_isolation_forest.py -v -m "not slow"

# Run data validation tests
echo ""
echo "4. Running data validation tests..."
echo "----------------------------------------------------------------------"
pytest tests/test_data_validation.py -v -m "not slow"

# Run all tests with coverage
echo ""
echo "5. Running full test suite with coverage..."
echo "----------------------------------------------------------------------"
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

echo ""
echo "======================================================================"
echo "✅ All tests completed!"
echo "======================================================================"
echo ""
echo "Coverage report generated at: htmlcov/index.html"
echo ""
echo "Test Summary:"
pytest tests/ --co -q | grep -E "test session starts|test_"
