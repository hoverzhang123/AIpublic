You can now run the tests:

# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Check coverage
pytest --cov=src tests/

# Run a single test file
pytest tests/test_config.py -v
