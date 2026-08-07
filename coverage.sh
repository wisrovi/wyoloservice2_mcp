#!/bin/bash
# Calculates test coverage inside a Docker container
echo "Calculating test coverage in Docker container..."
docker run --rm -v $(pwd):/app -w /app python:3.10-slim bash -c "pip install pytest pytest-cov && pytest --cov=. --cov-report=term-missing tests/"
