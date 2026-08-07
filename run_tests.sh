#!/bin/bash
# Runs pytest inside a Docker container
echo "Running tests in Docker container..."
docker run --rm -v $(pwd):/app -w /app python:3.10-slim bash -c "pip install pytest && pytest tests/"
