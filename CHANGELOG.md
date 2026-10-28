# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.6] - 2026-10-28
### Added
- Optional `until_time` field in `manage_invoker_queues` MCP tool to allow pausing invoker nodes until a specific wall-clock time (`"HH:MM"`).
- Unit test for scheduled pause calculation in `tests/test_example.py`.

## [0.4.5] - 2026-10-26
### Added
- Optional `time` field in `TrainingConfig` and `generate_training_yaml` tool to support time-bounded training in Ultralytics YOLO with automatic infrastructure timeout scaling.
- Comprehensive unit tests in `tests/test_example.py` for optional `time` parameter handling.
- Documentation for test execution and Dockerized coverage.

## [Unreleased]
### Added
- Standardized testing infrastructure (tests folder, run_tests.sh, coverage.sh via Docker).
- GitHub issue and PR templates.
- This CHANGELOG file to replace in-README changelogs.
