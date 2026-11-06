# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.2] - 2026-11-06
### Added
- **WPipe Architectural Compliance:** Refactored `wyoloservice2_mcp` to 100% compliance with WPipe architecture and QA validation rules.
- **Dedicated DAG Pipeline:** Created `mcp_server_pipeline` in `pipelines.py` with `mcp_startup_step`.
- **Modular Layout:** Decomposed `server.py` into dedicated SRP utility modules under `src/wyolo_mcp/utils/` (`cluster_tools.py`, `dataset_tools.py`, `training_tools.py`, `estimator_tools.py`, `study_tools.py`, `artifact_tools.py`, `invoker_tools.py`, `test_launcher.py`, `xai_tools.py`), strictly adhering to the 200-line SRP ceiling.
- **Typed DTOs:** Created Pydantic DTO models in `dto/training_config_dto.py` for structured study configuration.

## [0.5.1] - 2026-11-03
### Added
- **Vision Style Injection in Generated YAML:** `generate_training_yaml` now explicitly emits `style` parameter matching `config.task` (`detect`, `segment`, `classify`, etc.) to drive downstream Optuna objective fitness auto-mapping.

## [0.5.0] - 2026-11-03
### Added
- Integration with Kubernetes Cluster Admin API (`recommend_training_node`, `estimate_study_duration`, `preflight_check_node`).
- Explainable AI (XAI) and robustness metric inspection tools: `get_study_xai_metrics` and `get_study_heatmaps_summary`.
- Surgical artifact downloader `download_study_artifact` for targeted retrieval of weights (`best.pt`), reports (`LLM_Report.docx`), heatmaps, or LaTeX tables.
- Extended `TrainingConfig` and `generate_training_yaml` with full support for next-gen **YOLO26** architectures, multi-task settings (`obb`, `pose`), Optuna samplers (`tpe`, `cmaes`), and post-training XAI toggles.
- Pinned `mcp>=1.0.0,<2.0.0` for rock-solid stability with the FastMCP protocol standard.

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
