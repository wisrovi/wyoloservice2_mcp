# Contributing to the train_service2 Ecosystem

First off, thank you for considering contributing to this project! It's people like you that make open-source and collaborative tools such a great community.

## General Rules (Strict Compliance)

By contributing to this repository, you agree to adhere to the following rules established by the core architecture team:

1. **Language**: 
   - All code (variables, functions, classes), comments, and documentation MUST be written in **English**.
   - Git commit messages MUST be written in perfect **English**.

2. **Git Commits**:
   - You MUST make one commit per modified file (one file = one commit).
   - Commit messages MUST start with a descriptive category tag in brackets. Examples: `[FEATURE]`, `[DOCS]`, `[FIX]`, `[REFACTOR]`, `[TEST]`.

3. **Versioning and Documentation**:
   - Every time a modification is made, the version of the corresponding repository MUST be bumped (updated) internally in its main code files.
   - You MUST document your changes in the `README.md` changelog.
   - You MUST ensure the `README.md` file stays synchronized with the current state of the project at all times.

## Testing Guidelines

1. **Framework**: All unit tests must be written using `pytest`.
2. **Containerized Execution**: Unit tests MUST be executed inside a Docker container. Do not run tests directly on your host machine to avoid dependency conflicts.
3. **Coverage**: Every project with tests must maintain and run a script to calculate code coverage.
4. **Test Documentation**: Test files must contain extensive comments explaining what the test does and what it validates.

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Ensure you have followed all the rules mentioned above.
3. Run the tests via Docker and ensure they pass.
4. Issue your Pull Request and request a review from the maintainers.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
