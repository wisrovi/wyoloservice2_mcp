# 🚀 NeuralForgeAI MCP Server (`wyoloservice-mcp`)

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/wisrovi/wyoloservice2_mcp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wyoloservice-mcp.svg)](https://pypi.org/project/wyoloservice-mcp/)

An advanced, state-of-the-art **Model Context Protocol (MCP)** server that empowers AI agents to seamlessly interact with the NeuralForgeAI YOLO training cluster, remote Samba datasets, and distributed worker nodes. Designed and architected by William Rodriguez (Wisrovi), this MCP bridges the gap between Large Language Models and complex, remote MLOps infrastructure.

---

## 📖 Table of Contents
1. [Introduction](#introduction)
2. [Why Use This MCP?](#why-use-this-mcp)
3. [Core Features](#core-features)
4. [Installation Guide](#installation-guide)
5. [Connecting to AI Assistants](#connecting-to-ai-assistants)
6. [Available Tools (Agent Capabilities)](#available-tools-agent-capabilities)
7. [Agentic Workflows & Examples](#agentic-workflows--examples)
8. [Advanced Configuration](#advanced-configuration)
9. [Architecture & Design](#architecture--design)
10. [Troubleshooting](#troubleshooting)
11. [About the Author](#about-the-author)

---

## 🌟 Introduction

In modern AI workflows, training YOLO models across a distributed cluster requires interacting with remote storage (CIFS/Samba), scheduling hyperparameter sweeps, and monitoring GPU instances. Typically, these tasks are manual and prone to human error. 

The `wyoloservice-mcp` allows any LLM (like Claude, Gemini via Antigravity, or OpenAI) to **act on your behalf** inside the NeuralForge ecosystem. By defining standard tools via the MCP protocol, the AI can validate datasets remotely, construct complex YAML configuration files, launch training trials, and report back on convergence metrics—all through natural language.

---

## 🎯 Why Use This MCP?

- **Zero-Friction Orchestration:** Stop writing `curl` commands or memorizing YAML schemas. Tell your agent: *"Launch a detection training for project Alpha"* and watch it happen.
- **Embedded Intelligence:** The MCP server doesn't just expose APIs; it injects strict workflow instructions into the LLM's prompt. It teaches the AI *how* to use the tools effectively.
- **Remote Execution:** Bypasses local filesystem constraints by spinning up ephemeral Docker containers (`worker_executor`) to validate data residing deep inside corporate Samba shares.
- **Fully Traceable:** The AI automatically modifies your local configuration files to inject return values (like `study_id`), ensuring you never lose track of a running cluster job.

---

## ✨ Core Features

1. **Stateful Credential Management:**
   Securely stores and manages API and CIFS credentials locally (`~/.wyolo_mcp_config.json`). The agent will ask for your credentials once and remember them across sessions.
2. **Remote Dataset Validation:**
   Instead of downloading terabytes of images to your local machine, the MCP triggers a validation container inside the cluster that mounts the CIFS share directly, verifying YOLO structures (labels, directories, `yaml` files) at the source.
3. **Comprehensive Cluster Monitoring:**
   Fetches real-time telemetry from multiple endpoints (`/health`, `/workers`, and `/tasks`) in a single parallelized HTTP call, giving the AI a holistic view of GPU availability and Celery queues.
4. **Sweeper YAML Generation (v2):**
   Autogenerates NeuralForge "Sweeper v2" configuration files locally for user review. It intelligently discovers dataset classes and uses `getpass`/`socket` to inject proper metadata and author tracing.
5. **1-Click Training Launch & Tracking:**
   Submits configurations to the NeuralForge API and, upon success, automatically rewrites the local YAML file to include the assigned `study_id`.

---

## ⚙️ Installation Guide

The package is officially published on PyPI. You can install it globally or inside a virtual environment.

### From PyPI (Recommended)
```bash
pip install wyoloservice-mcp
```

### From Source (Development)
```bash
git clone https://github.com/wisrovi/wyoloservice2_mcp.git
cd wyoloservice2_mcp
pip install -e .
```

Installation automatically exposes the global binary `wyolo-mcp`, which acts as the `stdio` server for the MCP protocol.

---

## 🔌 Connecting to AI Assistants

To give your AI access to these tools, you must register the `wyolo-mcp` binary in your client's configuration file.

### For Antigravity (Gemini)
Edit your `~/.gemini/config/mcp.json`:
```json
{
  "mcpServers": {
    "neuralforge-mcp": {
      "command": "wyolo-mcp"
    }
  }
}
```

### For Claude Desktop
Edit your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "neuralforge-mcp": {
      "command": "wyolo-mcp"
    }
  }
}
```

### For Cursor IDE
Navigate to `Cursor Settings > Features > MCP` and add a new server using the command `wyolo-mcp`.

---

## 🛠️ Available Tools (Agent Capabilities)

When connected, the AI automatically gains access to the following toolkit:

1. `set_cluster_credentials(ip, cifs_user, cifs_pass)`
   - Saves cluster IP and Samba credentials to the local config file.
2. `get_cluster_status()`
   - Aggregates `/health`, `/workers`, and `/tasks` into a single online diagnostic report.
3. `check_dataset_path(dataset_path)`
   - Performs a quick remote existence check on the Samba share via Docker.
4. `validate_dataset_advanced(dataset_path, task)`
   - Deeply inspects the remote dataset to extract classes, counts, and directory integrity.
5. `generate_training_yaml(config, output_dir)`
   - Builds a strict NeuralForge Sweeper v2 YAML. Enforces naming conventions (`<project>_<dataset>`).
6. `launch_training(yaml_path)`
   - Pushes the local YAML to the cluster and auto-injects the returned `study_id` back into the file.
7. `get_study_details(study_id)`
   - Fetches live Optuna trial progress, metrics, and active invokers.
8. `cancel_study(study_id)`
   - Terminates a running study on the cluster.

---

## 🧠 Agentic Workflows & Examples

This MCP is embedded with "Agentic Prompting." This means the tool docstrings actively instruct the AI on how to behave. Here are some natural language prompts you can use with your AI:

### Example 1: The Initial Setup
> **User:** "I need to configure my NeuralForge cluster. The IP is 192.168.10.252, user is wisrovi, password is wyoloservice."
> **AI:** *(Calls `set_cluster_credentials`)* "Credentials saved successfully! Your agent is now linked to the cluster."

### Example 2: One-Shot Training
> **User:** "Launch a classification training for the 'medical' project using the dataset `/datasets/AIDIAGNOST/classification/ages_classification/`. Use 150 epochs, 3 trials, and YOLOv8s-cls."
> **AI:** *(Calls `validate_dataset_advanced` to discover classes, then `generate_training_yaml` to build the sweeper config, and finally `launch_training` to submit it. It then updates the local YAML with the study_id).* "Training launched! The study ID has been injected into your local YAML file."

### Example 3: The "Magic" Status Check
> **User:** "How is my training going?"
> **AI:** *(Reads internal tool instruction: DO NOT ASK FOR ID. Scans local directory, finds the YAML file, reads the `study_id` inside it, and calls `get_study_details`)*. "Your study 'medical_ages_classification' is currently on trial 2 of 3, achieving a top-1 accuracy of 94%."

---

## 🏗️ Architecture & Design

`wyoloservice-mcp` is built using the `FastMCP` framework. It utilizes `httpx` for asynchronous parallel requests to the NeuralForge API, preventing UI freezing on the agent's end.

For remote dataset validation, it employs the `subprocess` module to orchestrate `docker run` commands dynamically. By mounting the Samba CIFS network at runtime inside a lightweight `wisrovi/train_service` container, it entirely decouples the AI's local host constraints from the massive storage requirements of the training ecosystem.

---

## 📝 License
This project is open-sourced under the MIT License.

---
**Author:** William Rodriguez (Wisrovi)
*AI Leader & Solutions Architect*