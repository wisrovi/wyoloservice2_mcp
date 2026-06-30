# 🚀 NeuralForgeAI MCP Server (`wyoloservice-mcp`)

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/wisrovi/wyoloservice2_mcp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

An advanced Model Context Protocol (MCP) server that empowers AI agents to seamlessly interact with the NeuralForgeAI YOLO training cluster and remote Samba datasets.

## 🌟 Features

- **Stateful Credential Management:** Securely stores and manages API and CIFS credentials (`set_cluster_credentials`) so agents only ask once.
- **Remote Dataset Validation:** Spins up ephemeral Docker containers (`worker_executor`) to mount remote CIFS shares and validate YOLO structures (`check_dataset_path`, `validate_dataset_advanced`).
- **Comprehensive Cluster Monitoring:** Fetches real-time telemetry from `/health`, `/workers`, and `/tasks` in a single parallelized call (`get_cluster_status`).
- **Sweeper YAML Generation:** Autogenerates NeuralForge "Sweeper v2" configuration files locally for user review, dynamically discovering dataset classes and metadata (`generate_training_yaml`).
- **1-Click Training Launch:** Submits YAML configurations directly to the NeuralForge API and automatically injects the `study_id` back into the local YAML file for complete traceability (`launch_training`, `get_study_details`, `cancel_study`).

## ⚙️ Installation

Install the package directly via `pip`:

```bash
git clone https://github.com/wisrovi/wyoloservice2_mcp.git
cd wyoloservice2_mcp
pip install -e .
```

This will expose the global `wyolo-mcp` binary.

## 🔌 Connecting to your AI Assistant

Add the following to your AI Assistant's MCP configuration file (e.g. `~/.gemini/config/mcp.json` or Claude Desktop config):

```json
{
  "mcpServers": {
    "neuralforge-mcp": {
      "command": "wyolo-mcp"
    }
  }
}
```

## 🧠 Agentic Workflow (Built-in Intelligence)

This MCP server is designed to self-instruct the LLM. For instance:
- **Project Enforcement:** If you don't provide a project name, the agent knows it must ask you to conform to `<project>_<dataset>`.
- **Auto-Discovery:** When you ask "how is my training going?", the agent is instructed by the MCP docstrings to automatically scan your directory for `.yaml` files, extract the `study_id`, and fetch the status without you providing any IDs.

---
**Author:** Jose Manuel Pecero Blanco