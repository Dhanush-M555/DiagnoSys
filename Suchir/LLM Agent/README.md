# LLM Agent for System Diagnostics

This project implements an LLM-based agent that can diagnose system issues by analyzing configuration files, metrics, and logs from data instances.

## Features

- Automated system diagnostics based on predefined rules
- Analysis of configuration files, metrics, and logs
- Root cause analysis and resolution suggestions
- Integration with Autogen for agentic workflows

## Project Structure

```
LLM Agent/
├── agent.py           # Main agent implementation
├── config.py          # Configuration settings
├── utils.py           # Utility functions for file operations
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

The agent can be used to diagnose three main types of issues:
1. High Latency Due to System Saturation
2. High System Capacity (Snapshot Retention Settings)
3. Replication Link Issues

Example usage:
```python
from agent import SystemDiagnosticAgent

agent = SystemDiagnosticAgent()
diagnosis = agent.diagnose_issue("Check for high latency issues in data_instance_5001")
print(diagnosis)
```

## Configuration Files

The agent analyzes the following files in each data instance:
- system.json
- volume.json
- host.json
- settings.json
- snapshots.json
- system_metrics_{port}.json
- replication_metrics_{port}.json
- logs_{port}.txt
- io_metrics.json
- snapshot_log.txt 