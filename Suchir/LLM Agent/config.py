from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Using Llama 2 1B model
MODEL_CONFIG = {
    "max_length": 2048,
    "temperature": 0.01,
    "top_p": 0.7,
    "repetition_penalty": 1.7
}

# System Thresholds
SYSTEM_THRESHOLDS = {
    "saturation_threshold": 0.8,  # 80% system saturation
    "capacity_threshold": 0.8,    # 90% capacity usage
    "replication_delay": 300,     # 5 minutes max delay
    "max_snapshots": 100,         # Maximum number of snapshots
    "io_latency_threshold": 5,  # 5ms latency threshold
}

# File Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to parent directory
DATA_INSTANCE_PATTERN = "data_instance_{port}"

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "agent.log"
}

# Agent Configuration
AGENT_CONFIG = {
    "max_retries": 3,
    "timeout": 30,
    "temperature": 0.7,
}

# Diagnosis Rules
DIAGNOSIS_RULES = {
    "high_latency": {
        "metrics": ["system_saturation", "io_latency"],
        "thresholds": {
            "system_saturation": 0.8,
            "io_latency": 5
        }
    },
    "high_capacity": {
        "metrics": ["capacity_used", "snapshot_count"],
        "thresholds": {
            "capacity_used": 0.8,
            "snapshot_count": 100
        }
    },
    "replication_issues": {
        "metrics": ["replication_delay", "replication_status"],
        "thresholds": {
            "replication_delay": 300,
            "replication_status": "failed"
        }
    }
}

# Resolution Suggestions
RESOLUTION_SUGGESTIONS = {
    "high_latency": [
        "Check and optimize I/O operations",
        "Review and adjust system throughput settings",
        "Consider load balancing across volumes"
    ],
    "high_capacity": [
        "Review and adjust snapshot retention policies",
        "Clean up old snapshots",
        "Consider increasing system capacity"
    ],
    "replication_issues": [
        "Check network connectivity between systems",
        "Verify replication target system status",
        "Review replication delay settings"
    ]
} 