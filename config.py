from typing import Dict, Any

# Fault detection thresholds
THRESHOLDS = {
    "high_latency": {
        "saturation_threshold": 80,  # percentage
        "latency_threshold_ms": 3,
        "throughput_threshold": 0.9  # 90% of max throughput
    },
    "high_capacity": {
        "capacity_threshold": 90,  # percentage
        "snapshot_count_threshold": 10,
        "snapshot_size_ratio": 0.5  # 50% of volume size
    },
    "replication_issues": {
        "error_threshold": 0,
        "latency_threshold_ms": 100,
        "timeout_threshold_sec": 30
    }
}

# File patterns to monitor
FILE_PATTERNS = {
    "high_latency": [
        "volume.json",
        "system.json",
        "io_metrics.json",
        "logs_*.txt"
    ],
    "high_capacity": [
        "volume.json",
        "snapshots.json",
        "system_metrics_*.json"
    ],
    "replication_issues": [
        "settings.json",
        "replication_metrics_*.json",
        "volume.json",
        "host.json"
    ]
}

# RAG system settings
RAG_SETTINGS = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_store_path": "chroma_db"
}

# API settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
} 