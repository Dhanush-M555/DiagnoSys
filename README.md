# Storage System Fault Detection

A RAG-based fault detection system for storage subsystems that uses AI to identify and provide recommendations for common storage system issues.

## Features

- Real-time fault detection for:
  - High latency due to system saturation
  - High system capacity (snapshot retention)
  - Replication link issues
- AI-powered recommendations using RAG
- REST API interface
- Configurable thresholds and monitoring patterns

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_api_key_here
```

3. Run the API server:
```bash
python api.py
```

## API Endpoints

### 1. Analyze System
```
GET /analyze
```
Returns a list of detected faults with recommendations.

### 2. Get Recommendations
```
GET /recommendations/{fault_type}
```
Returns specific recommendations for a given fault type.

## Fault Types

### High Latency
- Monitors system saturation and throughput
- Checks volume I/O metrics
- Analyzes latency patterns

### High Capacity
- Tracks snapshot retention
- Monitors system capacity usage
- Analyzes volume and snapshot sizes

### Replication Issues
- Monitors replication metrics
- Checks for connection timeouts
- Analyzes replication errors

## Configuration

The system can be configured through `config.py`:

- Thresholds for different fault types
- File patterns to monitor
- RAG system settings
- API configuration

## Integration

The fault detection system can be integrated with the existing storage system by:

1. Monitoring the data instance directories
2. Analyzing system metrics in real-time
3. Providing recommendations through the API

## Example Usage

```python
import requests

# Analyze system for faults
response = requests.get("http://localhost:8000/analyze")
faults = response.json()

# Get recommendations for a specific fault
recommendations = requests.get("http://localhost:8000/recommendations/high_latency")
``` 