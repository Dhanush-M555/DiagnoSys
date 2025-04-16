import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import ijson  # For handling large JSON files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        
    def get_data_instance_path(self, port: int) -> Path:
        """Get the path to a data instance directory."""
        path = self.base_path / f"data_instance_{port}"
        if not path.exists():
            logger.warning(f"Data instance directory not found: {path}")
            raise FileNotFoundError(f"Data instance {port} does not exist")
        return path
    
    def read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return {
                    "error": f"Required file {file_path.name} not found",
                    "status": "error"
                }
            
            # For large files like io_metrics.json, use ijson
            if file_path.name == "io_metrics.json":
                try:
                    with open(file_path, 'rb') as f:
                        # Read only the first 1000 items to avoid memory issues
                        items = []
                        parser = ijson.items(f, 'item')
                        for i, item in enumerate(parser):
                            if i >= 1000:
                                break
                            items.append(item)
                        return {"items": items, "total_items": i + 1}
                except Exception as e:
                    # Log the full error for debugging
                    logger.error(f"Error parsing large JSON file {file_path}: {e}")
                    # Return a simplified error message
                    return {
                        "items": [],
                        "total_items": 0,
                        "error": "Unable to parse metrics file",
                        "status": "error"
                    }
            
            # For regular JSON files
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Ensure we return a dictionary
                if isinstance(data, list):
                    return {"items": data, "total_items": len(data)}
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return {
                "error": "Invalid JSON format",
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            return {
                "error": "Unable to read file",
                "status": "error"
            }
    
    def read_log_file(self, file_path: Path, max_lines: int = 1000) -> List[str]:
        """Read a log file and return its lines."""
        try:
            if not file_path.exists():
                logger.warning(f"Log file not found: {file_path}")
                return []
            
            with open(file_path, 'r') as f:
                # Read only the last max_lines to avoid memory issues
                lines = f.readlines()
                return lines
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
            return []
    
    def analyze_logs(self, port: int) -> Dict[str, Any]:
        """Analyze logs for a specific port and return summary."""
        try:
            log_file = self.get_data_instance_path(port) / f"logs_{port}.txt"
            if not log_file.exists():
                return {"error": f"Log file not found: {log_file}"}
            
            lines = self.read_log_file(log_file)
            if not lines:
                return {"error": f"Could not read log file: {log_file}"}
            
            error_count = 0
            warning_count = 0
            high_latency_count = 0
            high_cpu_count = 0
            high_saturation_count = 0
            recent_errors = []
            recent_latency_events = []
            recent_cpu_events = []
            recent_saturation_events = []
            
            for line in lines:
                # Check for errors and warnings
                if "[ERROR]" in line:
                    error_count += 1
                    recent_errors.append(line.strip())
                elif "[WARNING]" in line:
                    warning_count += 1
                
                # Check for system metrics
                if "System metrics updated" in line:
                    try:
                        # Extract metrics using string operations
                        metrics_str = line.split("System metrics updated - ")[1]
                        metrics = {}
                        for part in metrics_str.split(", "):
                            key, value = part.split(": ")
                            metrics[key.strip()] = float(value.split()[0])
                        
                        # Check for high saturation
                        if "Saturation" in metrics and metrics["Saturation"] >= 80.0:
                            high_saturation_count += 1
                            recent_saturation_events.append(f"Saturation: {metrics['Saturation']:.2f}%")
                        
                        # Check for high latency
                        if "Latency" in metrics and metrics["Latency"] >= 5.0:
                            high_latency_count += 1
                            recent_latency_events.append(f"Latency: {metrics['Latency']:.2f}ms")
                        
                        # Check for high CPU
                        if "CPU" in metrics and metrics["CPU"] >= 80.0:
                            high_cpu_count += 1
                            recent_cpu_events.append(f"CPU: {metrics['CPU']:.2f}%")
                    
                    except Exception as e:
                        continue
            
            # Keep only the most recent events
            recent_errors = recent_errors[-5:]
            recent_latency_events = recent_latency_events[-5:]
            recent_cpu_events = recent_cpu_events[-5:]
            recent_saturation_events = recent_saturation_events[-5:]
            
            return {
                "total_lines": len(lines),
                "error_count": error_count,
                "warning_count": warning_count,
                "high_latency_count": high_latency_count,
                "high_cpu_count": high_cpu_count,
                "high_saturation_count": high_saturation_count,
                "recent_errors": recent_errors,
                "recent_latency_events": recent_latency_events,
                "recent_cpu_events": recent_cpu_events,
                "recent_saturation_events": recent_saturation_events
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_metrics(self, port: int) -> Dict[str, Any]:
        """Get system metrics for a specific port."""
        instance_path = self.get_data_instance_path(port)
        metrics_file = instance_path / f"system_metrics_{port}.json"
        return self.read_json_file(metrics_file)
    
    def get_volume_config(self, port: int) -> Dict[str, Any]:
        """Get volume configuration for a specific port."""
        instance_path = self.get_data_instance_path(port)
        volume_file = instance_path / "volume.json"
        return self.read_json_file(volume_file)
    
    def get_system_config(self, port: int) -> Dict[str, Any]:
        """Get system configuration for a specific port."""
        instance_path = self.get_data_instance_path(port)
        system_file = instance_path / "system.json"
        return self.read_json_file(system_file)
    
    def get_replication_metrics(self, port: int) -> Dict[str, Any]:
        """Get replication metrics for a specific port."""
        instance_path = self.get_data_instance_path(port)
        metrics_file = instance_path / f"replication_metrics_{port}.json"
        return self.read_json_file(metrics_file)
    
    def get_snapshots(self, port: int) -> Dict[str, Any]:
        """Get snapshot information for a specific port."""
        instance_path = self.get_data_instance_path(port)
        snapshots_file = instance_path / "snapshots.json"
        return self.read_json_file(snapshots_file)
    
    def get_io_metrics(self, port: int) -> Dict[str, Any]:
        """Get I/O metrics for a specific port."""
        instance_path = self.get_data_instance_path(port)
        metrics_file = instance_path / "io_metrics.json"
        return self.read_json_file(metrics_file)
    
    def get_logs(self, port: int) -> List[str]:
        """Get system logs for a specific port."""
        instance_path = self.get_data_instance_path(port)
        log_file = instance_path / f"logs_{port}.txt"
        return self.read_log_file(log_file)
    
    def get_snapshot_logs(self, port: int) -> List[str]:
        """Get snapshot logs for a specific port."""
        instance_path = self.get_data_instance_path(port)
        log_file = instance_path / "snapshot_log.txt"
        return self.read_log_file(log_file)

class MetricsAnalyzer:
    @staticmethod
    def check_system_saturation(system_metrics: Dict[str, Any], threshold: float = 0.8) -> bool:
        """Check if system is saturated based on metrics."""
        if not system_metrics:
            return False
        return system_metrics.get('saturation', 0) > threshold
    
    @staticmethod
    def check_capacity_usage(system_metrics: Dict[str, Any], threshold: float = 0.9) -> bool:
        """Check if system capacity is nearing limit."""
        if not system_metrics:
            return False
        return system_metrics.get('capacity_used', 0) > threshold
    
    @staticmethod
    def check_replication_delay(replication_metrics: Dict[str, Any], max_delay: int = 300) -> bool:
        """Check if replication delay is too high."""
        if not replication_metrics:
            return False
        return replication_metrics.get('delay_sec', 0) > max_delay
    
    @staticmethod
    def analyze_snapshot_retention(snapshots: Dict[str, Any], max_snapshots: int = 100) -> bool:
        """Check if too many snapshots are being retained."""
        if not snapshots:
            return False
        return len(snapshots.get('snapshots', [])) > max_snapshots 