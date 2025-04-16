import os
from typing import Dict, Any, List, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import FileHandler, MetricsAnalyzer
import config
import torch

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_CONFIG["level"]),
    format=config.LOG_CONFIG["format"],
    filename=config.LOG_CONFIG["filename"]
)
logger = logging.getLogger(__name__)

class SystemDiagnosticAgent:
    def __init__(self):
        # Initialize Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            token=config.HF_API_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            token=config.HF_API_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.file_handler = FileHandler(config.BASE_PATH)
        self.metrics_analyzer = MetricsAnalyzer()
        
    def _get_port_from_query(self, query: str) -> Optional[int]:
        """Extract port number from query if present."""
        import re
        # Try to find port in data_instance pattern
        match = re.search(r'data_instance_(\d+)', query)
        if match:
            return int(match.group(1))
        
        # Try to find port in logs pattern
        match = re.search(r'logs_(\d+)', query)
        if match:
            return int(match.group(1))
        
        return None
    
    def _analyze_high_latency(self, port: int) -> Dict[str, Any]:
        """Analyze system for high latency issues due to system saturation."""
        try:
            # Get volume configuration
            volume_config = self.file_handler.get_volume_config(port)
            if 'error' in volume_config:
                return {
                    "issue_found": False,
                    "metrics": {"error": volume_config['error']}
                }
            
            # Get system configuration
            system_config = self.file_handler.get_system_config(port)
            if 'error' in system_config:
                return {
                    "issue_found": False,
                    "metrics": {"error": system_config['error']}
                }
            
            # Get log analysis
            log_analysis = self.file_handler.analyze_logs(port)
            if 'error' in log_analysis:
                return {
                    "issue_found": False,
                    "metrics": {"error": log_analysis['error']}
                }
            
            # Check for high latency indicators
            has_high_latency = False
            latency_metrics = {}
            
            # 1. Check CPU and Network Saturation
            if log_analysis.get('high_saturation_count', 0) > 0:
                has_high_latency = True
                latency_metrics['system_saturation'] = {
                    'saturation_events': log_analysis.get('high_saturation_count'),
                    'recent_events': log_analysis.get('recent_saturation_events', [])
                }
            
            # 2. Check Volume Throughput
            for volume in volume_config.get('volumes', []):
                if volume.get('throughput_used', 0) > volume.get('max_throughput', 0) * 0.8:
                    has_high_latency = True
                    latency_metrics['volume_throughput'] = {
                        'volume_id': volume.get('volume_id'),
                        'throughput_used': volume.get('throughput_used'),
                        'max_throughput': volume.get('max_throughput')
                    }
            
            # 3. Check System Performance Metrics
            if system_config.get('saturation', 0) > 0.8:  # 80% threshold
                has_high_latency = True
                latency_metrics['system_performance'] = {
                    'current_saturation': system_config.get('saturation'),
                    'max_throughput': system_config.get('max_throughput'),
                    'current_throughput': system_config.get('throughput_used')
                }
            
            return {
                "issue_found": has_high_latency,
                "metrics": latency_metrics
            }
        except Exception as e:
            logger.error(f"Error analyzing high latency: {e}")
            return {
                "issue_found": False,
                "metrics": {"error": str(e)}
            }
    
    def _analyze_system_capacity(self, port: int) -> Dict[str, Any]:
        """Analyze system for high capacity issues due to snapshot retention."""
        try:
            # Get volume configuration
            volume_config = self.file_handler.get_volume_config(port)
            if 'error' in volume_config:
                return {
                    "issue_found": False,
                    "metrics": {"error": volume_config['error']}
                }
            
            # Get snapshots
            snapshots = self.file_handler.get_snapshots(port)
            if 'error' in snapshots:
                return {
                    "issue_found": False,
                    "metrics": {"error": snapshots['error']}
                }
            
            # Get system metrics
            system_metrics = self.file_handler.get_system_metrics(port)
            if 'error' in system_metrics:
                return {
                    "issue_found": False,
                    "metrics": {"error": system_metrics['error']}
                }
            
            # Get log analysis
            log_analysis = self.file_handler.analyze_logs(port)
            if 'error' in log_analysis:
                return {
                    "issue_found": False,
                    "metrics": {"error": log_analysis['error']}
                }
            
            # Check for capacity issues
            has_capacity_issues = False
            capacity_metrics = {}
            
            # 1. Check Snapshot Retention Settings
            snapshot_count = len(snapshots.get('snapshots', []))
            if snapshot_count > 100:  # Threshold from RCA
                has_capacity_issues = True
                capacity_metrics['snapshot_retention'] = {
                    'total_snapshots': snapshot_count,
                    'threshold': 100
                }
            
            # 2. Check Volume Capacity
            for volume in volume_config.get('volumes', []):
                if volume.get('capacity_used', 0) > volume.get('max_capacity', 0) * 0.8:
                    has_capacity_issues = True
                    capacity_metrics['volume_capacity'] = {
                        'volume_id': volume.get('volume_id'),
                        'used': volume.get('capacity_used'),
                        'max': volume.get('max_capacity')
                    }
            
            # 3. Check System Capacity Metrics
            if system_metrics.get('capacity_used', 0) > system_metrics.get('total_capacity', 0) * 0.8:
                has_capacity_issues = True
                capacity_metrics['system_capacity'] = {
                    'used': system_metrics.get('capacity_used'),
                    'total': system_metrics.get('total_capacity')
                }
            
            return {
                "issue_found": has_capacity_issues,
                "metrics": capacity_metrics
            }
        except Exception as e:
            logger.error(f"Error analyzing system capacity: {e}")
            return {
                "issue_found": False,
                "metrics": {"error": str(e)}
            }
    
    def _analyze_replication_issues(self, port: int) -> Dict[str, Any]:
        """Analyze system for replication link issues."""
        try:
            # Get replication settings
            settings = self.file_handler.get_replication_metrics(port)
            if 'error' in settings:
                return {
                    "issue_found": False,
                    "metrics": {"error": settings['error']}
                }
            
            # Get replication metrics
            replication_metrics = self.file_handler.get_replication_metrics(port)
            if 'error' in replication_metrics:
                return {
                    "issue_found": False,
                    "metrics": {"error": replication_metrics['error']}
                }
            
            # Get log analysis
            log_analysis = self.file_handler.analyze_logs(port)
            if 'error' in log_analysis:
                return {
                    "issue_found": False,
                    "metrics": {"error": log_analysis['error']}
                }
            
            # Check for replication issues
            has_replication_issues = False
            replication_metrics_details = {}
            
            # 1. Check Replication Status
            if replication_metrics.get('status', '').lower() != 'active':
                has_replication_issues = True
                replication_metrics_details['status'] = {
                    'current_status': replication_metrics.get('status'),
                    'expected_status': 'active'
                }
            
            # 2. Check Replication Delay
            if replication_metrics.get('delay_sec', 0) > 300:  # 5 minutes threshold
                has_replication_issues = True
                replication_metrics_details['delay'] = {
                    'current_delay': replication_metrics.get('delay_sec'),
                    'threshold': 300
                }
            
            # 3. Check Target System
            if not replication_metrics.get('target_system_reachable', True):
                has_replication_issues = True
                replication_metrics_details['target_system'] = {
                    'status': 'unreachable',
                    'target_id': replication_metrics.get('target_system_id')
                }
            
            # 4. Check Replication Logs
            replication_errors = [line for line in log_analysis.get('recent_errors', []) 
                                if any(word in line.lower() for word in ['replication', 'sync', 'target', 'network'])]
            if replication_errors:
                has_replication_issues = True
                replication_metrics_details['log_errors'] = {
                    'error_count': len(replication_errors),
                    'recent_errors': replication_errors[:5]
                }
            
            return {
                "issue_found": has_replication_issues,
                "metrics": replication_metrics_details
            }
        except Exception as e:
            logger.error(f"Error analyzing replication issues: {e}")
            return {
                "issue_found": False,
                "metrics": {"error": str(e)}
            }
    
    def _analyze_logs(self, port: int) -> Dict[str, Any]:
        """Analyze system logs for a specific port."""
        log_analysis = self.file_handler.analyze_logs(port)
        return {
            "issue_found": log_analysis["error_count"] > 0 or log_analysis["warning_count"] > 0,
            "metrics": log_analysis
        }
    
    def _generate_diagnosis(self, query: str) -> str:
        """Generate diagnosis using LLM based on the query."""
        try:
            # Check if this is a log analysis request
            if "analyse logs" in query.lower() or "analyze logs" in query.lower():
                # Extract port number from query
                port = None
                if "5001" in query:
                    port = 5001
                elif "5002" in query:
                    port = 5002
                elif "5003" in query:
                    port = 5003
                elif "5004" in query:
                    port = 5004
                
                if port is None:
                    return "Error: Could not determine which logs to analyze. Please specify the port number (e.g., 'analyse logs_5001.txt')."
                
                # Get log analysis results
                log_analysis = self.file_handler.analyze_logs(port)
                
                # Format the diagnosis
                diagnosis = "Log Analysis Results:\n\n"
                
                # 1. Log Summary
                diagnosis += "1. Log Summary:\n"
                diagnosis += f"   • Total log entries analyzed: {log_analysis['total_lines']}\n"
                diagnosis += f"   • Number of errors: {log_analysis['error_count']}\n"
                diagnosis += f"   • Number of warnings: {log_analysis['warning_count']}\n"
                if log_analysis['recent_errors']:
                    diagnosis += "   • Recent errors:\n"
                    for error in log_analysis['recent_errors']:
                        diagnosis += f"     - {error}\n"
                else:
                    diagnosis += "   • Recent errors: None\n"
                
                # 2. System Health Assessment
                diagnosis += "\n2. System Health Assessment:\n"
                stability = "Unstable" if log_analysis['error_count'] > 0 or log_analysis['high_latency_count'] > 0 or log_analysis['high_saturation_count'] > 0 else "Stable"
                diagnosis += f"   • Overall system stability: {stability}\n"
                diagnosis += f"   • High latency events: {log_analysis['high_latency_count']}\n"
                if log_analysis['recent_latency_events']:
                    diagnosis += "     Recent high latency events:\n"
                    for event in log_analysis['recent_latency_events']:
                        diagnosis += f"     - {event}\n"
                diagnosis += f"   • High CPU events: {log_analysis['high_cpu_count']}\n"
                diagnosis += f"   • High saturation events: {log_analysis['high_saturation_count']}\n"
                if log_analysis['recent_saturation_events']:
                    diagnosis += "     Recent high saturation events:\n"
                    for event in log_analysis['recent_saturation_events']:
                        diagnosis += f"     - {event}\n"
                
                # 3. Recommendations
                diagnosis += "\n3. Recommendations:\n"
                if log_analysis['error_count'] > 0:
                    diagnosis += "   • Investigate and resolve the recent error events in the logs\n"
                if log_analysis['high_latency_count'] > 0:
                    diagnosis += "   • Investigate the high latency events and their patterns\n"
                if log_analysis['high_saturation_count'] > 0:
                    diagnosis += "   • Address the high system saturation events\n"
                if log_analysis['total_lines'] >= 1000:
                    diagnosis += "   • Implement automated log rotation to prevent disk space issues\n"
                diagnosis += "   • Set up real-time log monitoring with alerting for critical errors\n"
                diagnosis += "   • Schedule regular log analysis to identify patterns and potential issues\n"
                
                return diagnosis
            
            # For general system analysis
            # Extract port number from query
            port = None
            if "5001" in query:
                port = 5001
            elif "5002" in query:
                port = 5002
            elif "5003" in query:
                port = 5003
            elif "5004" in query:
                port = 5004
            
            if port is None:
                return "Error: Could not determine which data instance to analyze. Please specify the port number (e.g., 'Check for issues in data_instance_5001')."
            
            # Perform system analysis
            latency_analysis = self._analyze_high_latency(port)
            capacity_analysis = self._analyze_system_capacity(port)
            replication_analysis = self._analyze_replication_issues(port)
            
            # Format the diagnosis
            diagnosis = "System Analysis Results:\n\n"
            
            # 1. System Status
            diagnosis += "1. System Status:\n"
            overall_health = "Issues Detected" if any([
                latency_analysis["issue_found"],
                capacity_analysis["issue_found"],
                replication_analysis["issue_found"]
            ]) else "Healthy"
            diagnosis += f"   • Overall health: {overall_health}\n"
            
            critical_issues = []
            if latency_analysis["issue_found"]:
                critical_issues.append("high_latency")
            if capacity_analysis["issue_found"]:
                critical_issues.append("system_capacity")
            if replication_analysis["issue_found"]:
                critical_issues.append("replication_issues")
            diagnosis += f"   • Critical issues: {', '.join(critical_issues) if critical_issues else 'None'}\n"
            
            # 2. Detailed Analysis
            diagnosis += "\n2. Detailed Analysis:\n"
            diagnosis += f"   • Performance metrics: {'Issues Detected' if latency_analysis['issue_found'] else 'Normal'}\n"
            diagnosis += f"   • Capacity utilization: {'High' if capacity_analysis['issue_found'] else 'Normal'}\n"
            diagnosis += f"   • Replication status: {'Issues Detected' if replication_analysis['issue_found'] else 'Normal'}\n"
            
            # 3. Recommendations
            diagnosis += "\n3. Recommendations:\n"
            if latency_analysis["issue_found"]:
                diagnosis += "   • Optimize system resources to reduce latency\n"
            if capacity_analysis["issue_found"]:
                diagnosis += "   • Plan for system capacity expansion or implement data archiving\n"
            if replication_analysis["issue_found"]:
                diagnosis += "   • Verify and fix replication configuration settings\n"
            
            return diagnosis

        except Exception as e:
            logger.error(f"Error generating diagnosis: {e}")
            return f"Error generating diagnosis: {str(e)}"
    
    def diagnose_issue(self, query: str) -> str:
        """Main method to diagnose system issues based on user query."""
        try:
            port = self._get_port_from_query(query)
            if not port:
                return "Error: Could not determine data instance port from query."
            
            # Check if the data instance directory exists
            data_instance_path = os.path.join(config.BASE_PATH, f"data_instance_{port}")
            if not os.path.exists(data_instance_path):
                return f"Error: Data instance {port} does not exist."
            
            # Check if query is about logs
            if "log" in query.lower() or "logs" in query.lower():
                try:
                    log_analysis = self._analyze_logs(port)
                    if 'error' in log_analysis.get('metrics', {}):
                        return f"Error analyzing logs: {log_analysis['metrics']['error']}"
                    analysis_results = {"log_analysis": log_analysis}
                except Exception as e:
                    return f"Error analyzing logs: {str(e)}"
            else:
                try:
                    # Perform all analyses
                    latency_analysis = self._analyze_high_latency(port)
                    capacity_analysis = self._analyze_system_capacity(port)
                    replication_analysis = self._analyze_replication_issues(port)
                    
                    # Check for errors in any analysis
                    for analysis in [latency_analysis, capacity_analysis, replication_analysis]:
                        if 'error' in analysis.get('metrics', {}):
                            return f"Error in analysis: {analysis['metrics']['error']}"
                    
                    # Combine results
                    analysis_results = {
                        "high_latency": latency_analysis,
                        "system_capacity": capacity_analysis,
                        "replication_issues": replication_analysis
                    }
                except Exception as e:
                    return f"Error performing system analysis: {str(e)}"
            
            # Generate diagnosis
            diagnosis = self._generate_diagnosis(query)
            return diagnosis
        except Exception as e:
            logger.error(f"Error in diagnose_issue: {e}")
            return f"Error: {str(e)}"

def main():
    """Main function to run the agent interactively."""
    print("\n\n\t\t\t\t\t\t\t🤖 System Diagnostic Agent 🤖\n")
    print("Enter 'quit' to exit")
    
    agent = SystemDiagnosticAgent()
    
    while True:
        query = input("\nEnter your query (e.g., 'Check for issues in data_instance_5001'): ")
        if query.lower() in ["quit", "exit", 'q', 'bye', 'goodbye']:
            break
            
        try:
            result = agent.diagnose_issue(query)
            print("\nDiagnosis:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 