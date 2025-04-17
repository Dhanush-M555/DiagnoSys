from typing import List, Dict, Any
import json
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class StorageFaultDetector:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir  # This should be the workspace root directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with storage system fault patterns and historical data."""
        # Base fault patterns
        knowledge_base = [
            {
                "pattern": "Saturation-induced latency pattern: High latency correlated with system saturation.",
                "indicators": {
                    "latency": ">=5ms",
                    "saturation": ">100%",
                    "cpu_utilization": "high",
                    "io_wait_times": "increased",
                    "queue_depth": "increased"
                },
                "type": "saturation_induced_latency",
                "severity": "high",
                "context": "High latency caused by system resource saturation, typically due to excessive concurrent operations or resource exhaustion.",
                "resolution": "Consider scaling resources or implementing request throttling"
            },
            {
                "pattern": "Capacity-induced latency pattern: High latency due to storage capacity issues.",
                "indicators": {
                    "capacity_usage": ">100%",
                    "write_latency": "spikes during write operations",
                    "cleanup_operations": "failed",
                    "write_throughput": "reduced",
                    "storage_fragmentation": "increased"
                },
                "type": "capacity_induced_latency",
                "severity": "high",
                "context": "High latency caused by storage system operating near capacity limits, affecting write performance and garbage collection.",
                "resolution": "Implement cleanup procedures or increase storage capacity"
            },
            {
                "pattern": "Active synchronous replication for volume {volume_id} to target System {destination} (TimeTaken {time_taken}ms)",
                "indicators": {
                    "time_taken": ">=5",
                    "replication_status": "active",
                    "replication_type": "synchronous"
                },
                "type": "replication_induced_latency",
                "severity": "high",
                "context": "Replication latency exceeds threshold of 5ms, indicating potential network or resource constraints affecting synchronous replication performance",
                "resolution": "Investigate network connectivity, resource utilization, and replication configuration"
            }
        ]

        # Convert knowledge base entries to text documents with semantic context
        documents = []
        metadatas = []
        
        for entry in knowledge_base:
            # Create rich semantic context
            doc_text = f"""
            Fault Pattern Analysis:
            Pattern Description: {entry['pattern']}
            Fault Type: {entry['type']}
            Severity Level: {entry['severity']}
            
            Key Indicators:
            {chr(10).join(f'- {key}: {value}' for key, value in entry['indicators'].items())}
            
            Contextual Information:
            {entry['context']}
            
            Resolution Strategy:
            {entry['resolution']}
            
            Related Metrics:
            - System Performance Metrics
            - Resource Utilization
            - Operation Logs
            - Error Patterns
            """
            
            # Create metadata for better retrieval - convert lists to strings
            metadata = {
                "type": entry['type'],
                "severity": entry['severity'],
                "indicators": "; ".join(f"{key}: {value}" for key, value in entry['indicators'].items()),  # Convert dictionary to semicolon-separated string
                "context": entry['context']
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)

        # Create vector store with metadata
        self.vector_store = Chroma.from_texts(
            documents,
            self.embeddings,
            metadatas=metadatas,
            collection_name="fault_patterns"
        )

    def _extract_timestamp(self, log_line: str) -> str:
        """Extract timestamp from log line."""
        try:
            # Extract timestamp between first [] brackets
            return log_line.split('[')[1].split(']')[0]
        except:
            return datetime.now().isoformat()

    def _analyze_metrics_with_rag(self, metrics: Dict[str, float], log_lines: List[str], timestamp: str) -> List[Dict[str, Any]]:
        """Use RAG to analyze metrics and detect faults based on semantic pattern matching."""
        faults = []
        
        # Create a rich semantic query that captures the system state
        query = f"""
        System State Analysis:
        
        Performance Metrics:
        - Latency: {metrics.get('Latency', 0)}ms (Normal range: 0-5ms)
        - System Saturation: {metrics.get('Saturation', 0)}% (Threshold: 100%)
        - Total Capacity Usage: {metrics.get('Total Capacity_percentage', 0)}% (Threshold: 100%)
        - Throughput: {metrics.get('Throughput', 0)} MB/s
        
        System Behavior Indicators:
        - Cleanup Operations: {sum(1 for line in log_lines if '[CLEANUP]' in line)} attempts
        - Failed Cleanup Operations: {sum(1 for line in log_lines if '[CLEANUP]' in line and ('failed' in line.lower() or 'error' in line.lower()))} failures
        - Write Operations: {sum(1 for line in log_lines if 'write' in line.lower())} operations
        - Queue Events: {sum(1 for line in log_lines if 'queue' in line.lower())} events
        - I/O Wait Events: {{sum(1 for line in log_lines if 'io wait' in line.lower() or 'io_wait' in line.lower())}} events
        - Replication Issues: {sum(1 for line in log_lines if 'replication' in line.lower() and ('error' in line.lower() or 'failed' in line.lower()))} errors
        - Sync Delays: {sum(1 for line in log_lines if 'sync' in line.lower() and ('delay' in line.lower() or 'timeout' in line.lower()))} delays
        
        Recent Log Patterns:
        {chr(10).join(log_lines[-5:])}
        """
        
        # Get semantically similar patterns with metadata
        similar_patterns = self.vector_store.similarity_search_with_score(
            query,
            k=3,
            filter={"severity": "high"}  # Focus on high severity patterns first
        )
        
        # Track detected issues to avoid duplicates
        detected_issues = set()
        
        for pattern, score in similar_patterns:
            pattern_text = pattern.page_content
            metadata = pattern.metadata
            
            # Extract pattern information
            pattern_type = metadata['type']
            pattern_severity = metadata['severity']
            pattern_context = metadata['context']
            
            # Calculate confidence based on semantic similarity and metric alignment
            base_confidence = 1.0 - score  # Convert distance to confidence
            
            # Analyze based on pattern type and metrics
            if pattern_type == "saturation_induced_latency":
                if (metrics.get('Latency', 0) >=5 and 
                    metrics.get('Saturation', 0) >=100 and
                    "saturation_induced_latency" not in detected_issues):
                    
                    # Calculate metric alignment score
                    metric_alignment = min(
                        (metrics.get('Saturation', 0) - 100) / 20,  # Normalize saturation excess
                        (metrics.get('Latency', 0) - 5) / 7,  # Normalize latency excess
                        1.0
                    )
                    
                    confidence = base_confidence * (0.7 + 0.3 * metric_alignment)
                    
                    details = (
                        f"Saturation-induced high latency detected:\n"
                        f"- Current latency: {metrics['Latency']}ms (threshold: 5ms)\n"
                        f"- System saturation: {metrics['Saturation']}% (threshold: 100%)\n"
                        f"- Queue events: {sum(1 for line in log_lines if 'queue' in line.lower())}\n"
                        f"- I/O wait events: {sum(1 for line in log_lines if 'io wait' in line.lower() or 'io_wait' in line.lower())}\n"
                        f"Semantic similarity score: {1.0 - score:.2f}\n"
                        f"Metric alignment score: {metric_alignment:.2f}\n"
                        f"This indicates performance degradation due to system resource exhaustion."
                    )
                    
                    faults.append({
                        "type": pattern_type,
                        "severity": pattern_severity,
                        "details": details,
                        "context": pattern_context,
                        "confidence": confidence,
                        "timestamp": timestamp,
                        "related_metrics": {
                            "latency": metrics['Latency'],
                            "saturation": metrics.get('Saturation', 0),
                            "queue_events": sum(1 for line in log_lines if 'queue' in line.lower()),
                            "io_wait_events": sum(1 for line in log_lines if 'io wait' in line.lower() or 'io_wait' in line.lower())
                        }
                    })
                    detected_issues.add("saturation_induced_latency")
            
            elif pattern_type == "capacity_induced_latency":
                if (metrics.get('Latency', 0) >=5 and 
                    metrics.get('Total Capacity_percentage', 0) > 100 and
                    "capacity_induced_latency" not in detected_issues):
                    
                    # Calculate metric alignment score
                    metric_alignment = min(
                        (metrics.get('Total Capacity_percentage', 0) - 100) / 10,  # Normalize capacity excess
                        (metrics.get('Latency', 0) - 5) / 7,  # Normalize latency excess
                        1.0
                    )
                    
                    confidence = base_confidence * (0.7 + 0.3 * metric_alignment)
                    
                    details = (
                        f"Capacity-induced high latency detected:\n"
                        f"- Current latency: {metrics['Latency']}ms (threshold: 5ms)\n"
                        f"- Storage capacity usage: {metrics['Total Capacity_percentage']}% (threshold: 100%)\n"
                        f"- Failed cleanup operations: {sum(1 for line in log_lines if '[CLEANUP]' in line and ('failed' in line.lower() or 'error' in line.lower()))}\n"
                        f"- Recent write operations: {sum(1 for line in log_lines if 'write' in line.lower())}\n"
                        f"Semantic similarity score: {1.0 - score:.2f}\n"
                        f"Metric alignment score: {metric_alignment:.2f}\n"
                        f"This indicates performance degradation due to storage system operating near capacity limits."
                    )
                    
                    faults.append({
                        "type": pattern_type,
                        "severity": pattern_severity,
                        "details": details,
                        "context": pattern_context,
                        "confidence": confidence,
                        "timestamp": timestamp,
                        "related_metrics": {
                            "latency": metrics['Latency'],
                            "capacity_usage": metrics['Total Capacity_percentage'],
                            "cleanup_failures": sum(1 for line in log_lines if '[CLEANUP]' in line and ('failed' in line.lower() or 'error' in line.lower())),
                            "write_operations": sum(1 for line in log_lines if 'write' in line.lower())
                        }
                    })
                    detected_issues.add("capacity_induced_latency")
            
            elif pattern_type == "replication_induced_latency":
                # Check for replication issues with high latency
                replication_issues = 0
                high_time_taken_count = 0
                time_taken_values = []
                
                # Direct pattern matching for replication link errors
                for line in log_lines:
                    # Check for replication line with TimeTaken >= 5ms
                    if "Active synchronous replication" in line and "TimeTaken" in line:
                        try:
                            # Extract time taken value - handle both formats
                            if "TimeTaken" in line:
                                # Handle format "TimeTaken: 0.03ms"
                                if "TimeTaken:" in line:
                                    time_taken_str = line.split("TimeTaken:")[1].split("ms")[0].strip()
                                else:
                                    # Handle format "TimeTaken 0.03ms"
                                    time_taken_str = line.split("TimeTaken")[1].split("ms")[0].strip()
                                
                                time_taken = float(time_taken_str)
                                time_taken_values.append(time_taken)
                                
                                if time_taken >= 5:
                                    high_time_taken_count += 1
                                    print(f"Found high TimeTaken: {time_taken}ms in line: {line}")
                        except Exception as e:
                            print(f"Error parsing TimeTaken value: {e}")
                            continue
                
                # Calculate average of last 4 TimeTaken values
                avg_time_taken = 0
                if time_taken_values:
                    last_4_values = time_taken_values[-4:] if len(time_taken_values) >= 4 else time_taken_values
                    avg_time_taken = sum(last_4_values) / len(last_4_values)
                
                # Count general replication errors
                replication_issues = sum(1 for line in log_lines if 'replication' in line.lower() and ('error' in line.lower() or 'failed' in line.lower()))
                sync_delays = sum(1 for line in log_lines if 'sync' in line.lower() and ('delay' in line.lower() or 'timeout' in line.lower()))
                
                # Debug output
                print(f"Replication check for {timestamp}: high_time_taken={high_time_taken_count}, replication_issues={replication_issues}, sync_delays={sync_delays}, avg_time_taken={avg_time_taken:.3f}ms")
                
                # Report replication-induced latency if we have high TimeTaken events
                if (high_time_taken_count >= 1 and 
                    "replication_induced_latency" not in detected_issues):
                    
                    # Calculate metric alignment score
                    metric_alignment = min(
                        high_time_taken_count / 5,  # Normalize replication count
                        1.0
                    )
                    
                    confidence = base_confidence * (0.7 + 0.3 * metric_alignment)
                    
                    details = (
                        f"Replication-induced latency detected:\n"
                        f"- Average TimeTaken (last 4): {avg_time_taken:.3f}ms\n"
                        f"This indicates potential network or resource constraints affecting synchronous replication performance."
                    )
                    
                    faults.append({
                        "type": pattern_type,
                        "severity": pattern_severity,
                        "details": details,
                        "context": pattern_context,
                        "confidence": confidence,
                        "timestamp": timestamp,
                        "related_metrics": {
                            "avg_time_taken": avg_time_taken,
                            "high_time_taken_events": high_time_taken_count
                        }
                    })
                    detected_issues.add("replication_induced_latency")
        
        return faults

    def format_fault_report(self, analysis_result: Dict[str, Any]) -> str:
        """Format the fault detection results in a readable way."""
        formatted_report = []
        
        # Add header
        formatted_report.append("=== STORAGE SYSTEM FAULT ANALYSIS REPORT ===")
        formatted_report.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        formatted_report.append("")
        
        # Process faults
        if analysis_result.get("faults"):
            formatted_report.append("DETECTED FAULTS:")
            formatted_report.append("=" * 50)
            
            for i, fault in enumerate(analysis_result["faults"], 1):
                formatted_report.append(f"FAULT #{i}: {fault['type'].upper()}")
                formatted_report.append("-" * 50)
                
                # Basic information
                formatted_report.append(f"Instance: {fault['instance']}")
                formatted_report.append(f"Severity: {fault['severity'].upper()}")
                formatted_report.append(f"Timestamp: {fault['timestamp']}")
                formatted_report.append("")
                
                # Details - customize based on fault type
                formatted_report.append("DETAILS:")
                if fault['type'] == 'capacity_induced_latency':
                    formatted_report.append(f"Capacity-induced high latency detected")
                    formatted_report.append(f"- Current latency {fault['related_metrics']['latency']}ms (threshold 5ms)")
                    formatted_report.append(f"- Storage capacity usage {fault['related_metrics']['capacity_usage']}% (threshold 100%)")
                elif fault['type'] == 'saturation_induced_latency':
                    formatted_report.append(f"Saturation-induced high latency detected")
                    formatted_report.append(f"- Current latency {fault['related_metrics']['latency']}ms (threshold 5ms)")
                    formatted_report.append(f"- System saturation {fault['related_metrics']['saturation']}% (threshold 100%)")
                elif fault['type'] == 'replication_induced_latency':
                    formatted_report.append(f"Replication-induced latency detected")
                    formatted_report.append(f"- Average TimeTaken (last 4): {fault['related_metrics']['avg_time_taken']:.3f}ms")
                    formatted_report.append(f"This indicates potential network or resource constraints affecting synchronous replication performance.")
                formatted_report.append("")
                
                # Context
                formatted_report.append("CONTEXT:")
                formatted_report.append(f"  {fault['context']}")
                formatted_report.append("")
                
                # Sample logs
                formatted_report.append("RECENT LOGS:")
                for log in fault.get('sample_logs', []):
                    formatted_report.append(f"  {log.strip()}")
                formatted_report.append("")
                
                formatted_report.append("=" * 50)
                formatted_report.append("")
        else:
            formatted_report.append("No faults detected in the system.")
            formatted_report.append("")
        
        # System state summary
        formatted_report.append("SYSTEM STATE SUMMARY:")
        formatted_report.append("=" * 50)
        
        for instance, state in analysis_result.get("system_state", {}).items():
            formatted_report.append(f"Instance: {instance}")
            formatted_report.append(f"Timestamp: {state['timestamp']}")
            formatted_report.append("")
            
            formatted_report.append("Current Metrics:")
            for metric, value in state['metrics'].items():
                formatted_report.append(f"  {metric}: {value}")
            formatted_report.append("")
            
            formatted_report.append("Recent Logs:")
            for log in state.get('latest_logs', []):
                formatted_report.append(f"  {log.strip()}")
            formatted_report.append("")
            
            formatted_report.append("-" * 50)
            formatted_report.append("")
        
        return "\n".join(formatted_report)

    def analyze_system_state(self) -> Dict[str, Any]:
        """Analyze the current system state and detect potential faults using RAG."""
        all_faults = []
        latest_system_state = {}
        
        # Check each data instance directory
        for instance_dir in os.listdir(self.data_dir):
            if instance_dir.startswith("data_instance_"):
                instance_path = os.path.join(self.data_dir, instance_dir)
                instance_number = instance_dir.split('_')[2]
                
                try:
                    log_file = os.path.join(instance_path, f"logs_{instance_number}.txt")
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_lines = f.readlines()
                            
                            # Track metrics over time
                            metrics_history = []
                            latest_logs = log_lines[-10:]
                            latest_timestamp = None
                            
                            for line in log_lines:
                                if "[INFO] System metrics updated" in line:
                                    try:
                                        # Extract timestamp
                                        timestamp = self._extract_timestamp(line)
                                        latest_timestamp = timestamp
                                        
                                        # Extract metrics from the log line
                                        metrics = {}
                                        metrics_part = line.split("System metrics updated - ")[1].strip()
                                        
                                        # Parse metrics
                                        metrics_list = metrics_part.split(", ")
                                        for metric in metrics_list:
                                            key_value = metric.split(": ")
                                            if len(key_value) == 2:
                                                key, value = key_value
                                                
                                                if "(" in value and ")" in value:
                                                    base_value = value.split(" (")[0].strip()
                                                    numeric_value = float(base_value.split(" ")[0])
                                                    metrics[key] = numeric_value
                                                    
                                                    percentage = value.split("(")[1].split(")")[0].strip("%")
                                                    metrics[f"{key}_percentage"] = float(percentage)
                                                else:
                                                    value_parts = value.strip().split(" ")
                                                    numeric_part = value_parts[0]
                                                    
                                                    if key == "Latency":
                                                        numeric_part = numeric_part.replace("ms", "")
                                                    elif "MB/s" in value:
                                                        numeric_part = numeric_part.replace("MB/s", "")
                                                    elif "GB" in value:
                                                        numeric_part = numeric_part.replace("GB", "")
                                                    elif "%" in value:
                                                        numeric_part = numeric_part.replace("%", "")
                                                        
                                                    metrics[key] = float(numeric_part)
                                                    if len(value_parts) > 1:
                                                        metrics[f"{key}_unit"] = value_parts[1]
                                        
                                        metrics_history.append((timestamp, metrics))
                                    except Exception as e:
                                        print(f"Warning: Error parsing metrics from line: {line}\nError: {str(e)}")
                                        continue
                            
                            # Store latest state and analyze with RAG
                            if metrics_history:
                                latest_timestamp, latest_metrics = metrics_history[-1]
                                latest_system_state[instance_dir] = {
                                    "metrics": latest_metrics,
                                    "latest_logs": latest_logs,
                                    "timestamp": latest_timestamp
                                }
                                
                                # Use RAG to analyze metrics and detect faults
                                instance_faults = self._analyze_metrics_with_rag(latest_metrics, latest_logs, latest_timestamp)
                                for fault in instance_faults:
                                    fault["instance"] = instance_dir
                                    fault["metrics"] = latest_metrics
                                    fault["sample_logs"] = latest_logs[-3:]
                                    all_faults.append(fault)
                
                except Exception as e:
                    print(f"Warning: Error processing logs for {instance_dir}: {str(e)}")
                    continue
        
        result = {
            "faults": all_faults,
            "system_state": latest_system_state
        }
        
        # Format the report
        formatted_report = self.format_fault_report(result)
        print(formatted_report)
        
        return result 