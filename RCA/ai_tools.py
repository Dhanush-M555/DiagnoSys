import json
import os
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any, Optional

class StorageTools:
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.system_id_to_name = self._load_system_mapping()
        
    def _load_system_mapping(self) -> Dict[str, str]:
        mapping = {}
        global_file = os.path.join(self.data_dir, "global_systems.json")
        if os.path.exists(global_file):
            try:
                with open(global_file, 'r') as f:
                    systems = json.load(f)
                for system in systems:
                    if "id" in system and "name" in system:
                        mapping[system["id"]] = str(system["name"])
            except Exception as e:
                print(f"Error loading system mapping from {global_file}: {e}")
        else:
            print(f"Warning: {global_file} not found. Cannot map system IDs to names.")
        print(f"DEBUG - Loaded system mapping: {mapping}")
        return mapping

    def _get_system_name(self, system_id: str) -> Optional[str]:
        system_name = self.system_id_to_name.get(system_id)
        if not system_name:
            print(f"Error: System ID {system_id} not found in mapping.")
            return None
        return system_name

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Returns the tool definitions for use with Groq API
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_io_metrics",
                    "description": "Fetch IO metrics for the specified time range and filters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            },
                            "start_timestamp": {
                                "type": "string",
                                "description": "Start time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_timestamp": {
                                "type": "string",
                                "description": "End time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "volume_id": {
                                "type": "string",
                                "description": "Optional: Filter by volume ID"
                            },
                            "host_id": {
                                "type": "string",
                                "description": "Optional: Filter by host ID"
                            },
                            "min_latency": {
                                "type": "number",
                                "description": "Optional: Filter metrics by minimum latency threshold (in ms)"
                            }
                        },
                        "required": ["system_id", "start_timestamp", "end_timestamp"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_system_metrics",
                    "description": "Fetch system-wide metrics for the specified time range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            },
                            "start_timestamp": {
                                "type": "string",
                                "description": "Start time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_timestamp": {
                                "type": "string",
                                "description": "End time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "min_latency": {
                                "type": "number",
                                "description": "Optional: Filter metrics by minimum latency threshold (in ms)"
                            }
                        },
                        "required": ["system_id", "start_timestamp", "end_timestamp"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_replication_metrics",
                    "description": "Fetch replication metrics for the specified time range and filters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            },
                            "start_timestamp": {
                                "type": "string",
                                "description": "Start time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_timestamp": {
                                "type": "string",
                                "description": "End time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "volume_id": {
                                "type": "string",
                                "description": "Optional: Filter by volume ID"
                            },
                            "target_system_id": {
                                "type": "string",
                                "description": "Optional: Filter by target system ID"
                            },
                            "min_latency": {
                                "type": "number",
                                "description": "Optional: Filter metrics by minimum latency threshold (in ms)"
                            }
                        },
                        "required": ["system_id", "start_timestamp", "end_timestamp"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_logs",
                    "description": "Fetch logs for the specified time range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            },
                            "start_timestamp": {
                                "type": "string",
                                "description": "Start time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "end_timestamp": {
                                "type": "string",
                                "description": "End time in ISO format (YYYY-MM-DD HH:MM:SS)"
                            },
                            "log_level": {
                                "type": "string",
                                "description": "Optional: Filter logs by severity level (INFO, WARN, CLEANUP, ERROR)",
                                "enum": ["INFO", "WARN", "CLEANUP", "ERROR"]
                            }
                        },
                        "required": ["system_id", "start_timestamp", "end_timestamp"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_system_config",
                    "description": "Fetch system configuration",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            }
                        },
                        "required": ["system_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_volumes",
                    "description": "Fetch volumes configuration",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            }
                        },
                        "required": ["system_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_hosts",
                    "description": "Fetch hosts configuration",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_id": {
                                "type": "string",
                                "description": "ID of the storage system"
                            }
                        },
                        "required": ["system_id"]
                    }
                }
            }
        ]
    
    def fetch_io_metrics(self, system_id: str, start_timestamp: str, end_timestamp: str, 
                          volume_id: Optional[str] = None, host_id: Optional[str] = None,
                          min_latency: Optional[float] = None) -> Dict[str, Any]:
        """Fetch IO metrics for the specified time range and filters"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            start_time = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
            
            io_metrics_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "io_metrics.json")
            print(f"DEBUG - Reading IO metrics from: {io_metrics_file}")
            
            if not os.path.exists(io_metrics_file):
                return {"error": f"IO metrics file not found for system {system_name} (ID: {system_id}) at path {io_metrics_file}"}
                
            with open(io_metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            filtered_metrics = []
            for metric in all_metrics:
                try:
                    metric_time = datetime.strptime(metric.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    if start_time <= metric_time <= end_time:
                        # Apply volume and host filters if provided
                        if (volume_id is None or metric.get("volume_id") == volume_id) and \
                           (host_id is None or metric.get("host_id") == host_id):
                            # Apply min_latency filter if provided
                            latency = metric.get("latency")
                            if min_latency is None or (latency is not None and latency >= min_latency):
                                filtered_metrics.append(metric)
                except (ValueError, TypeError):
                    continue
                    
            return {"io_metrics": filtered_metrics}
        except Exception as e:
            return {"error": f"Error fetching IO metrics for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_system_metrics(self, system_id: str, start_timestamp: str, end_timestamp: str,
                             min_latency: Optional[float] = None) -> Dict[str, Any]:
        """Fetch system metrics for the specified time range"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Convert string timestamps to datetime objects
            start_time = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
            
            # Use system_name to build the path
            metrics_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "system_metrics.json")
            print(f"DEBUG - Reading system metrics from: {metrics_file}")
            
            if not os.path.exists(metrics_file):
                return {"error": f"System metrics file not found for system {system_name} (ID: {system_id}) at path {metrics_file}"}
                
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            # Convert timestamps and filter by time range
            filtered_metrics = []
            for metric in all_metrics:
                try:
                    metric_time = datetime.strptime(metric.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    if start_time <= metric_time <= end_time:
                        # Apply min_latency filter if provided
                        current_latency = metric.get("current_latency")
                        if min_latency is None or (current_latency is not None and current_latency >= min_latency):
                            filtered_metrics.append(metric)
                except (ValueError, TypeError):
                    continue
                    
            return {"system_metrics": filtered_metrics}
        except Exception as e:
            return {"error": f"Error fetching system metrics for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_replication_metrics(self, system_id: str, start_timestamp: str, end_timestamp: str,
                                   volume_id: Optional[str] = None, target_system_id: Optional[str] = None,
                                   min_latency: Optional[float] = None) -> Dict[str, Any]:
        """Fetch replication metrics for the specified time range and filters"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Convert string timestamps to datetime objects
            start_time = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
            
            # Use system_name to build the path
            replication_metrics_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "replication_metrics.json")
            print(f"DEBUG - Reading replication metrics from: {replication_metrics_file}")
            
            if not os.path.exists(replication_metrics_file):
                return {"error": f"Replication metrics file not found for system {system_name} (ID: {system_id}) at path {replication_metrics_file}"}
                
            with open(replication_metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            # Convert timestamps and filter by time range
            filtered_metrics = []
            for metric in all_metrics:
                try:
                    metric_time = datetime.strptime(metric.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    if start_time <= metric_time <= end_time:
                        # Apply volume and target system filters if provided
                        if (volume_id is None or metric.get("volume_id") == volume_id) and \
                           (target_system_id is None or metric.get("target_system_id") == target_system_id):
                            # Apply min_latency filter if provided
                            replication_latency = metric.get("latency")
                            if min_latency is None or (replication_latency is not None and replication_latency >= min_latency):
                                filtered_metrics.append(metric)
                except (ValueError, TypeError):
                    continue
                    
            return {"replication_metrics": filtered_metrics}
        except Exception as e:
            return {"error": f"Error fetching replication metrics for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_logs(self, system_id: str, start_timestamp: str, end_timestamp: str, 
                    log_level: Optional[str] = None) -> Dict[str, Any]:
        """Fetch logs for the specified time range"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Convert string timestamps to datetime objects
            start_time = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
            
            # Use system_name to build the path
            log_file = os.path.join(self.data_dir, f"data_instance_{system_name}", f"logs_{system_name}.txt")
            print(f"DEBUG - Reading logs from: {log_file}")
            
            if not os.path.exists(log_file):
                return {"error": f"Log file not found for system {system_name} (ID: {system_id}) at path {log_file}"}
                
            with open(log_file, 'r') as f:
                all_logs = f.readlines()
            
            # Parse timestamps and filter by time range
            filtered_logs = []
            timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
            log_level_pattern = r'\[\w+\]\s+\[(INFO|WARN|CLEANUP|ERROR)\]'
            
            for log in all_logs:
                # Extract timestamp
                timestamp_match = re.search(timestamp_pattern, log)
                if timestamp_match:
                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                        # Filter by time range
                        if start_time <= log_time <= end_time:
                            # Filter by log level if specified
                            if log_level:
                                level_match = re.search(log_level_pattern, log)
                                if level_match and level_match.group(1) == log_level:
                                    filtered_logs.append(log.strip())
                            else:
                                # Include all logs if no log level filter
                                filtered_logs.append(log.strip())
                    except (ValueError, TypeError):
                        continue
                        
            return {"logs": filtered_logs}
        except Exception as e:
            return {"error": f"Error fetching logs for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_system_config(self, system_id: str) -> Dict[str, Any]:
        """Fetch system configuration"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Use system_name to build the path
            system_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "system.json")
            print(f"DEBUG - Reading system config from: {system_file}")
            
            if not os.path.exists(system_file):
                return {"error": f"System configuration file not found for system {system_name} (ID: {system_id}) at path {system_file}"}
                
            with open(system_file, 'r') as f:
                system_config = json.load(f)
                    
            return {"system_config": system_config}
        except Exception as e:
            return {"error": f"Error fetching system configuration for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_volumes(self, system_id: str) -> Dict[str, Any]:
        """Fetch volumes configuration"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Use system_name to build the path
            volume_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "volume.json")
            print(f"DEBUG - Reading volume config from: {volume_file}")
            
            if not os.path.exists(volume_file):
                return {"error": f"Volume configuration file not found for system {system_name} (ID: {system_id}) at path {volume_file}"}
                
            with open(volume_file, 'r') as f:
                volumes = json.load(f)
                    
            return {"volumes": volumes}
        except Exception as e:
            return {"error": f"Error fetching volumes for system {system_name} (ID: {system_id}): {str(e)}"}

    def fetch_hosts(self, system_id: str) -> Dict[str, Any]:
        """Fetch hosts configuration"""
        system_name = self._get_system_name(system_id)
        if not system_name:
             return {"error": f"System ID {system_id} not found in mapping."}
             
        try:
            # Use system_name to build the path
            host_file = os.path.join(self.data_dir, f"data_instance_{system_name}", "host.json")
            print(f"DEBUG - Reading host config from: {host_file}")
            
            if not os.path.exists(host_file):
                return {"error": f"Host configuration file not found for system {system_name} (ID: {system_id}) at path {host_file}"}
                
            with open(host_file, 'r') as f:
                hosts = json.load(f)
                    
            return {"hosts": hosts}
        except Exception as e:
            return {"error": f"Error fetching hosts for system {system_name} (ID: {system_id}): {str(e)}"}

    def handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specified tool call and return the result
        
        Args:
            tool_call: Tool call object from Groq API response
            
        Returns:
            Dictionary with the tool result
        """
        # This method is no longer used by ai_agent.py but kept for potential future use/reference
        try:
            # Debug: Print tool_call object details
            print(f"DEBUG - (handle_tool_call) TOOL CALL TYPE: {type(tool_call)}")
            # print(f"DEBUG - (handle_tool_call) TOOL CALL DIR: {dir(tool_call)}")
            
            # Handle either ChatCompletionMessageToolCall object or dictionary
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                # ChatCompletionMessageToolCall object (dot notation)
                print(f"DEBUG - (handle_tool_call) Using object attribute access for tool call")
                function_name = tool_call.function.name
                # print(f"DEBUG - (handle_tool_call) Function name: {function_name}")
                function_args = tool_call.function.arguments
                # print(f"DEBUG - (handle_tool_call) Function arguments (raw): {function_args}")
                arguments = json.loads(function_args) if function_args else {}
                # print(f"DEBUG - (handle_tool_call) Parsed arguments: {arguments}")
            elif isinstance(tool_call, dict) and "function" in tool_call:
                # Dictionary (subscript notation)
                print(f"DEBUG - (handle_tool_call) Using dictionary access for tool call")
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"]) if tool_call["function"].get("arguments") else {}
            else:
                print(f"DEBUG - (handle_tool_call) Unsupported tool call format: {type(tool_call)}")
                return {"error": f"Unsupported tool call format: {type(tool_call)}"}
            
            # Map function names to methods
            function_map = {
                "fetch_io_metrics": self.fetch_io_metrics,
                "fetch_system_metrics": self.fetch_system_metrics,
                "fetch_replication_metrics": self.fetch_replication_metrics,
                "fetch_logs": self.fetch_logs,
                "fetch_system_config": self.fetch_system_config,
                "fetch_volumes": self.fetch_volumes,
                "fetch_hosts": self.fetch_hosts
            }
            
            if function_name not in function_map:
                return {"error": f"Unknown tool function: {function_name}"}
                
            # Call the corresponding method with the provided arguments
            print(f"DEBUG - (handle_tool_call) Calling {function_name} with arguments: {arguments}")
            result = function_map[function_name](**arguments)
            # print(f"DEBUG - (handle_tool_call) Function result: {result}")
            return result
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"DEBUG - (handle_tool_call) Error in handle_tool_call: {str(e)}")
            print(f"DEBUG - (handle_tool_call) Traceback: {traceback_str}")
            return {"error": f"Error handling tool call: {str(e)}"} 