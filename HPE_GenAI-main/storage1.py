import json
import os
import uuid
import threading
import time
import random
from datetime import datetime
import requests

class StorageManager:
    def __init__(self, data_dir, global_file="global_systems.json", logger=None):
        self.data_dir = data_dir
        self.global_file = global_file
        self.logger = logger
        self.metrics_file = os.path.join(data_dir, f"system_metrics_{self.get_port()}.json")
        self.replication_metrics_file = os.path.join(data_dir, f"replication_metrics_{self.get_port()}.json")
        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(self.global_file) or os.stat(self.global_file).st_size == 0:
            with open(self.global_file, "w") as f:
                json.dump([], f, indent=4)

        # Initialize metrics files if they don't exist
        if not os.path.exists(self.metrics_file):
            self.save_metrics({"throughput_used": 0, "capacity_used": 0})
        if not os.path.exists(self.replication_metrics_file):
            self.save_replication_metrics({})

        # Dictionary to keep track of ongoing replication tasks (one per volume)
        self.replication_tasks = {}

    def get_port(self):
        return self.data_dir.split('_')[-1]

    def save_metrics(self, metrics):
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    def load_metrics(self):
        if not os.path.exists(self.metrics_file):
            return {"throughput_used": 0, "capacity_used": 0}
        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def update_capacity_used(self, size_gb):
        metrics = self.load_metrics()
        metrics["capacity_used"] += size_gb
        self.save_metrics(metrics)
        return metrics["capacity_used"]

    def load_resource(self, resource_type):
        file_path = os.path.join(self.data_dir, f"{resource_type}.json")
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump([], f, indent=4)
            return []

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def save_resource(self, resource_type, data):
        file_path = os.path.join(self.data_dir, f"{resource_type}.json")
        existing_data = self.load_resource(resource_type)

        if not isinstance(existing_data, list):
            print(f"Warning: {resource_type}.json is not a list. Resetting to an empty list.")
            existing_data = []
        if isinstance(data, dict):
            if any(item["id"] == data["id"] for item in existing_data):
                raise ValueError(f"{resource_type} with ID {data['id']} already exists.")

        existing_data.append(data)
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    def add_system_to_global(self, system_id, system_name, port):
        with open(self.global_file, "r") as f:
            global_systems = json.load(f)

        if any(s["id"] == system_id for s in global_systems):
            return
        
        global_systems.append({"id": system_id, "name": system_name, "port": port})
        with open(self.global_file, "w") as f:
            json.dump(global_systems, f, indent=4)

    def get_all_systems(self):
        with open(self.global_file, "r") as f:
            return json.load(f)

    def update_resource(self, resource_type, resource_id, updated_data):
        file_path = os.path.join(self.data_dir, f"{resource_type}.json")
        existing_data = self.load_resource(resource_type)
        for i, item in enumerate(existing_data):
            if item["id"] == resource_id:
                existing_data[i] = updated_data
                break
        try:
            with open(file_path, "w") as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            raise Exception(f"Failed to update {resource_type}: {str(e)}")

    def delete_resource(self, resource_type, resource_id):
        file_path = os.path.join(self.data_dir, f"{resource_type}.json")
        existing_data = self.load_resource(resource_type)
        if resource_id is None:
            existing_data = []
        else:
            existing_data = [item for item in existing_data if item["id"] != resource_id]
        try:
            with open(file_path, "w") as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            raise Exception(f"Failed to delete {resource_type}: {str(e)}")
    
    def remove_system_from_global(self, system_id):
        """Removes a system from global_systems.json when deleted."""
        try:
            with open(self.global_file, "r") as f:
                global_systems = json.load(f)

            # Remove the system with the matching ID
            updated_systems = [sys for sys in global_systems if sys["id"] != system_id]

            with open(self.global_file, "w") as f:
                json.dump(updated_systems, f, indent=4)

            print(f"System {system_id} removed from global_systems.json")

        except Exception as e:
            raise Exception(f"Failed to remove system from global tracking: {str(e)}")
    def delete_related_resources(self, resource_type, system_id):
        """Deletes all resources (nodes, volumes, settings) associated with a system."""
        file_path = os.path.join(self.data_dir, f"{resource_type}.json")
        existing_data = self.load_resource(resource_type)

        # Keep only resources that DO NOT belong to the deleted system
        updated_data = [item for item in existing_data if item["system_id"] != system_id]

        try:
            with open(file_path, "w") as f:
                json.dump(updated_data, f, indent=4)

            print(f"All {resource_type} related to system {system_id} deleted.")

        except Exception as e:
            raise Exception(f"Failed to delete {resource_type} for system {system_id}: {str(e)}")
        
    def update_snapshot_in_settings(self, system_id, volume_id, snapshot_frequency):
        """Ensures snapshot settings for the volume are stored in settings.json."""
        file_path = os.path.join(self.data_dir, "settings.json")

        # If settings.json does not exist, create it
        if not os.path.exists(file_path):
            print("settings.json does not exist, creating a new file...")
            with open(file_path, "w") as f:
                json.dump([], f, indent=4)

        settings = self.load_resource("settings")

        # Find or create the system settings entry
        system_setting = next((s for s in settings if s["system_id"] == system_id), None)

        if not system_setting:
            print(f"No settings found for system {system_id}, creating a new entry.")
            system_setting = {
                "id": str(uuid.uuid4()),  # Generate a unique settings ID
                "system_id": system_id,
                "volume_snapshots": {}  # Initialize snapshot tracking
            }
            settings.append(system_setting)  # Add new settings entry

        # Update snapshot settings for the specific volume
        system_setting["volume_snapshots"][volume_id] = snapshot_frequency
        print(f"Updated settings: {settings}")

        # Save changes back to settings.json
        try:
            with open(file_path, "w") as f:
                json.dump(settings, f, indent=4)
            print(f"Snapshot settings updated successfully for volume {volume_id} in system {system_id}")

        except Exception as e:
            raise Exception(f"Failed to update snapshot settings in settings.json: {str(e)}")
    
    def update_replication_in_settings(self, system_id, replication_type, replication_target, replication_frequency):
        """Updates replication type and frequency in settings.json."""
        settings = self.load_resource("settings")

        # Find system settings entry
        system_setting = next((s for s in settings if s["system_id"] == system_id), None)

        if not system_setting:
            system_setting = {
                "id": str(uuid.uuid4()),
                "system_id": system_id,
                "replication_type": replication_type,
                "replication_target": replication_target
            }
            settings.append(system_setting)

        # Update replication type & target
        system_setting["replication_type"] = replication_type
        system_setting["replication_target"] = replication_target

        # Update frequency if async
        if replication_type == "asynchronous":
            system_setting["replication_frequency"] = replication_frequency
        else:
            system_setting.pop("replication_frequency", None)

        # Save changes
        try:
            file_path = os.path.join(self.data_dir, "settings.json")
            with open(file_path, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            raise Exception(f"Failed to update replication settings in settings.json: {str(e)}")
        

    def export_volume(self, volume_id, host_id, workload_size):
        print(f"Exporting volume {volume_id} to host {host_id}")  # Debug log

        # Load volumes and hosts
        volumes = self.load_resource("volume")
        hosts = self.load_resource("host")

        # Find the volume and host
        volume = next((v for v in volumes if v["id"] == volume_id), None)
        host = next((h for h in hosts if h["id"] == host_id), None)

        if not volume or not host:
            raise ValueError("Invalid volume or host ID")

        # Check if volume is already exported
        if volume.get("is_exported"):
            raise ValueError("Volume is already exported")

        # Mark volume as exported
        volume["is_exported"] = True
        volume["exported_host_id"] = host_id
        volume["workload_size"] = workload_size

        print(f"Starting Host I/O for volume {volume_id}")

        # Use update_resource() instead of save_resource()
        self.update_resource("volume", volume_id, volume)  # Updates only this volume

        # Start background tasks: host I/O, snapshots, and replication.
        self.start_host_io(volume_id)
        if volume.get("snapshot_settings"):
            self.start_snapshot(volume_id)
        # If replication settings exist, start replication for this volume.
        if volume.get("replication_settings"):
            self.start_replication(volume_id)

        return f"Volume {volume_id} exported successfully to Host {host_id}"

    def start_host_io(self, volume_id):
        """Simulate I/O operations for a volume using logger"""
        print(f"Host I/O started for volume {volume_id}")

        def io_worker():
            try:
                while True:
                    volumes = self.load_resource("volume")
                    volume = next((v for v in volumes if v["id"] == volume_id), None)

                    if not volume or not volume.get("is_exported", False):
                        break

                    host_id = volume.get("exported_host_id", "Unknown")
                    io_count = random.randint(100, 1000)
                    latency = round(random.uniform(1.0, 10.0), 2)
                    throughput = round(io_count / latency, 2)

                    # Log using the logger (both local and global)
                    log_message = (
                        f"Volume: {volume_id}, Host: {host_id}, "
                        f"IOPS: {io_count}, Latency: {latency}ms, Throughput: {throughput} MB/s"
                    )
                    if self.logger:
                        self.logger.info(log_message, global_log=True)

                    # Update metrics (renamed to io_metrics)
                    metrics = self.load_resource("io_metrics")
                    if not isinstance(metrics, list):  
                        metrics = []

                    metrics.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "volume_id": volume_id,
                        "host_id": host_id,
                        "io_count": io_count,
                        "latency": latency,
                        "throughput": throughput
                    })
                    self.save_resource("io_metrics", metrics)

                    time.sleep(30)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Host I/O error: {str(e)}", global_log=True)

        worker_thread = threading.Thread(target=io_worker, daemon=True)
        worker_thread.start()

    def unexport_volume(self, volume_id, reason="Manual unexport"):
        """
        Unexport a volume and cleanup all associated processes
        """
        volumes = self.load_resource("volume")
        volume = next((v for v in volumes if v["id"] == volume_id), None)
        if not volume:
            raise ValueError("Invalid volume ID")
        if not volume.get("is_exported", False):
            raise ValueError("Volume is not exported")

        # First cleanup all processes
        self.cleanup_volume_processes(volume_id, reason=reason)

        # Then update volume state
        volume["is_exported"] = False
        volume["exported_host_id"] = None
        volume["workload_size"] = None

        self.logger.info(f"Volume {volume_id} unexported: {reason}", global_log=True)
        self.update_resource("volume", volume_id, volume)
        return f"Volume {volume_id} unexported successfully"

    def start_replication(self, volume_id):
        """
        Starts a replication process for the given volume_id if replication settings exist.
        """
        # If a replication task for this volume is already running, do nothing.
        if volume_id in self.replication_tasks:
            return

        # Create an Event to signal termination of the replication thread.
        stop_event = threading.Event()
        self.replication_tasks[volume_id] = stop_event

        # Start main replication coordinator thread
        thread = threading.Thread(target=self.replication_coordinator, args=(volume_id, stop_event), daemon=True)
        thread.start()

    def replication_coordinator(self, volume_id, stop_event):
        """
        Coordinates replication to multiple targets, spawning a worker thread for each target.
        """
        worker_threads = {}  # Keep track of worker threads by target_id

        while not stop_event.is_set():
            # Reload volume to check current state
            volumes = self.load_resource("volume")
            volume = next((v for v in volumes if v["id"] == volume_id), None)

            if not volume or not volume.get("is_exported") or not volume.get("replication_settings"):
                break

            # Get current replication settings
            current_settings = volume.get("replication_settings", [])
            current_target_ids = {s.get("replication_target", {}).get("id") for s in current_settings 
                                if s.get("replication_target", {}).get("id") is not None}

            # Stop threads for removed targets
            for target_id in list(worker_threads.keys()):
                if target_id not in current_target_ids:
                    worker_threads[target_id]["stop_event"].set()
                    worker_threads[target_id]["thread"].join(timeout=1)
                    del worker_threads[target_id]

            # Start new threads for new targets
            for rep_setting in current_settings:
                target_id = rep_setting.get("replication_target", {}).get("id")
                if target_id is not None and target_id not in worker_threads:
                    target_stop_event = threading.Event()
                    worker_thread = threading.Thread(
                        target=self.replication_worker,
                        args=(volume_id, target_stop_event, rep_setting),
                        daemon=True
                    )
                    worker_threads[target_id] = {
                        "thread": worker_thread,
                        "stop_event": target_stop_event
                    }
                    worker_thread.start()

            time.sleep(5)  # Check for changes every 5 seconds

        # Clean up all worker threads
        for worker in worker_threads.values():
            worker["stop_event"].set()
            worker["thread"].join(timeout=1)

        if volume_id in self.replication_tasks:
            del self.replication_tasks[volume_id]

    def replication_worker(self, volume_id, stop_event, rep_setting):
        """
        Worker function that simulates replication of a volume to a specific target.
        """
        replication_type = rep_setting.get("replication_type")
        target = rep_setting.get("replication_target", {})
        target_id = target.get("id")
        last_log_time = 0
        SYNC_LOG_INTERVAL = 200  # Log every 200 seconds for sync replication

        # Log replication start
        start_log = (f"Started {replication_type} replication for volume {volume_id} "
                     f"to target {target.get('name')}")
        self.logger.info(start_log, global_log=True)

        while not stop_event.is_set():
            # Reload volumes to check current state.
            volumes = self.load_resource("volume")
            volume = next((v for v in volumes if v["id"] == volume_id), None)
            if not volume or not volume.get("is_exported"):
                break

            delay_sec = rep_setting.get("delay_sec", 0)
            
            # Simulate replication throughput calculation.
            io_count = random.randint(50, 500)
            latency = round(random.uniform(1.0, 5.0), 2)
            replication_throughput = round(io_count / latency, 2)  # MB/s
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update replication metrics
            metrics = {
                "throughput": replication_throughput,
                "latency": latency,
                "io_count": io_count,
                "replication_type": replication_type,
                "timestamp": timestamp
            }
            self.update_replication_metrics(volume_id, target_id, metrics)

            current_time = time.time()
            should_log = (
                replication_type != "synchronous" or  # Always log async
                last_log_time == 0 or  # First log
                (current_time - last_log_time) >= SYNC_LOG_INTERVAL  # Periodic sync log
            )

            if should_log:
                # Log sender replication event.
                if replication_type == "synchronous":
                    sender_log = (f"Active synchronous replication for volume {volume_id} "
                                f"to target {target.get('name')} - "
                                f"Throughput: {replication_throughput} MB/s, Latency: {latency}ms")
                else:
                    sender_log = (f"Replicating volume {volume_id} with throughput "
                                f"{replication_throughput} MB/s to target {target.get('name')}")
                if self.logger:
                    self.logger.info(sender_log, global_log=True)
                last_log_time = current_time

            # Determine target endpoint by looking up the target system in global systems.
            try:
                global_systems = self.get_all_systems()
                target_sys = next((s for s in global_systems if s["id"] == target_id), None)
                if target_sys:
                    target_port = target_sys["port"]
                    target_url = f"http://localhost:{target_port}/replication-receive"
                    payload = {
                        "volume_id": volume_id,
                        "replication_throughput": replication_throughput,
                        "sender": self.data_dir,
                        "timestamp": timestamp,
                        "replication_type": replication_type,
                        "should_log": should_log,  # Tell target whether to log
                        "latency": latency
                    }
                    response = requests.post(target_url, json=payload, timeout=5)
                    if response.status_code != 200:
                        self.logger.warn(f"Failed to deliver replication data to target {target.get('name')}: {response.text}", global_log=True)
                else:
                    self.logger.warn(f"Target system with id {target_id} not found", global_log=True)
            except Exception as ex:
                self.logger.error(f"Replication error for volume {volume_id}: {str(ex)}", global_log=True)

            # Wait based on replication type and delay setting
            wait_time = delay_sec if replication_type == "asynchronous" and delay_sec > 0 else 10
            time.sleep(wait_time)

        # Log replication stop
        stop_log = f"Stopped {replication_type} replication for volume {volume_id} to target {target.get('name')}"
        self.logger.info(stop_log, global_log=True)

    def cleanup_volume_processes(self, volume_id, reason="", notify_targets=True):
        """
        Cleanup all processes for a volume and notify targets if needed
        """
        try:
            volume = next((v for v in self.load_resource("volume") if v["id"] == volume_id), None)
            if not volume:
                return

            # Stop replication tasks if running
            if volume_id in self.replication_tasks:
                self.replication_tasks[volume_id].set()  # Signal thread to stop
                if volume.get("replication_settings") and notify_targets:
                    # Notify all targets about replication stop
                    for rep_setting in volume.get("replication_settings", []):
                        target = rep_setting.get("replication_target", {})
                        target_port = next((s["port"] for s in self.get_all_systems() 
                                         if s["id"] == target.get("id")), None)
                        if target_port:
                            try:
                                url = f"http://localhost:{target_port}/replication-stop"
                                requests.post(url, json={
                                    "volume_id": volume_id,
                                    "reason": reason,
                                    "sender": self.data_dir
                                }, timeout=5)
                            except Exception as e:
                                self.logger.error(f"Failed to notify target {target.get('name')}: {str(e)}", 
                                               global_log=True)

            # Log the cleanup
            self.logger.info(f"Stopped all processes for volume {volume_id}: {reason}", global_log=True)

        except Exception as e:
            self.logger.error(f"Error during cleanup for volume {volume_id}: {str(e)}", global_log=True)

    def save_replication_metrics(self, metrics):
        with open(self.replication_metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    def load_replication_metrics(self):
        if not os.path.exists(self.replication_metrics_file):
            return {}
        with open(self.replication_metrics_file, 'r') as f:
            return json.load(f)

    def update_replication_metrics(self, volume_id, target_id, metrics):
        """Update replication metrics for a specific volume-target pair"""
        all_metrics = self.load_replication_metrics()
        if volume_id not in all_metrics:
            all_metrics[volume_id] = {}
        all_metrics[volume_id][target_id] = {
            **metrics,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_replication_metrics(all_metrics)

class Settings:
    def __init__(self, id, system_id):
        self.id = id
        self.system_id = system_id
        

    def to_dict(self):
        return {
            "id": self.id,
            "system_id": self.system_id,
            
        }
