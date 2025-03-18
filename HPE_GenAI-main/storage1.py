
import json
import os
import uuid
import threading
import time
import random
from datetime import datetime

class StorageManager:
    def __init__(self, data_dir, global_file="global_systems.json"):
        self.data_dir = data_dir
        self.global_file = global_file
        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(self.global_file) or os.stat(self.global_file).st_size == 0:
            with open(self.global_file, "w") as f:
                json.dump([], f, indent=4)

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
        if volume.get("is_exported", False):
            raise ValueError("Volume is already exported")

        # Mark volume as exported
        volume["is_exported"] = True
        volume["exported_host_id"] = host_id
        volume["workload_size"] = workload_size

        print(f"Starting Host I/O for volume {volume_id}")

        # Use update_resource() instead of save_resource()
        self.update_resource("volume", volume_id, volume)  # Updates only this volume

        # Start background tasks (run host IO, snapshots, replication)
        self.start_host_io(volume_id)
        if volume.get("snapshot_settings"):
            self.start_snapshot(volume_id)
        if volume.get("replication_settings"):
            self.start_replication(volume_id)

        return f"Volume {volume_id} exported successfully to Host {host_id}"

    import threading
    import time
    import random
    from datetime import datetime

    def start_host_io(self, volume_id):
        """Simulate I/O operations for a volume and log data separately every 5 seconds."""
        print(f"Host I/O started for volume {volume_id}")  # Debug log

        def io_worker():
            try:
                while True:
                    print(f"Writing I/O metrics for volume {volume_id}")  # Debug log

                    volumes = self.load_resource("volume")
                    volume = next((v for v in volumes if v["id"] == volume_id), None)

                    if not volume or not volume.get("is_exported", False):
                        print(f"Stopping Host I/O for volume {volume_id}")  # Debug log
                        break  # Stop if volume is unexported

                    host_id = volume.get("exported_host_id", "Unknown")

                    io_count = random.randint(100, 1000)
                    latency = round(random.uniform(1.0, 10.0), 2)
                    throughput = round(io_count / latency, 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    io_log_entry = f"[{timestamp}] Volume: {volume_id}, Host: {host_id}, IOPS: {io_count}, Latency: {latency}ms, Throughput: {throughput} MB/s\n"
                    with open("io_log.txt", "a") as log_file:
                        log_file.write(io_log_entry)

                    print(f"Logged to io_log.txt for volume {volume_id}")  # Debug log

                    metrics = self.load_resource("metrics")
                    if not isinstance(metrics, list):  
                        print("Warning: metrics.json is not a list, resetting to an empty list.")
                        metrics = []  # Reset to an empty list

                    metrics.append({
                        "timestamp": timestamp,
                        "volume_id": volume_id,
                        "host_id": host_id,
                        "io_count": io_count,
                        "latency": latency,
                        "throughput": throughput
                    })
                    self.save_resource("metrics", metrics)

                    print(f"Metrics updated for volume {volume_id}")  # Debug log

                    time.sleep(100)  # Log every 5 seconds
            except Exception as e:
                import traceback
                print(f"ERROR in start_host_io(): {traceback.format_exc()}")  # Print full error traceback

        # Move thread creation OUTSIDE the loop
        worker_thread = threading.Thread(target=io_worker, daemon=True)
        worker_thread.start()

        print(f"Background thread started for volume {volume_id}")  # Debug log
    
    def unexport_volume(self, volume_id):
        volumes = self.load_resource("volume")
        volume = next((v for v in volumes if v["id"] == volume_id), None)
        if not volume:
            raise ValueError("Invalid volume ID")
        if not volume.get("is_exported", False):
            raise ValueError("Volume is not exported")
        volume["is_exported"] = False
        volume["exported_host_id"] = None
        volume["workload_size"] = None
        print(f"📢 Unexporting volume {volume_id}")
        self.update_resource("volume", volume_id, volume)
        print(f"✅ Volume {volume_id} unexported successfully")
        return f"Volume {volume_id} unexported successfully"







class Settings:
    def __init__(self, id, system_id):
        self.id = id
        self.system_id = system_id
        

    def to_dict(self):
        return {
            "id": self.id,
            "system_id": self.system_id,
            
        }
