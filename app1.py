import os
import uuid
import socket
import datetime
import flask
from flask import Flask, request, jsonify, send_from_directory, send_file
from models1 import System, Volume, Host, Settings
from storage1 import StorageManager
from logger import Logger
import json
import requests
import re

app = Flask(__name__)

print("Flask app is starting...")

# Configuration for multi-instance simulation
GLOBAL_FILE = "global_systems.json"  # Tracks all instances
ENABLE_UI = True  # Enable UI serving

# Automatically find the next available port (5000+)

def find_available_port(start=5000, max_instances=50):
    for port in range(start, start + max_instances):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No available ports found!")

PORT = find_available_port() 

# Unique data directory for this instance
DATA_DIR = f"data_instance_{PORT}"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize logger
logger = Logger(port=PORT, data_dir=DATA_DIR)

# Initialize storage manager for this instance
storage_mgr = StorageManager(DATA_DIR, GLOBAL_FILE, logger=logger)

# Helper to check if a system exists (guard rail)
def ensure_system_exists():
    systems = storage_mgr.load_resource("system")
    if not systems:
        return False, jsonify({"error": "No system exists. Create one first."}), 400
    return True, systems[0], 200

# --- System Routes ---
@app.route('/system', methods=['POST'])
def create_system():
    systems = storage_mgr.load_resource("system")
    if systems:
        logger.warn("Attempt to create system when one already exists", global_log=True)
        return jsonify({"error": "System already exists in this instance."}), 400

    data = request.get_json(silent=True) or {}
    try:
        system_id = str(uuid.uuid4())
        system_name = str(PORT)
        max_throughput = data.get("max_throughput", 200)  # Default 200 MBPS
        max_capacity = data.get("max_capacity", 1024)    # Default 1024 GB
        
        system = System(
            id=system_id,
            name=system_name,
            max_throughput=max_throughput,
            max_capacity=max_capacity
        )

        storage_mgr.save_resource("system", system.to_dict())
        storage_mgr.add_system_to_global(system_id, system.name, PORT)
        # Initialize system metrics
        storage_mgr.save_metrics({"throughput_used": 0, "capacity_used": 0})
        logger.info(f"System created with ID: {system_id}", global_log=True)

        return jsonify({"system_id": system.id, "port": PORT}), 201

    except Exception as e:
        logger.error(f"Failed to create system: {str(e)}", global_log=True)
        return jsonify({"error": f"Failed to create system: {str(e)}"}), 500

@app.route('/system', methods=['GET'])
def get_system():
    exists, system_data, status = ensure_system_exists()
    if not exists:
        return system_data, status
    return jsonify(system_data), 200

@app.route('/all-systems', methods=['GET'])
def get_all_systems():
    try:
        systems = storage_mgr.get_all_systems()
        return jsonify(systems), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve systems: {str(e)}"}), 500

@app.route('/system', methods=['PUT'])
def update_system():
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status

    data = request.get_json(silent=True) or {}

    try:
        # Return error if trying to update max_throughput or max_capacity after system creation
        if "max_throughput" in data or "max_capacity" in data:
            return jsonify({
                "error": "Cannot modify max_throughput or max_capacity after system creation"
            }), 400
        
        storage_mgr.update_resource("system", system["id"], system)
        return jsonify(system), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update system: {str(e)}"}), 500

@app.route('/system', methods=['DELETE'])
def delete_system():
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status  # System does not exist, return error

    try:
        system_id = system["id"]
        
        # Remove all related data
        storage_mgr.delete_related_resources("volume", system_id)
        storage_mgr.delete_related_resources("settings", system_id)
        storage_mgr.delete_related_resources("host", system_id)
        

        # Delete the system itself
        storage_mgr.delete_resource("system", None)  # Delete system locally
        storage_mgr.remove_system_from_global(system_id)  # Delete from global tracking

        return jsonify({"message": "System and all related data deleted successfully"}), 204

    except Exception as e:
        return jsonify({"error": f"Failed to delete system: {str(e)}"}), 500
    

@app.route('/settings/', methods=['GET'])
def get_all_settings():
    settings = storage_mgr.load_resource("settings")
    return jsonify(settings)



# --- Volume Routes ---
@app.route('/volume', methods=['POST'])
def create_volume():
    data = request.get_json(silent=True) or {}
    system_id = data.get("system_id")
    name = data.get("name")
    try:
        size = int(data.get("size"))
    except (ValueError, TypeError):
        return jsonify({"error": "Volume size must be a valid integer"}), 400

    # Validate required fields
    if not system_id or not name or size is None:
        return jsonify({"error": "System ID, volume name, and volume size are required"}), 400

    # (Optional) Retrieve the system record to compare against max_capacity
    systems = storage_mgr.load_resource("system")
    system = next((s for s in systems if s["id"] == system_id), None)
    if not system:
        return jsonify({"error": "System not found"}), 404

    # Example capacity check:
    try:
        max_capacity = int(system.get("max_capacity", 1024))
    except (ValueError, TypeError):
        max_capacity = 1024

    # (Optional) If you check that the size does not exceed max_capacity:
    if size > max_capacity:
        return jsonify({"error": f"Volume size exceeds system capacity of {max_capacity} GB"}), 400

    # Construct the Volume object (assuming you have a Volume model or similar)
    try:
        volume_id = str(uuid.uuid4())
        volume = {
            "id": volume_id,
            "name": name,
            "system_id": system_id,
            "size": size,
            "is_exported": False,
            "exported_host_id": None,
            "workload_size": 0,
            "snapshot_settings": {},
            "replication_settings": []
        }
        storage_mgr.save_resource("volume", volume)
        # Optionally update system metrics (if using update_capacity_used, ensure it handles int math)
        storage_mgr.update_capacity_used(size)
        return jsonify({"message": "Volume created successfully", "volume": volume}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to create volume: {str(e)}"}), 500

@app.route('/volume/<volume_id>', methods=['GET'])
def get_volume(volume_id):
    volumes = storage_mgr.load_resource("volume")
    volume = next((v for v in volumes if v["id"] == volume_id), None)
    if not volume:
        return jsonify({"error": "Volume not found."}), 404
    return jsonify(volume), 200

@app.route("/data/volume", methods=["GET"])
def get_all_volumes():
    volumes = storage_mgr.load_resource("volume")  # Load all volumes
    return jsonify(volumes), 200


def _convert_time(time_str):
    """Parses a string like '2 minutes' and converts it to seconds."""
    match = re.match(r"(\d+)\s*(seconds?|minutes?|hours?)", time_str.strip().lower())
    if not match:
        raise ValueError("Invalid time format. Use '30 seconds', '1 minute', or '2 hours'.")

    value, unit = int(match.group(1)), match.group(2)

    if "second" in unit:
        return value  # No conversion needed
    elif "minute" in unit:
        return value * 60  # Convert minutes to seconds
    elif "hour" in unit:
        return value * 3600  # Convert hours to seconds
    else:
        raise ValueError("Invalid unit. Use 'seconds', 'minutes', or 'hours'.")



@app.route('/volume/<volume_id>', methods=['PUT'])
def update_volume(volume_id):
    try:
        print(f"🔄 Received request to update volume {volume_id}")  # Debug log

        # ✅ Load volume and ensure it exists
        volumes = storage_mgr.load_resource("volume")
        volume = next((v for v in volumes if v["id"] == volume_id), None)
        if not volume:
            print(f"❌ ERROR: Volume {volume_id} not found.")
            return jsonify({"error": "Volume not found."}), 404

        # ✅ Unexport if volume is currently exported
        if volume.get("is_exported"):
            print(f"🚨 Unexporting volume {volume_id} before updating settings.")
            storage_mgr.unexport_volume(volume_id, reason="Volume update")

        # ✅ Get incoming data
        data = request.get_json(silent=True) or {}
        print(f"📥 Incoming data: {data}")  # Debug log
        setting_ids = data.get("setting_ids", [])  # List of setting IDs to apply

        # ✅ Load settings to validate setting IDs
        settings = storage_mgr.load_resource("settings")
        valid_setting_ids = {s["id"] for s in settings}
        invalid_ids = [sid for sid in setting_ids if sid not in valid_setting_ids]
        if invalid_ids:
            return jsonify({"error": f"Invalid setting IDs: {invalid_ids}"}), 400

        try:
            # ✅ Ensure settings containers exist
            volume.setdefault("snapshot_settings", {})
            volume.setdefault("replication_settings", [])

            # ✅ Remove settings that are no longer applied
            current_settings = set(volume["snapshot_settings"].keys()) | {
                s.get("setting_id") for s in volume["replication_settings"]
            }
            for old_id in current_settings - set(setting_ids):
                volume["snapshot_settings"].pop(old_id, None)
                volume["replication_settings"] = [
                    r for r in volume["replication_settings"] if r.get("setting_id") != old_id
                ]

            # ✅ Apply new settings and ensure values are in seconds
            snapshot_frequencies = []
            for setting_id in setting_ids:
                setting = next(s for s in settings if s["id"] == setting_id)

                if setting["type"] == "snapshot":
                    # Check if value is already in seconds, otherwise convert
                    if isinstance(setting["value"], int):  
                        converted_value = setting["value"]  # Already in seconds
                    else:
                        converted_value = _convert_time(setting["value"])  # Convert only if needed

                    if setting_id not in volume["snapshot_settings"]:
                        volume["snapshot_settings"][setting_id] = converted_value

                    snapshot_frequencies.append(converted_value)

                elif setting["type"] == "replication":
                    if not any(r.get("setting_id") == setting_id for r in volume["replication_settings"]):
                        target = setting.get("replication_target", {})
                        if not target or not target.get("id"):
                            return jsonify({"error": f"Setting {setting_id} has invalid replication target"}), 400

                        volume["replication_settings"].append({
                            "setting_id": setting_id,
                            "replication_type": setting["replication_type"],
                            "delay_sec": setting["delay_sec"],
                            "replication_target": setting["replication_target"]
                        })

            # ✅ Save updated volume
            volume["snapshot_frequencies"] = snapshot_frequencies  # ✅ Store converted values
            storage_mgr.update_resource("volume", volume_id, volume)

            # ✅ Restart snapshot with converted frequencies
            print(f"🚀 Restarting snapshot for volume {volume_id} with frequencies {snapshot_frequencies}")
            storage_mgr.start_snapshot(volume_id, snapshot_frequencies)

            return jsonify({"message": "Settings updated successfully", "volume": volume}), 200

        except Exception as e:
            print(f"❌ ERROR updating volume settings: {str(e)}")
            return jsonify({"error": f"Failed to update volume settings: {str(e)}"}), 500

    except Exception as e:
        print(f"❌ ERROR in update_volume(): {str(e)}")
        return jsonify({"error": f"Failed to update volume: {str(e)}"}), 500


@app.route('/volume/<volume_id>', methods=['DELETE'])
def delete_volume(volume_id):
    try:
        volume = next((v for v in storage_mgr.load_resource("volume") 
                      if v["id"] == volume_id), None)
        if volume and volume.get("is_exported"):
            # Cleanup processes before deletion
            storage_mgr.unexport_volume(volume_id, reason="Volume deletion")
        
        storage_mgr.delete_resource("volume", volume_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Host Routes ---
@app.route('/host', methods=['POST'])
def create_host():
    data = request.get_json(silent=True) or {}

    # Validate system ID
    system_id = data.get("system_id")
    if not system_id:
        return jsonify({"error": "❌ System ID is required to create a host."}), 400

    # Ensure system exists
    systems = storage_mgr.load_resource("system")
    if not any(s["id"] == system_id for s in systems):
        return jsonify({"error": "❌ Invalid system ID."}), 400

    # Load existing hosts
    hosts = storage_mgr.load_resource("host")
    host_name = data.get("name", "DefaultHost")

    # Check if a host with the same name already exists for this system_id
    if any(h["name"] == host_name and h["system_id"] == system_id for h in hosts):
        return jsonify({
            "error": f"❌ Host '{host_name}' already exists for system {system_id}."
            }), 400

    try:
        host = Host(
            id=str(uuid.uuid4()),
            system_id=system_id,
            name=host_name,
            application_type=data.get("application_type", "Unknown"),
            protocol=data.get("protocol", "Unknown")
        )
        storage_mgr.save_resource("host", host.to_dict())
        return jsonify({"host_id": host.id}), 201
    except Exception as e:
        return jsonify({"error": f"❌ Failed to create host: {str(e)}"}), 500


@app.route('/host', methods=['GET'])
def get_all_hosts():
    try:
        hosts = storage_mgr.load_resource("host")
        if not isinstance(hosts, list):
            hosts = []  # Ensure valid JSON format
        return jsonify(hosts), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch hosts: {str(e)}"}), 500

@app.route('/host/<host_id>', methods=['GET'])
def get_host(host_id):
    hosts = storage_mgr.load_resource("host")
    host = next((h for h in hosts if h["id"] == host_id), None)
    if not host:
        return jsonify({"error": "Host not found."}), 404
    return jsonify(host), 200

@app.route('/host/<host_id>', methods=['PUT'])
def update_host(host_id):
    hosts = storage_mgr.load_resource("host")
    host = next((h for h in hosts if h["id"] == host_id), None)
    
    if not host:
        return jsonify({"error": "❌ Host not found."}), 404

    data = request.get_json(silent=True) or {}

    try:
        # Update fields if provided
        host["name"] = data.get("name", host["name"])
        host["application_type"] = data.get("application_type", host["application_type"])
        host["protocol"] = data.get("protocol", host["protocol"])

        # Save the updated host
        storage_mgr.update_resource("host", host_id, host)

        return jsonify(host), 200
    except Exception as e:
        return jsonify({"error": f"❌ Failed to update host: {str(e)}"}), 500

@app.route('/host/<host_id>', methods=['DELETE'])
def delete_host(host_id):
    try:
        # Check if host has any exported volumes
        volumes = storage_mgr.load_resource("volume")
        exported_volumes = [v for v in volumes if v.get("exported_host_id") == host_id]
        
        # Unexport all volumes connected to this host
        for volume in exported_volumes:
            storage_mgr.unexport_volume(volume["id"], reason=f"Host {host_id} deleted")

        # Then delete the host
        storage_mgr.delete_resource("host", host_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Settings Routes ---
@app.route('/settings', methods=['POST'])
def create_settings():
    """Creates a new setting, ensuring snapshot max value is saved properly."""
    
    # ✅ Ensure system exists before proceeding
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status
    
    data = request.get_json(silent=True) or {}

    if data.get("system_id") != system["id"]:
        return jsonify({"error": "Invalid system_id."}), 400

    setting_name = data.get("name")
    setting_type = data.get("type")
    system_id = data.get("system_id")

    if not all([setting_name, setting_type, system_id]):
        return jsonify({"error": "Name, type, and system_id are required"}), 400

    try:
        setting_id = str(uuid.uuid4())
        setting_data = {
            "id": setting_id,
            "system_id": system_id,
            "name": setting_name,
            "type": setting_type
        }

        if setting_type == "snapshot":
            setting_value = data.get("value")
            max_snapshots = data.get("max_snapshots")

            if not setting_value:
                return jsonify({"error": "Value is required for snapshot settings"}), 400

            # ✅ Ensure max_snapshots is stored as an integer (even if None)
            if max_snapshots is not None:
                try:
                    max_snapshots = int(max_snapshots)
                    if max_snapshots <= 0:
                        raise ValueError
                except ValueError:
                    return jsonify({"error": "max_snapshots must be a positive integer"}), 400
            else:
                max_snapshots = 0  # ✅ Default to 0 if not provided

            setting_data["value"] = setting_value
            setting_data["max_snapshots"] = max_snapshots  # ✅ Ensures it is saved
        elif setting_type == "replication":
            # ✅ Replication-specific logic
            replication_type = data.get("replication_type")
            delay_sec = int(data.get("delay_sec", 0))
            target_system_id = data.get("replication_target_id")
            target_system_name = data.get("replication_target_name")

            if not replication_type or replication_type not in ["synchronous", "asynchronous"]:
                return jsonify({"error": "Invalid replication type"}), 400

            if replication_type == "synchronous" and delay_sec != 0:
                return jsonify({"error": "Synchronous replication must have delay_sec = 0"}), 400

            if replication_type == "asynchronous" and delay_sec <= 0:
                return jsonify({"error": "Asynchronous replication must have delay_sec > 0"}), 400

            if not target_system_id or target_system_id == system_id:
                return jsonify({"error": "Invalid replication target"}), 400

            setting_data.update({
                "replication_type": replication_type,
                "delay_sec": delay_sec,
                "replication_target": {
                    "id": target_system_id,
                    "name": target_system_name
                }
            })
        else:
            return jsonify({"error": "Invalid setting type"}), 400

        # ✅ Debugging: Print setting_data before saving
        print("Saving setting:", setting_data)

        # ✅ Save settings (make sure this function writes to settings.json)
        storage_mgr.save_resource("settings", setting_data)

        # ✅ Verify if settings.json actually updates
        saved_settings = storage_mgr.load_resource("settings")  # Debugging step
        print("Current settings.json:", saved_settings)

        return jsonify({"message": "Setting created successfully!", "setting_id": setting_id}), 201

    except ValueError as e:
        return jsonify({"error": f"Invalid delay_sec value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to create setting: {str(e)}"}), 500


@app.route('/settings/<settings_id>', methods=['GET'])
def get_settings(settings_id):
    settings_list = storage_mgr.load_resource("settings")
    settings = next((s for s in settings_list if s["id"] == settings_id), None)
    if not settings:
        return jsonify({"error": "Settings not found."}), 404
    return jsonify(settings), 200

@app.route('/settings/<settings_id>', methods=['PUT'])
def update_settings(settings_id):
    try:
        # Get volumes using this setting
        volumes = storage_mgr.load_resource("volume")
        affected_volumes = [v for v in volumes 
                          if any(r.get("setting_id") == settings_id 
                                for r in v.get("replication_settings", []))]
        
        # For exported volumes, stop their replication
        for volume in affected_volumes:
            if volume.get("is_exported"):
                storage_mgr.cleanup_volume_processes(volume["id"], 
                    reason=f"Settings {settings_id} update", 
                    notify_targets=True)
        
        # Update the setting
        data = request.get_json(silent=True) or {}
        setting_name = data.get("name")
        setting_type = data.get("type")
        system_id = data.get("system_id")

        # Only require value for non-replication settings
        if setting_type != "replication":
            setting_value = data.get("value")
            if not setting_value:
                return jsonify({"error": "Value is required for non-replication settings"}), 400
        else:
            setting_value = None

        if not all([setting_name, setting_type, system_id]):
            return jsonify({"error": "Name, type, and system_id are required"}), 400

        try:
            setting_id = str(uuid.uuid4())
            setting_data = {
                "id": setting_id,
                "system_id": system_id,
                "name": setting_name,
                "type": setting_type,
            }

            if setting_type == "replication":
                replication_type = data.get("replication_type")
                delay_sec = int(data.get("delay_sec", 0))
                target_system_id = data.get("replication_target_id")
                target_system_name = data.get("replication_target_name")

                # Validate replication settings
                if not replication_type or replication_type not in ["synchronous", "asynchronous"]:
                    return jsonify({"error": "Invalid replication type"}), 400

                if replication_type == "synchronous" and delay_sec != 0:
                    return jsonify({"error": "Synchronous replication must have delay_sec = 0"}), 400

                if replication_type == "asynchronous" and delay_sec <= 0:
                    return jsonify({"error": "Asynchronous replication must have delay_sec > 0"}), 400

                if not target_system_id or target_system_id == system_id:
                    return jsonify({"error": "Invalid replication target"}), 400

                setting_data.update({
                    "replication_type": replication_type,
                    "delay_sec": delay_sec,
                    "replication_target": {
                        "id": target_system_id,
                        "name": target_system_name
                    }
                })
            else:
                setting_data["value"] = setting_value

            storage_mgr.save_resource("settings", setting_data)

            # Restart processes for exported volumes
            for volume in affected_volumes:
                if volume.get("is_exported"):
                    storage_mgr.start_replication(volume["id"])
                
            return jsonify({"message": "Settings updated successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/settings/<settings_id>', methods=['DELETE'])
def delete_settings(settings_id):
    settings_list = storage_mgr.load_resource("settings")

    print(f"Received settings_id for deletion: {settings_id}")
    print(f"Existing settings before deletion: {settings_list}")

    if not any(str(s["id"]) == str(settings_id) for s in settings_list):
        return jsonify({"error": "Settings not found."}), 404

    try:
        # ✅ Manually delete the setting and update storage
        settings_list = [s for s in settings_list if str(s["id"]) != str(settings_id)]
        storage_mgr.save_resource("settings", settings_list)

        print(f"Settings after deletion: {storage_mgr.load_resource('settings')}")

        return jsonify({"message": "Settings deleted successfully"}), 204
    except Exception as e:
        return jsonify({"error": f"Failed to delete settings: {str(e)}"}), 500

# --- New Endpoint for Raw JSON Files ---
@app.route('/data/<resource_type>', methods=['GET'])
def get_raw_json(resource_type):
    valid_resources = ["system", "volume", "host", "settings"]
    if resource_type not in valid_resources:
        return jsonify({"error": "Invalid resource type."}), 400
    file_path = os.path.join(DATA_DIR, f"{resource_type}.json")
    if not os.path.exists(file_path):
        return jsonify([]), 200  # Return empty array if file doesn't exist
    return send_file(file_path, mimetype='application/json')

# --- Plug-and-Play UI ---
print(f"ENABLE_UI is set to {ENABLE_UI}")
if ENABLE_UI:
    @app.route('/ui')
    def serve_ui():
        print(f"ENABLE_UI is set to {ENABLE_UI}")
        return send_from_directory('ui', 'index.html')
    
@app.route("/export-volume", methods=["POST"])
def export_volume():
    data = request.json
    print(data)
    
    volume_id = data.get("volume_id")
    host_id = data.get("host_id")
    workload_size = int(data.get("workload_size"))

    print(f"📢 Received request - Volume: {volume_id}, Host: {host_id}, Workload: {workload_size}")  # Debugging

    if not volume_id or not host_id or not workload_size:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        result = storage_mgr.export_volume(volume_id, host_id, workload_size)
        # Update system saturation after export
        storage_mgr.cleanup()
        return jsonify({"message": result}), 200
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {traceback.format_exc()}")  # Print full error traceback
        return jsonify({"error": str(e)}), 500

data_dir = "data_instance_5000"
volume_file = os.path.join(DATA_DIR, "volume.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load volumes
if not os.path.exists(volume_file):
    with open(volume_file, "w") as f:
        json.dump([], f, indent=4)

def load_volumes():
    with open(volume_file, "r") as f:
        return json.load(f)

def save_volumes(volumes):
    with open(volume_file, "w") as f:
        json.dump(volumes, f, indent=4)


@app.route("/unexport-volume", methods=["POST"])
def unexport_volume():
    try:
        data = request.get_json()
        volume_id = data.get("volume_id")

        print(f"📌 Unexporting Volume ID: {volume_id}")

        volumes = load_volumes()  # Load from volume.json
        volume_found = False

        for volume in volumes:
            if volume["id"] == volume_id:
                print(f"✅ Found Volume: {volume}")
                volume["is_exported"] = False  # Update is_exported
                volume_found = True
                break  # Stop searching

        if not volume_found:
            print("❌ Volume ID not found!")
            return jsonify({"error": "Volume not found"}), 404
        
        volume_file_path = os.path.join(DATA_DIR, "volume.json")
        # 🔥 Save changes back to volume.json
        with open(volume_file_path, "w") as f:
            json.dump(volumes, f, indent=4)
            print("💾 Updated volume.json successfully!")
        # Update system saturation after unexport
        storage_mgr.cleanup()
        return jsonify({"message": "Volume unexported successfully!"}), 200

    except Exception as e:
        print(f"❌ Error in unexport_volume: {e}")
        return jsonify({"error": "Failed to unexport volume"}), 500

@app.route("/data/exported-volumes", methods=["GET"])
def get_exported_volumes():
    try:
        volumes = load_volumes()
        exported_volumes = [v for v in volumes if v.get("is_exported", False)]

        print("📤 Exported Volumes:", exported_volumes)  # Debugging log

        return jsonify(exported_volumes), 200  # ✅ Return only the list, no extra nesting
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Failed to load exported volumes"}), 400


@app.route('/data/all-settings', methods=['GET'])
def fetch_all_settings():
    try:
        settings = storage_mgr.load_resource("settings")
        print(f"📢 All Settings Loaded: {settings}")  # Debugging log
        return jsonify(settings), 200
    except Exception as e:
        print(f"❌ Error fetching settings: {str(e)}")
        return jsonify({"error": f"Failed to retrieve settings: {str(e)}"}), 500

@app.route('/data/global-systems', methods=['GET'])
def get_global_systems():
    try:
        with open("global_systems.json", "r") as f:
            systems = json.load(f)
            if isinstance(systems, list):  # Make sure it's a list
                return jsonify(systems), 200
            else:
                return jsonify({"error": "Data format in global_systems.json is invalid"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load global systems: {str(e)}"}), 500

from app1 import load_volumes  # Import your function

volumes = load_volumes()
print("Test Loaded Volumes:", volumes)

# Add new routes for logs
@app.route('/logs/local', methods=['GET'])
def get_local_logs():
    logs = logger.get_local_logs()
    return jsonify(logs), 200

@app.route('/logs/global', methods=['GET'])
def get_global_logs():
    logs = logger.get_global_logs()
    return jsonify(logs), 200

@app.route('/system/metrics', methods=['GET'])
def get_system_metrics():
    try:
        metrics = storage_mgr.load_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500

# Add new API endpoint for replication reception (target system)
@app.route('/replication-receive', methods=['POST'])
def replication_receive():
    data = request.get_json(silent=True) or {}
    volume_id = data.get("volume_id")
    replication_throughput = data.get("replication_throughput")
    sender = data.get("sender")
    timestamp = data.get("timestamp")
    replication_type = data.get("replication_type", "unknown")
    should_log = data.get("should_log", True)
    latency = data.get("latency", 0)
    
    # Get source volume details
    source_volume = data.get("source_volume")
    if source_volume:
        # Create target volume if it doesn't exist
        volumes = storage_mgr.load_resource("volume")
        target_volume_name = f"{source_volume['name']}_{replication_type}{source_volume['system_name']}"
        
        # Check if target volume already exists
        target_volume = next((v for v in volumes if v["name"] == target_volume_name), None)
        
        if not target_volume:
            # Get local system info
            systems = storage_mgr.load_resource("system")
            local_system = systems[0] if systems else None
            
            if local_system:
                # Create new volume with target system specifics
                new_volume = {
                    "id": str(uuid.uuid4()),
                    "name": target_volume_name,
                    "system_id": local_system["id"],
                    "size": int(source_volume["size"]),  # Ensure size is integer
                    "is_exported": False,
                    "exported_host_id": None,
                    "workload_size": 0,
                    "snapshot_settings": {},
                    "replication_settings": []
                }
                
                # Update system metrics for the target system
                try:
                    # Update capacity used - ensure all values are integers
                    current_metrics = storage_mgr.load_metrics()
                    current_capacity = int(current_metrics["capacity_used"])
                    new_volume_size = int(source_volume["size"])
                    new_capacity = current_capacity + new_volume_size
                    max_capacity = int(local_system.get("max_capacity", 1024))
                    
                    # Check if we exceed max capacity
                    if new_capacity > max_capacity:
                        logger.error(f"Cannot create replicated volume: would exceed system capacity ({new_capacity} > {max_capacity})", global_log=True)
                        return jsonify({"error": "Target system capacity would be exceeded"}), 400
                    
                    # Update metrics
                    storage_mgr.save_metrics({
                        "throughput_used": int(current_metrics["throughput_used"]),
                        "capacity_used": new_capacity
                    })
                    
                    # Save the new volume
                    storage_mgr.save_resource("volume", new_volume)
                    logger.info(f"Created target volume {target_volume_name} for replication and updated system metrics", global_log=True)
                except Exception as e:
                    logger.error(f"Failed to update system metrics: {str(e)}", global_log=True)
                    return jsonify({"error": f"Failed to update system metrics: {str(e)}"}), 500

    if should_log:
        # Log receiver replication event in receiver's log format
        if replication_type == "synchronous":
            receiver_log = (f"Active synchronous replication received for volume {volume_id} "
                        f"from {sender} - Throughput: {replication_throughput} MB/s, "
                        f"Latency: {latency}ms")
        else:
            receiver_log = (f"Received {replication_type} replication for volume {volume_id} "
                        f"with throughput {replication_throughput} MB/s from sender {sender}")
        logger.info(receiver_log, global_log=True)

    # Update target's replication metrics
    storage_mgr.update_replication_metrics(
        volume_id, 
        "received_from_" + sender,
        {
            "throughput": replication_throughput,
            "latency": latency,
            "timestamp": timestamp,
            "replication_type": replication_type
        }
    )

    return jsonify({"message": "Replication data received"}), 200

@app.route('/replication-stop', methods=['POST'])
def replication_stop():
    data = request.get_json(silent=True) or {}
    volume_id = data.get("volume_id")
    reason = data.get("reason", "Unknown reason")
    sender = data.get("sender")

    # Log the replication stop event
    log_msg = f"Replication stopped for volume {volume_id} from {sender}: {reason}"
    logger.info(log_msg, global_log=True)
    return jsonify({"message": "Replication stop acknowledged"}), 200

#LOG_FILE = os.path.join(storage_mgr.data_dir, "data_instance_5000/logs_5000.txt")
LOG_FILE = os.path.join(data_dir, "logs_5000.txt")
VOLUME_FILE= os.path.join(data_dir, "volume.json")
#VOLUME_FILE = os.path.join(storage_mgr.data_dir, "data_instance_5000/volume.json")



@app.route('/api/latency', methods=['GET'])
def get_latency():
    print("Checking if log file exists:", os.path.exists(LOG_FILE))
    print("Checking if volume file exists:", os.path.exists(VOLUME_FILE))
    try:
        if not os.path.exists(LOG_FILE) or not os.path.exists(VOLUME_FILE):
            return jsonify({"error": "Log file or volume file not found"}), 404
            
        # Load exported volumes
        with open(VOLUME_FILE, "r") as f:
            volumes = json.load(f)
        exported_volumes = {v["id"] for v in volumes if v.get("is_exported", False)}
        
        now = datetime.datetime.utcnow()
        fifteen_minutes_ago = now - datetime.timedelta(minutes=15)
        volume_latency_data = {}
        
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        
        log_pattern = re.compile(r'\[(.*?)\]\[INFO\] Volume: (.*?), Host: (.*?), IOPS: (\d+), Latency: ([\d\.]+)ms, Throughput: ([\d\.]+) MB/s')
        
        for line in lines:
            match = log_pattern.search(line)
            if match:
                timestamp_str, volume_id, host_id, iops, latency, throughput = match.groups()
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                if timestamp >= fifteen_minutes_ago and volume_id in exported_volumes:
                    if volume_id not in volume_latency_data:
                        volume_latency_data[volume_id] = {"timestamps": [], "values": []}
                    volume_latency_data[volume_id]["timestamps"].append(timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"))
                    volume_latency_data[volume_id]["values"].append(float(latency))
        
        return jsonify(volume_latency_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

LOG_FILE = os.path.join(data_dir, "logs_5000.txt")

@app.route('/api/top-latency', methods=['GET'])
def get_top_latency():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"error": "Log file not found"}), 404
        
        now = datetime.datetime.utcnow()
        fifteen_minutes_ago = now - datetime.timedelta(minutes=15)
        volume_latency = {}

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        log_pattern = re.compile(r'\[(.*?)\]\[INFO\] Volume: (.*?), .*? Latency: ([\d\.]+)ms')

        for line in lines:
            match = log_pattern.search(line)
            if match:
                timestamp_str, volume_id, latency = match.groups()
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if timestamp >= fifteen_minutes_ago:
                    if volume_id not in volume_latency:
                        volume_latency[volume_id] = []
                    volume_latency[volume_id].append(float(latency))

        # Compute average latency per volume
        avg_latency = {vol: sum(lats) / len(lats) for vol, lats in volume_latency.items() if lats}

        # Get top 3 volumes by highest average latency
        top_volumes = sorted(avg_latency.items(), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({"top_volumes": [{"volume_id": v, "avg_latency": round(l, 2)} for v, l in top_volumes]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def run_cleanup():
    """
    Trigger cleanup manually via API
    """
    try:
        storage_mgr.cleanup()
        return jsonify({"message": "Housekeeping executed successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to execute housekeeping: {str(e)}"}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True,use_reloader=False)
    print("Registered routes:", app.url_map)
