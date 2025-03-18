import os
import uuid
import socket
import flask
from flask import Flask, request, jsonify, send_from_directory, send_file
from models1 import System, Node, Volume, Host, Settings
from storage1 import StorageManager
import json

app = Flask(__name__)

# Configuration for multi-instance simulation
GLOBAL_FILE = "global_systems.json"  # Tracks all instances
ENABLE_UI = True  # Enable UI serving

# Automatically find the next available port (5000+)
'''
def find_available_port(start=5000, max_instances=50):
    for port in range(start, start + max_instances):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No available ports found!")'
'''

def find_available_port():
    return int(os.getenv("FLASK_PORT", 5000))  # Default to 5000, override with FLASK_PORT

PORT = find_available_port() 

# Unique data directory for this instance
DATA_DIR = f"data_instance_{PORT}"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize storage manager for this instance
storage_mgr = StorageManager(DATA_DIR, GLOBAL_FILE)
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
        return jsonify({"error": "System already exists in this instance."}), 400

    data = request.get_json(silent=True) or {}
    try:
        system_id = str(uuid.uuid4())
        system = System(
            id=system_id, 
            name=data.get("name", f"Storage System {PORT}"),
            remote_replication=False,  # Default to false
            replication_target=None  # No target initially
        )

        # Save locally in this instance
        storage_mgr.save_resource("system", system.to_dict())

        # Track globally
        storage_mgr.add_system_to_global(system_id, system.name, PORT)

        return jsonify({"system_id": system.id, "port": PORT}), 201

    except Exception as e:
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
        system["name"] = data.get("name", system["name"])
        remote_replication = data.get("remote_replication", system.get("remote_replication", False))
        replication_target = data.get("replication_target")
        replication_type = data.get("replication_type")

        # Ensure valid replication settings
        if remote_replication:
            if not replication_target:
                return jsonify({"error": "A target system must be selected for replication."}), 400

            if replication_target == system["id"]:
                return jsonify({"error": "Target system cannot be the same as the source system."}), 400

            # Load available systems
            all_systems = storage_mgr.get_all_systems()
            target_system = next((s for s in all_systems if s["id"] == replication_target), None)

            if not target_system:
                return jsonify({"error": "Invalid target system selected for replication."}), 400

            system["remote_replication"] = True
            system["replication_target"] = replication_target
            system["replication_type"] = replication_type

            # If asynchronous, set frequency to 4 minutes, otherwise remove frequency
            if replication_type == "asynchronous":
                system["replication_frequency"] = "4 minutes"
            else:
                system.pop("replication_frequency", None)
        else:
            system["remote_replication"] = False
            system["replication_target"] = None
            system["replication_type"] = None
            system.pop("replication_frequency", None)

        # Save updates to system.json
        storage_mgr.update_resource("system", system["id"], system)

        # Save replication settings to settings.json
        storage_mgr.update_replication_in_settings(system["id"], system["replication_type"], system["replication_target"], system.get("replication_frequency"))

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
        
        storage_mgr.delete_related_resources("node", system_id)
        storage_mgr.delete_related_resources("volume", system_id)
        storage_mgr.delete_related_resources("settings", system_id)
        storage_mgr.delete_related_resources("host", system_id)
        

        # Delete the system itself
        storage_mgr.delete_resource("system", None)  # Delete system locally
        storage_mgr.remove_system_from_global(system_id)  # Delete from global tracking

        return jsonify({"message": "System and all related data deleted successfully"}), 204

    except Exception as e:
        return jsonify({"error": f"Failed to delete system: {str(e)}"}), 500


# --- Node Routes ---
@app.route('/node', methods=['POST'])
def create_node():
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status
    
    data = request.get_json(silent=True) or {}

    # Validate system_id
    if data.get("system_id") != system["id"]:
        return jsonify({"error": "Invalid system_id."}), 400

    # Load all nodes and check if the system already has 4 nodes
    nodes = storage_mgr.load_resource("node")
    system_nodes = [n for n in nodes if n["system_id"] == system["id"]]

    if len(system_nodes) >= 4:
        return jsonify({"error": "Maximum limit of 4 nodes per system reached."}), 400

    try:
        node = Node(id=str(uuid.uuid4()), name=data.get("name", "DefaultNode"), system_id=system["id"])
        storage_mgr.save_resource("node", node.to_dict())
        return jsonify({"node_id": node.id}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to create node: {str(e)}"}), 500


@app.route('/node/<node_id>', methods=['GET'])
def get_node(node_id):
    nodes = storage_mgr.load_resource("node")
    node = next((n for n in nodes if n["id"] == node_id), None)
    if not node:
        return jsonify({"error": "Node not found."}), 404
    return jsonify(node), 200

@app.route('/node/<node_id>', methods=['PUT'])
def update_node(node_id):
    nodes = storage_mgr.load_resource("node")
    node = next((n for n in nodes if n["id"] == node_id), None)
    if not node:
        return jsonify({"error": "Node not found."}), 404
    
    data = request.get_json(silent=True) or {}
    try:
        node["name"] = data.get("name", node["name"])
        storage_mgr.update_resource("node", node_id, node)
        return jsonify(node), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update node: {str(e)}"}), 500

@app.route('/node/<node_id>', methods=['DELETE'])
def delete_node(node_id):
    nodes = storage_mgr.load_resource("node")
    if not any(n["id"] == node_id for n in nodes):
        return jsonify({"error": "Node not found."}), 404
    
    try:
        storage_mgr.delete_resource("node", node_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": f"Failed to delete node: {str(e)}"}), 500

# --- Volume Routes ---
@app.route('/volume', methods=['POST'])
def create_volume():
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status
    
    data = request.get_json(silent=True) or {}
    try:
        if data.get("system_id") != system["id"]:
            return jsonify({"error": "Invalid system_id."}), 400
        volume = Volume(id=str(uuid.uuid4()), name=data.get("name", "DefaultVolume"), system_id=system["id"])
        storage_mgr.save_resource("volume", volume.to_dict())
        return jsonify({"volume_id": volume.id}), 201
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


@app.route('/volume/<volume_id>', methods=['PUT'])
def update_volume(volume_id):
    volumes = storage_mgr.load_resource("volume")
    volume = next((v for v in volumes if v["id"] == volume_id), None)
    if not volume:
        return jsonify({"error": "Volume not found."}), 404

    data = request.get_json(silent=True) or {}

    try:
        # Update volume attributes
        volume["name"] = data.get("name", volume["name"])
        volume["snapshot_frequency"] = data.get("snapshot_frequency", volume.get("snapshot_frequency", "daily"))

        # Save updated volume
        storage_mgr.update_resource("volume", volume_id, volume)

        # Debug print to check function call
        print(f"📌 Updating snapshot settings for volume {volume_id} in system {volume['system_id']}")

        # Also update snapshot frequency in settings.json
        storage_mgr.update_snapshot_in_settings(volume["system_id"], volume_id, volume["snapshot_frequency"])

        return jsonify(volume), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update volume: {str(e)}"}), 500

@app.route('/volume/<volume_id>', methods=['DELETE'])
def delete_volume(volume_id):
    volumes = storage_mgr.load_resource("volume")
    if not any(v["id"] == volume_id for v in volumes):
        return jsonify({"error": "Volume not found."}), 404
    
    try:
        storage_mgr.delete_resource("volume", volume_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": f"Failed to delete volume: {str(e)}"}), 500

# --- Host Routes ---
@app.route('/host', methods=['POST'])
# def create_host():
#     data = request.get_json(silent=True) or {}

#     # Validate system ID
#     system_id = data.get("system_id")
#     if not system_id:
#         return jsonify({"error": "❌ System ID is required to create a host."}), 400

#     # Ensure system exists
#     systems = storage_mgr.load_resource("system")
#     if not any(s["id"] == system_id for s in systems):
#         return jsonify({"error": "❌ Invalid system ID."}), 400

#     try:
#         host = Host(
#             id=str(uuid.uuid4()),
#             system_id=system_id,
#             name=data.get("name", "DefaultHost"),
#             application_type=data.get("application_type", "Unknown"),
#             protocol=data.get("protocol", "Unknown")
#         )
#         storage_mgr.save_resource("host", host.to_dict())
#         return jsonify({"host_id": host.id}), 201
#     except Exception as e:
#         return jsonify({"error": f"❌ Failed to create host: {str(e)}"}), 500
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
    hosts = storage_mgr.load_resource("host")
    if not any(h["id"] == host_id for h in hosts):
        return jsonify({"error": "Host not found."}), 404
    
    try:
        storage_mgr.delete_resource("host", host_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": f"Failed to delete host: {str(e)}"}), 500

# --- Settings Routes ---
@app.route('/settings', methods=['POST'])
def create_settings():
    exists, system, status = ensure_system_exists()
    if not exists:
        return system, status
    
    data = request.get_json(silent=True) or {}
    try:
        if data.get("system_id") != system["id"]:
            return jsonify({"error": "Invalid system_id."}), 400
        
        settings = Settings(
            id=str(uuid.uuid4()),
            system_id=system["id"],
            snapshot_frequency=data.get("snapshot_frequency", "daily")  # Default to 'daily'
        )
        storage_mgr.save_resource("settings", settings.to_dict())
        return jsonify({"settings_id": settings.id}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to create settings: {str(e)}"}), 500


@app.route('/settings/<settings_id>', methods=['GET'])
def get_settings(settings_id):
    settings_list = storage_mgr.load_resource("settings")
    settings = next((s for s in settings_list if s["id"] == settings_id), None)
    if not settings:
        return jsonify({"error": "Settings not found."}), 404
    return jsonify(settings), 200

@app.route('/settings/<settings_id>', methods=['PUT'])
def update_settings(settings_id):
    settings_list = storage_mgr.load_resource("settings")
    settings = next((s for s in settings_list if s["id"] == settings_id), None)
    if not settings:
        return jsonify({"error": "Settings not found."}), 404

    data = request.get_json(silent=True) or {}

    try:
        # Update replication type
        replication_type = data.get("replication_type", settings.get("replication_type", "synchronous"))
        settings["replication_type"] = replication_type

        # If asynchronous, set frequency to 4 minutes
        if replication_type == "asynchronous":
            settings["replication_frequency"] = "4 minutes"
        else:
            settings.pop("replication_frequency", None)  # Remove if not async

        # Update settings.json
        storage_mgr.update_resource("settings", settings_id, settings)

        # Update system.json to reflect changes
        system_id = settings["system_id"]
        system_list = storage_mgr.load_resource("system")
        system = next((s for s in system_list if s["id"] == system_id), None)
        if system:
            system["replication_type"] = replication_type
            if replication_type == "asynchronous":
                system["replication_frequency"] = "4 minutes"
            else:
                system.pop("replication_frequency", None)

            storage_mgr.update_resource("system", system_id, system)

        return jsonify(settings), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500


@app.route('/settings/<settings_id>', methods=['DELETE'])
def delete_settings(settings_id):
    settings_list = storage_mgr.load_resource("settings")
    if not any(s["id"] == settings_id for s in settings_list):
        return jsonify({"error": "Settings not found."}), 404
    
    try:
        storage_mgr.delete_resource("settings", settings_id)
        return "", 204
    except Exception as e:
        return jsonify({"error": f"Failed to delete settings: {str(e)}"}), 500

# --- New Endpoint for Raw JSON Files ---
@app.route('/data/<resource_type>', methods=['GET'])
def get_raw_json(resource_type):
    valid_resources = ["system", "node", "volume", "host", "settings"]
    if resource_type not in valid_resources:
        return jsonify({"error": "Invalid resource type."}), 400
    file_path = os.path.join(DATA_DIR, f"{resource_type}.json")
    if not os.path.exists(file_path):
        return jsonify([]), 200  # Return empty array if file doesn’t exist
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
        result =storage_mgr.export_volume(volume_id, host_id, workload_size)
        return jsonify({"message": result}), 200
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {traceback.format_exc()}")  # Print full error traceback
        return jsonify({"error": str(e)}), 500
        #return jsonify({"error": str(e)}), 500
        
# @app.route("/unexport-volume", methods=["POST"])
# def unexport_volume():
#     data = request.json
#     volume_id = data.get("volume_id")

#     print(f"📢 Received request to unexport volume {volume_id}")  # Debug log

#     if not volume_id:
#         return jsonify({"error": "Missing volume_id"}), 400

#     try:
#         result = storage_mgr.unexport_volume(volume_id)  # ✅ Call the function
#         return jsonify({"message": result}), 200
#     except Exception as e:
#         import traceback
#         print(f"❌ ERROR: {traceback.format_exc()}")  # Debugging
#         return jsonify({"error": str(e)}), 500

data_dir = "data_instance_5000"
volume_file = os.path.join(data_dir, "volume.json")

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

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

# @app.route("/data/exported-volumes", methods=["GET"])
# def get_exported_volumes():
#     """Returns only exported volumes for dropdown selection."""
#     volumes = load_volumes()
#     print("Volumes",volumes)
#     exported_volumes = [v for v in volumes if v.get("is_exported", False)]
#     print(exported_volumes)
#     return jsonify("Exported",exported_volumes), 200

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
        
        volume_file_path = os.path.abspath(r"C:\Users\Pranav\Downloads\HPE_GenAI-main\HPE_GenAI-main\data_instance_5000\volume.json")
        # 🔥 Save changes back to volume.json
        with open(volume_file_path, "w") as f:
            json.dump(volumes, f, indent=4)
            print("💾 Updated volume.json successfully!")

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


# if __name__ == "__main__":
#     app.run(debug=True)

from app1 import load_volumes  # Import your function

volumes = load_volumes()
print("Test Loaded Volumes:", volumes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)