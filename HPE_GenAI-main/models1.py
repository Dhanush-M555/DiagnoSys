class System:
    def __init__(self, id, name, max_throughput=200, max_capacity=1024):
        self.id = id
        self.name = name
        self.max_throughput = max_throughput  # MBPS
        self.max_capacity = max_capacity      # GB

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "max_throughput": self.max_throughput,
            "max_capacity": self.max_capacity
        }

class Volume:
    def __init__(self, id, name, system_id, size=0, is_exported=False, exported_host_id=None, workload_size=0, snapshot_settings=None, replication_settings=None):
        self.id = id
        self.name = name
        self.system_id = system_id
        self.size = size  # Size in GB
        self.is_exported = is_exported
        self.exported_host_id = exported_host_id
        self.workload_size = workload_size
        self.snapshot_settings = snapshot_settings or {}  # Keep as dict
        self.replication_settings = replication_settings or []  # Change to list

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "system_id": self.system_id,
            "size": self.size,
            "is_exported": self.is_exported,
            "exported_host_id": self.exported_host_id,
            "workload_size": self.workload_size,
            "snapshot_settings": self.snapshot_settings,
            "replication_settings": self.replication_settings
        }

class Host:
    def __init__(self, id, system_id, name, application_type, protocol):
        self.id = id
        self.system_id = system_id
        self.name = name
        self.application_type = application_type
        self.protocol = protocol

    def to_dict(self):
        return {
            "id": self.id,
            "system_id": self.system_id,
            "name": self.name,
            "application_type": self.application_type,
            "protocol": self.protocol
        }

class Settings:
    def __init__(self, id, system_id, name, type, value=None, replication_type=None, delay_sec=0, replication_target=None):
        self.id = id
        self.system_id = system_id
        self.name = name
        self.type = type
        # Only set value if type is not replication
        self.value = value if type != "replication" else None
        self.replication_type = replication_type  # 'synchronous' or 'asynchronous'
        self.delay_sec = delay_sec  # 0 for sync, >0 for async
        self.replication_target = replication_target  # Target system info

    def to_dict(self):
        data = {
            "id": self.id,
            "system_id": self.system_id,
            "name": self.name,
            "type": self.type,
        }
        
        if self.type == "replication":
            data.update({
                "replication_type": self.replication_type,
                "delay_sec": self.delay_sec,
                "replication_target": self.replication_target
            })
        else:
            data["value"] = self.value
            
        return data
