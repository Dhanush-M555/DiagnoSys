The changes made are as follows:
1) The create system doesn't have an option to enter the system name, it is set to the port number by default
2) The select system is normally being updated in all dropdowns on loading the window
3) The log.txt file was global earlier, it has been moved into the data_instance_portno. folder
4) The settings panel can be used to create the settings. For replication settings, the target system needs to be selected,
For asynchronous replication , the frequency of replication can be set as a value ![check image](image-1.png)
5) The setting can then be applied to the volume 
6) Multiple replication and snapshot settings are supported
7) Export-unexport function has been fixed

Mar 22,2025 | Commit Changes

1) Multiple Settings per Volume – A single volume can now have multiple settings, such as one snapshotSettings and multiple replicationSettings.

2) Consistent Logging Format– Logging is now standardized across the system. I’ve implemented both local system-level logging and global logging for better traceability. Also, added a Show Logs button on the UI (not sure if this is necessary!).

3) Additional Configurations – Added max_throughput and max_capacity for the system, as well as Size for volumes. These are now tracked in system_metrics.json (currently storing just two values).

4) Replication Implementation – Each volume can now be replicated with different settings across multiple systems. A separate endpoint facilitates communication between the source and target while logging the replication details.

5) Replication Metrics Tracking – Replication-specific metrics are now stored in replication_metrics.json.

Extras:
6) Added clear.py—a simple script to clear data files and folders for testing purposes. Thought it might be useful!

