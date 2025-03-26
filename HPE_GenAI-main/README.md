# Storage System Simulator

A Python-based REST API app simulating a storage system with JSON persistence and an optional modern UI.

## Setup
1. Install dependencies: `pip install Flask`- Just one for now.
2. Run the app: `python app.py`
   - For UI: `export ENABLE_UI=True` then `python app.py`

## Usage
- API: Access at `http://localhost:5000` (e.g., `POST /system`)
- UI (optional): Visit `http://localhost:5000/ui` when enabled
- Raw JSON: `GET /data/<resource_type>`
- Terminal: API responses print in green to distinguish from persistent JSON

## Features
- CRUD for System, Node, Volume, Host, Settings
- One system per instance (guard rail)
- Flat JSON persistence in `/data`
- Plug-and-play modern UI with dropdowns, toggled with `ENABLE_UI`
- Sidebar to view raw JSON files
