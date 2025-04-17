from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from models.fault_detector import StorageFaultDetector

app = FastAPI(title="Storage Fault Detection API")

# Get the workspace root directory
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize the fault detector
fault_detector = StorageFaultDetector(data_dir=workspace_root)

class FaultResponse(BaseModel):
    faults: List[Dict[str, Any]]
    system_state: Dict[str, Any]

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_system():
    """Analyze the storage system for potential faults."""
    try:
        result = fault_detector.analyze_system_state()
        
        # Convert the formatted report to HTML
        formatted_report = fault_detector.format_fault_report(result)
        
        # Convert to HTML with proper formatting
        html_report = formatted_report.replace("\n", "<br>")
        html_report = html_report.replace("=== ", "<h1>").replace(" ===", "</h1>")
        html_report = html_report.replace("FAULT #", "<h2>FAULT #").replace(":", "</h2>")
        html_report = html_report.replace("DETAILS:", "<h3>DETAILS:</h3>")
        html_report = html_report.replace("CONTEXT:", "<h3>CONTEXT:</h3>")
        html_report = html_report.replace("RELATED METRICS:", "<h3>RELATED METRICS:</h3>")
        html_report = html_report.replace("RECENT LOGS:", "<h3>RECENT LOGS:</h3>")
        html_report = html_report.replace("SYSTEM STATE SUMMARY:", "<h2>SYSTEM STATE SUMMARY:</h2>")
        html_report = html_report.replace("Current Metrics:", "<h3>Current Metrics:</h3>")
        html_report = html_report.replace("Recent Logs:", "<h3>Recent Logs:</h3>")
        
        # Add styling and auto-refresh functionality
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Storage System Fault Analysis</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #e74c3c; margin-top: 20px; }}
                h3 {{ color: #2980b9; }}
                .fault {{ background-color: #f9f9f9; border-left: 5px solid #e74c3c; padding: 10px; margin: 10px 0; }}
                .instance {{ background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 10px; margin: 10px 0; }}
                .metrics {{ background-color: #f0fff0; padding: 5px; }}
                .logs {{ background-color: #fff0f0; padding: 5px; font-family: monospace; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
                .confidence {{ color: #e67e22; font-weight: bold; }}
                .severity {{ color: #c0392b; font-weight: bold; }}
                .controls {{ 
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .refresh-btn {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                .refresh-btn:hover {{
                    background-color: #2980b9;
                }}
                .auto-refresh {{
                    margin-top: 10px;
                    font-size: 0.9em;
                    color: #7f8c8d;
                }}
            </style>
            <script>
                function refreshPage() {{
                    window.location.reload();
                }}
            </script>
        </head>
        <body>
            <div class="controls">
                <button class="refresh-btn" onclick="refreshPage()">Refresh Now</button>
                <div class="auto-refresh">Auto-refreshing every 30 seconds</div>
            </div>
            <div class="report">
                {html_report}
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/json")
async def analyze_system_json():
    """Analyze the storage system for potential faults and return JSON response."""
    try:
        result = fault_detector.analyze_system_state()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 