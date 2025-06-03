#!/usr/bin/env python3
import os
import subprocess
import sys

if __name__ == "__main__":
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    streamlit_path = "ai_agent.py"
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_path]
    subprocess.run(cmd) 