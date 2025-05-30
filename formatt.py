import os
import json
import re
from typing import Any, List, Optional
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# === CONFIG ===
TEXT_PATH = "rca1.txt"
GROQ_API_KEY = "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" #Enter your Groq API key here
GROQ_MODEL = "llama3-8b-8192"

# === LOAD AND CHUNK TEXT FILE ===
print("🔍 Loading and splitting RCA document...")
if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"Text file not found at {TEXT_PATH}")
loader = TextLoader(TEXT_PATH)
docs = loader.load()
if not docs:
    raise ValueError("No content loaded from the text file")
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === EMBEDDINGS AND VECTOR STORE ===
print("📡 Embedding and indexing...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# === GROQ CHAT MODEL ===
llm = ChatOpenAI(
    model=GROQ_MODEL,
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    temperature=0
)

# === RAG CHAIN ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === DATA LOADING ===
def load_json_file(file_path: str) -> Any:
    """Load and return JSON file content."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ Warning: Failed to load {file_path}: {e}")
        return None

def get_relevant_context(port: int, query: str) -> str:
    """Retrieve relevant context from JSON files based on the query."""
    context_parts = []
    data_dir = f"data_instance_{port}"
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: Data directory {data_dir} not found")
        return ""

    # System info
    system_file = f"{data_dir}/system.json"
    system_data = load_json_file(system_file)
    if system_data:
        context_parts.append(f"System Information:\n{json.dumps(system_data, indent=2)}")

    # Latest metrics
    metrics_file = f"{data_dir}/system_metrics.json"
    metrics_data = load_json_file(metrics_file)
    if metrics_data and isinstance(metrics_data, list) and len(metrics_data) > 0:
        latest_metrics = metrics_data[-1]
        context_parts.append(f"Latest Metrics:\n{json.dumps(latest_metrics, indent=2)}")
    else:
        context_parts.append(f"Latest Metrics: Unavailable for port {port}")

    # Volumes info
    volumes_file = f"{data_dir}/volume.json"
    volumes_data = load_json_file(volumes_file)
    if volumes_data:
        volumes = volumes_data if isinstance(volumes_data, list) else [volumes_data]
        context_parts.append(f"Volumes Information:\n{json.dumps(volumes, indent=2)}")

    # IO metrics
    io_metrics_file = f"{data_dir}/io_metrics.json"
    io_metrics_data = load_json_file(io_metrics_file)
    if io_metrics_data:
        context_parts.append(f"IO Metrics:\n{json.dumps(io_metrics_data, indent=2)}")

    # Replication metrics
    replication_file = f"{data_dir}/replication_metrics.json"
    replication_data = load_json_file(replication_file)
    if replication_data:
        context_parts.append(f"Replication Metrics:\n{json.dumps(replication_data, indent=2)}")
    else:
        context_parts.append(f"Replication Metrics: Missing or unavailable for port {port}")

    # Snapshots info
    snapshots_file = f"{data_dir}/snapshots.json"
    snapshots_data = load_json_file(snapshots_file)
    if snapshots_data:
        context_parts.append(f"Snapshots Information:\n{json.dumps(snapshots_data, indent=2)}")

    # Logs
    logs_file = f"{data_dir}/logs_{port}.txt"
    if os.path.exists(logs_file):
        with open(logs_file, 'r') as f:
            logs_content = f.read()[:1000]  # Limit log size
        context_parts.append(f"System Logs:\n{logs_content}")

    context = "\n\n".join(context_parts)
    #print(f"DEBUG: Loaded context for port {port}:\n{context[:500]}...")  # Log partial context
    return context

# === HELPER: Flatten Nested JSON or Context String ===
def flatten_json(obj, prefix=""):
    """Flatten a JSON object or context string into key-value pairs."""
    lines = []
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                lines.extend(flatten_json(v, prefix=full_key))
            else:
                lines.append(f"{full_key} = {v}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            full_key = f"{prefix}[{idx}]"
            if isinstance(item, (dict, list)):
                lines.extend(flatten_json(item, prefix=full_key))
            else:
                lines.append(f"{full_key} = {item}")
    elif isinstance(obj, str):
        sections = obj.split("\n\n")
        for section in sections:
            if section.strip():
                section_lines = section.split('\n')
                section_title = section_lines[0].strip()
                section_content = section
                lines.append(f"{prefix}.{section_title} = {section_content}")
    return lines

# === FAULT REPORT FORMATTING WITH LLaMA ===
def format_fault_report_with_llama(response: dict, system_name: str, port: int) -> str:
    """Format the fault report using LLaMA (via Grok API)."""
    result = response.get("result", "No analysis available")
    
    # Try to parse result as JSON
    try:
        fault_analysis = json.loads(result)
    except json.JSONDecodeError:
        fault_analysis = {"error": "Invalid analysis output", "raw_result": result}

    # Prepare prompt for LLaMA
    formatting_prompt = (
        f"Format the following JSON fault analysis into a concise, human-readable report for system {system_name} (Port: {port}). "
        "Include the fault type, key details (e.g., latency, capacity, saturation), bully volume (highest contributor to the fault) with its contribution percentage, "
        "and relevant volume, snapshot, or replication information. Use bullet points for volumes, snapshots, and replication issues. "
        "Keep it clear, structured, and under 300 words. Avoid raw JSON or code-like formatting. "
        "Example format:\n"
        "Fault Report for {system_name} (Port: {port})\n"
        "Fault Type: <type>\n"
        "Key Details: <metrics>\n"
        "Bully Volume: <volume and contribution>\n"
        "Replication Issues: <details>\n"
        "Volume Information: <details>\n"
        "Snapshot Information: <details>\n"
        "Next Actions: <actions>\n\n"
        f"JSON Analysis:\n{json.dumps(fault_analysis, indent=2)}"
    )

    # Generate formatted report using LLaMA
    formatted_report = llm.invoke(formatting_prompt).content
    
    # Append source snippets (optional)
    report = [formatted_report]
    report.append("\n" + "="*50)
    return "\n".join(report)

# === MAIN LOOP ===
def main():
    print("\n✅ Pure RAG-based Fault Detection System with LLaMA Formatting")
    print("Example queries:")
    print("- Why is system 5000 experiencing high latency?")
    print("- Check replication issues in system 5001")
    print("- Analyze snapshot capacity in system 5000")
    print("- Show faults across all systems")
    print("Type 'exit' to quit\n")

    # Find available ports
    data_dirs = [d for d in os.listdir() if d.startswith("data_instance_") and os.path.isdir(d)]
    ports = [int(d.split("_")[-1]) for d in data_dirs]

    while True:
        query = input("🔎 Enter your query: ").strip()
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting. Take care!")
            break

        try:
            # Extract port from query
            port_match = re.search(r'(?:system|port)\s+(\d+)', query.lower())
            if port_match:
                port = int(port_match.group(1))
                ports_to_check = [port] if port in ports else []
            else:
                ports_to_check = ports

            if not ports_to_check:
                print(f"❌ No valid system/port found. Available ports: {ports}")
                continue

            for port in ports_to_check:
                # Load system data
                context = get_relevant_context(port, query)
                if not context:
                    print(f"❌ No data found for system {port}")
                    continue

                # Flatten context string
                system_data = load_json_file(f"data_instance_{port}/system.json")
                if isinstance(system_data, list) and len(system_data) > 0:
                    system_info = system_data[0]
                else:
                    system_info = system_data
                system_name = system_info.get("name", f"System_{port}") if system_info else f"System_{port}"
                
                flattened = flatten_json(context) if context else []
                formatted_input = "\n".join(flattened)

                # Define JSON structure
                json_structure = """{
    "fault_type": "High latency due to high saturation" or "High latency due to high capacity" or "High latency due to replication link issues" or "No fault",
    "details": {
        "latency": <latency value>,
        "capacity_percentage": <capacity percentage>,
        "saturation": <system saturation percentage>,
        "volume_capacity": <volume capacity percentage>,
        "snapshot_capacity": <snapshot capacity percentage>,
        "high_capacity_volumes": [
            {
                "volume_id": <volume id>,
                "name": <volume name>,
                "capacity_percentage": <capacity percentage>,
                "size": <size>,
                "snapshot_count": <snapshot count>
            }
        ],
        "high_saturation_volumes": [
            {
                "volume_id": <volume id>,
                "name": <volume name>,
                "throughput": <throughput in MB/s>,
                "saturation_contribution": <saturation contribution percentage>
            }
        ],
        "snapshot_details": [
            {
                "volume_id": <volume id>,
                "name": <volume name>,
                "snapshot_count": <snapshot count>,
                "capacity_contribution": <capacity contribution percentage>
            }
        ],
        "replication_issues": [
            {
                "volume_id": <volume id>,
                "volume_name": <volume name>,
                "target_id": <target system id>,
                "target_system_name": <target system name>,
                "latency": <latency value>,
                "timestamp": <timestamp>
            }
        ],
        "bully_volume": {
            "volume_id": <volume id>,
            "name": <volume name>,
            "contribution_percentage": <percentage>
        }
    }
}"""

                # Construct analysis prompt without hardcoded RCA logic
                analysis_prompt = (
                    "You are an RCA assistant. Based on the following structured system data and the RCA logic retrieved from rca.txt, diagnose the root cause of the issue described in this question:\n\n"
                    f'"{query}"\n\n'
                    "Extracted system data:\n"
                    f"{formatted_input}\n\n"
                    "Use the RCA logic from rca.txt to identify the highest causing fault (saturation, capacity, or replication). "
                    "Follow these steps:\n"
                    "1. Apply the fault diagnosis rules from rca.txt to determine the fault type, prioritizing replication issues when system latency is high and saturation/capacity are low.\n"
                    "2. Identify the bully volume (volume with the highest contribution to the fault) and calculate its contribution percentage as per rca1.txt. For replication issues, assign 100% to the primary affected volume if no other volumes are involved.\n"
                    "3. Return only the highest causing fault.\n"
                    "4. Include relevant details such as latency, capacity, saturation, volume, snapshot, and replication information.\n"
                    "5. If replication metrics are available, explicitly check for replication latency issues as per rca.txt.\n"
                    "6. Ensure the response is based solely on rca.txt logic and system data, without assuming hardcoded thresholds.\n"
                    "Return a JSON object with the following structure, ensuring the bully_volume field is populated for bully volume:\n\n"
                    f"{json_structure}"
                )

                # Run analysis and log raw response for debugging
                response = qa_chain.invoke(analysis_prompt)
                #print("DEBUG: Raw RAG Response:", response["result"])
                print("DEBUG: Retrieved RCA Chunks:", [doc.page_content[:200] for doc in response["source_documents"]])
                print(format_fault_report_with_llama(response, system_name, port))

        except Exception as e:
            print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()