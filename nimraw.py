import os
import json
import re
from typing import Any, Dict, List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import TypedDict

# === CONFIG ===
TEXT_PATH = "rca1.txt"
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# === Initialize LLM ===
llm = ChatOpenAI(
    model=GROQ_MODEL,
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    temperature=0
)

# === Load and Chunk RCA Document ===
print("🔍 Loading and splitting RCA document...")
if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"Text file not found at {TEXT_PATH}")
loader = TextLoader(TEXT_PATH)
docs = loader.load()
if not docs:
    raise ValueError("No content loaded from the text file")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === Embeddings and Vector Store ===
print("📡 Embedding and indexing...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# === State Definition for LangGraph ===
class AgentState(TypedDict):
    query: str
    port: int
    system_name: str
    context: str
    fault_analysis: Dict[str, Any]
    formatted_report: str
    raw_analysis_response: str  # New field for raw LLM analysis response
    raw_formatting_response: str  # New field for raw LLM formatting response

# === Agent 1: Data Extraction Agent ===
def extract_relevant_data(state: AgentState) -> AgentState:
    """Extract relevant files and context for the given port and query."""
    query = state["query"]
    port_match = re.search(r'(?:system|port)\s+(\d+)', query.lower())
    port = int(port_match.group(1)) if port_match else 5000  # Default port if none specified

    # Load system data
    data_dir = f"data_instance_{port}"
    if not os.path.exists(data_dir):
        state["context"] = f"⚠️ Warning: Data directory {data_dir} not found"
        state["port"] = port
        state["system_name"] = f"System_{port}"
        state["raw_analysis_response"] = ""
        state["raw_formatting_response"] = ""
        return state

    context_parts = []
    
    # System info
    system_file = f"{data_dir}/system.json"
    if os.path.exists(system_file):
        with open(system_file, 'r') as f:
            system_data = json.load(f)
        if isinstance(system_data, list) and len(system_data) > 0:
            system_name = system_data[0].get("name", f"System_{port}")
            context_parts.append(f"System Information:\n{json.dumps(system_data[0], indent=2)}")
        else:
            system_name = system_data.get("name", f"System_{port}") if system_data else f"System_{port}"
            context_parts.append(f"System Information:\n{json.dumps(system_data, indent=2)}")

    # Latest metrics
    metrics_file = f"{data_dir}/system_metrics_{port}.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        if metrics_data and isinstance(metrics_data, list) and len(metrics_data) > 0:
            context_parts.append(f"Latest Metrics:\n{json.dumps(metrics_data[-1], indent=2)}")

    # Volumes info
    volumes_file = f"{data_dir}/volume.json"
    if os.path.exists(volumes_file):
        with open(volumes_file, 'r') as f:
            volumes_data = json.load(f)
        context_parts.append(f"Volumes Information:\n{json.dumps(volumes_data, indent=2)}")

    # IO metrics
    io_metrics_file = f"{data_dir}/io_metrics.json"
    if os.path.exists(io_metrics_file):
        with open(io_metrics_file, 'r') as f:
            io_metrics_data = json.load(f)
        context_parts.append(f"IO Metrics:\n{json.dumps(io_metrics_data, indent=2)}")

    # Replication metrics
    replication_file = f"{data_dir}/replication_metrics_{port}.json"
    if os.path.exists(replication_file):
        with open(replication_file, 'r') as f:
            replication_data = json.load(f)
        context_parts.append(f"Replication Metrics:\n{json.dumps(replication_data, indent=2)}")

    # Snapshots info
    snapshots_file = f"{data_dir}/snapshots.json"
    if os.path.exists(snapshots_file):
        with open(snapshots_file, 'r') as f:
            snapshots_data = json.load(f)
        context_parts.append(f"Snapshots Information:\n{json.dumps(snapshots_data, indent=2)}")

    # Logs
    logs_file = f"{data_dir}/logs_{port}.txt"
    if os.path.exists(logs_file):
        with open(logs_file, 'r') as f:
            logs_content = f.read()[:1000]  # Limit log size
        context_parts.append(f"System Logs:\n{logs_content}")

    state["context"] = "\n\n".join(context_parts)
    state["port"] = port
    state["system_name"] = system_name
    state["raw_analysis_response"] = ""  # Initialize
    state["raw_formatting_response"] = ""  # Initialize
    return state

# === Agent 2: Fault Analysis Agent ===
def analyze_fault(state: AgentState) -> AgentState:
    """Analyze the fault using rca1.txt logic and system data."""
    query = state["query"]
    context = state["context"]
    
    # Flatten context for analysis
    def flatten_json(obj, prefix=""):
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

    flattened = flatten_json(context) if context else []
    formatted_input = "\n".join(flattened)

    # JSON structure for fault analysis
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

    # Construct analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an RCA assistant. Based on the structured system data and the RCA logic retrieved from rca1.txt, diagnose the root cause of the issue described in the query. "
            "Steps:\n"
            "1. Apply the fault diagnosis rules from rca1.txt to determine the fault type, prioritizing replication issues when system latency is high and saturation/capacity are low.\n"
            "2. Identify the bully volume (highest contributor to the fault) and calculate its contribution percentage as per rca1.txt. For replication issues, assign 100% to the primary affected volume if no other volumes are involved.\n"
            "3. Return only the highest causing fault.\n"
            "4. Include relevant details such as latency, capacity, saturation, volume, snapshot, and replication information.\n"
            "5. If replication metrics are available, check for replication impairment issues as per rca1.txt.\n"
            "6. Ensure the response is based solely on rca1.txt logic and system data, without assuming hardcoded thresholds.\n"
            "7. Check latency, saturation, and capacity values before retrieving relevant chunks.\n"
            "8. Latency < 3ms indicates no fault.\n"
            "Return a JSON object with the structure provided."
        )),
        HumanMessage(content=(
            f"Query: {query}\n\n"
            f"Extracted system data:\n{formatted_input}\n\n"
            f"Expected JSON structure:\n{json_structure}"
        ))
    ])

    # Retrieve relevant RCA chunks
    relevant_docs = retriever.invoke(query)
    context_with_rca = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Update the system message with RCA context
    system_message = analysis_prompt.messages[0].content + f"\nRCA Logic from rca1.txt:\n{context_with_rca}"
    
    # Create the final messages list
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=analysis_prompt.messages[1].content)
    ]
    
    # Invoke the LLM with the messages
    response = llm.invoke(messages)
    state["raw_analysis_response"] = response.content  # Store raw LLM response
    try:
        fault_analysis = json.loads(response.content)
    except json.JSONDecodeError:
        fault_analysis = {"error": "Invalid analysis output", "raw_result": response.content}

    state["fault_analysis"] = fault_analysis
    return state

# === Agent 3: Response Formatting Agent ===
def format_response(state: AgentState) -> AgentState:
    """Format the fault analysis into a human-readable report."""
    fault_analysis = state["fault_analysis"]
    system_name = state["system_name"]
    port = state["port"]

    formatting_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            f"Format the following JSON fault analysis into a concise, human-readable report for system {system_name} (Port: {port}). "
            "Include the fault type, key details (e.g., latency, capacity, saturation), bully volume (highest contributor to the fault) with its contribution percentage, "
            "and relevant volume, snapshot, or replication information. Use bullet points for volumes, snapshots, and replication issues. "
            "Keep it clear, structured, and under 300 words. Avoid raw JSON or code-like formatting. "
            "Only include the highest causing fault and ensure the report is actionable."
            "Only the Volume contribution part should include the volume contribution details of all volumes (volume+snapshot) in the report as per rca1.txt logic, volume contribution is (volume size+snapshot contribution)*100/max_capacity(from system.json).\n"
            "System saturation contribution per volume = (Volume Throughput / Max Throughput) × 100, max throughput can be inferred from system.json"
            "Example format:\n"
            f"Fault Report for {system_name} (Port: {port})\n"
            "Fault Type: <type>\n"
            "Key Details: <metrics>\n"
            "Bully Volume: <volume and contribution>\n"
            "Replication Issues: <details>\n"
            "Volume Information: <details>\n"
            "Snapshot Information: <details>\n"
            "Volume contribution:<details>"
            "Next Actions: <actions>\n"
        )),
        HumanMessage(content=f"JSON Analysis:\n{json.dumps(fault_analysis, indent=2)}")
    ])

    # Create the final messages list
    messages = [
        SystemMessage(content=formatting_prompt.messages[0].content),
        HumanMessage(content=formatting_prompt.messages[1].content)
    ]
    
    # Invoke the LLM with the messages
    formatted_report = llm.invoke(messages).content
    state["raw_formatting_response"] = formatted_report  # Store raw LLM response
    state["formatted_report"] = formatted_report
    return state

# === LangGraph Workflow ===
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("extract_data", extract_relevant_data)
workflow.add_node("analyze_fault", analyze_fault)
workflow.add_node("format_response", format_response)

# Define edges
workflow.add_edge("extract_data", "analyze_fault")
workflow.add_edge("analyze_fault", "format_response")
workflow.add_edge("format_response", END)

# Set entry point
workflow.set_entry_point("extract_data")

# Compile the graph
app = workflow.compile()

# === Main Loop ===
def main():
    print("\n✅ LangGraph Agentic Fault Detection System")
    print("Example queries:")
    print("- Why is system 5000 experiencing high latency?")
    print("- Check replication issues in system 5001")
    print("- Analyze snapshot capacity in system 5000")
    print("- Show faults across all systems")
    print("Type 'exit' to quit\n")

    while True:
        query = input("🔎 Enter your query: ").strip()
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting. Take care!")
            break

        try:
            # Initialize state
            state = {
                "query": query,
                "port": 0,
                "system_name": "",
                "context": "",
                "fault_analysis": {},
                "formatted_report": "",
                "raw_analysis_response": "",
                "raw_formatting_response": ""
            }
            
            # Run the workflow
            result = app.invoke(state)
            
            # Print the raw LLM responses and formatted report
            print("\n" + "="*50)
            print("Raw LLM Analysis Response:")
            print(result["raw_analysis_response"])
            print("\n" + "-"*50)
            print("Raw LLM Formatting Response:")
            print(result["raw_formatting_response"])
            print("\n" + "="*50)
            print("Formatted Report:")
            print(result["formatted_report"])
            print("="*50 + "\n")

        except Exception as e:
            print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()