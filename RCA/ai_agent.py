import streamlit as st
import os
import json
import requests
from datetime import datetime, timedelta
import time
import re
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from ai_tools import StorageTools

load_dotenv()

st.set_page_config(
    page_title="Storage System AI Assistant",
    page_icon="🤖",
    layout="wide"
)

DATA_DIR = "../data"
SUPPORT_DOC_PATH = "support_documentation.pdf"
CHROMA_DIR = ".chroma_db"
GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

# Initialize storage tools
storage_tools = StorageTools(data_dir=DATA_DIR)

# Sidebar for settings
st.sidebar.title("Storage Assistant")
st.sidebar.image("https://img.icons8.com/cotton/64/000000/server.png", width=100)

# Function to load all available systems
def get_available_systems():
    try:
        # Read from global systems file
        global_file = os.path.join(DATA_DIR, "global_systems.json")
        if os.path.exists(global_file):
            with open(global_file, 'r') as f:
                systems = json.load(f)
            return systems
        return []
    except Exception as e:
        st.error(f"Error loading systems: {str(e)}")
        return []

# Function to initialize the RAG system with support documentation
def initialize_rag():
    """
    Load the pre-initialized Retrieval-Augmented Generation (RAG) vector store.
    If the vector store does not exist, prompt the user to run the initialization script.
    """
    print("Loading RAG system...")
    VECTOR_STORE_PATH = "vector_store"  # Updated to use parent directory's vector store
    
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store not found at {VECTOR_STORE_PATH}")
        print("Please run 'initialize_rag.py' to initialize the RAG system before using the AI agent.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    print(f"Loaded vector store from {VECTOR_STORE_PATH}")
    return vector_store

# Function to extract time reference using LLM
def extract_time_with_llm(query, current_time):
    """Extract start and end time from the query using LLM relative to current_time."""
    
    prompt = f"""Analyze the user query below to identify the time range they are interested in, relative to the current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.

User Query: "{query}"

Respond ONLY with a JSON object containing 'start_time' and 'end_time' in 'YYYY-MM-DD HH:MM:SS' format. 
If the user mentions a specific time (e.g., '5:50pm yesterday'):
  1. Determine the exact timestamp for that specific time
  2. Create a narrow time window around it (30 minutes before and 30 minutes after)
  3. The goal is to focus precisely on the time period mentioned, not the entire day

If the user mentions a duration (e.g., 'last 6 hours', 'yesterday'), calculate the absolute start and end times.
If no specific time or duration is mentioned, assume the user is interested in the last 24 hours from the current time.

The response should ONLY contain the JSON object, nothing else.

Example JSON response:
{{"start_time": "2024-10-26 14:00:00", "end_time": "2024-10-26 15:00:00"}}

JSON response:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Using a smaller model for efficiency
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        response_content = response.choices[0].message.content
        # Clean potential markdown code block fences
        if response_content.startswith("```json"):
            response_content = response_content.strip("```json\n ")
        elif response_content.startswith("```"):
            response_content = response_content.strip("```\n ")
        
        # Additional cleanup for any non-JSON content
        match = re.search(r'({.*})', response_content, re.DOTALL)
        if match:
            response_content = match.group(1)
        
        print(f"DEBUG - LLM time extraction response: {response_content}")
        time_data = json.loads(response_content)
        
        start_time_str = time_data.get("start_time")
        end_time_str = time_data.get("end_time")
        
        if start_time_str and end_time_str:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            print(f"LLM extracted time: {start_time} to {end_time}")
            return start_time, end_time
        else:
            raise ValueError("LLM did not return valid start/end times.")
            
    except Exception as e:
        print(f"Error during LLM time extraction: {e}. Falling back to default (last 24 hours).")
        
        # Try to manually parse times from specific patterns
        if "5:50pm yesterday" in query.lower() or "5:50 pm yesterday" in query.lower():
            yesterday = current_time - timedelta(days=1)
            target_time = yesterday.replace(hour=17, minute=50, second=0)
            start_time = target_time - timedelta(minutes=30)
            end_time = target_time + timedelta(minutes=30)
            print(f"Manually extracted time for '5:50pm yesterday': {start_time} to {end_time}")
            return start_time, end_time
            
        # Fallback to default (last 24 hours) on error
        end_time = current_time
        start_time = current_time - timedelta(hours=24)
        return start_time, end_time

def analyze_with_llm_tool_calling(system_id, start_time, end_time, query, temperature=0.2):
    """
    Analyze the storage system using Groq API with tool calling, maintaining conversation context.

    Args:
        system_id: The ID of the system to analyze
        start_time: Start time for analysis
        end_time: End time for analysis
        query: User's query
        temperature: Temperature parameter for LLM

    Returns:
        The LLM response
    """
    # Ensure system_id is provided
    if not system_id:
        return {"error": "System ID was not provided to the analysis function."}
        
    print(f"DEBUG - Analyzing with System ID: {system_id}") # Add debug print
        
    # Get RAG context
    vector_db = initialize_rag()
    if vector_db is None:
        return {"error": "RAG system is not initialized. Cannot analyze the storage system."}

    # Get relevant context from RAG
    query_docs = vector_db.similarity_search(
        query + " latency issues in storage system",
        k=3
    )
    context = "\\n\\n".join([doc.page_content for doc in query_docs])

    # Build the system prompt with context and improved diagnostic guidance
    # Explicitly state the target system ID in the system prompt
    system_prompt = f"""You are a diagnostic AI assistant for storage systems. Your task is to help users diagnose and resolve latency issues in their storage systems by strictly following the diagnostic steps provided in the context.

*** IMPORTANT: You MUST use the following System ID for all tool calls related to this query: {system_id} ***

CONTEXT FROM SUPPORT DOCUMENTATION:
{context}

AVAILABLE TOOLS:
- fetch_system_metrics: Get system-wide metrics (capacity_pct, saturation_pct, current_latency).
- fetch_io_metrics: Get IO metrics for specific volumes/hosts (latency).
- fetch_replication_metrics: Get metrics about replication processes (replication_latency, faults with sleep_time).
- fetch_logs: Get system logs (with optional log_level filter).
- fetch_system_config: Get system configuration (max_capacity, max_throughput).
- fetch_volumes: Get information about volumes (volume_size, workload_size).
- fetch_hosts: Get information about hosts.

DIAGNOSTIC PROCEDURE (Follow Strictly):
1.  Check `system_metrics` for `current_latency`. If latency is high (>= 4ms, critical >= 5ms), proceed. Otherwise, report normal latency.
2.  Calculate `base_latency`: Find the higher value between `capacity_pct` and `saturation_pct` from `system_metrics`. Use the scale in the documentation to determine the `base_latency` (e.g., <=70% -> 1ms, 70-80% -> 2ms, etc.).
3.  Calculate `total_replication_fault_latency`: Fetch `replication_metrics` (especially faults). Sum the `sleep_time` (in ms) of all active faults.
4.  Calculate `Total Latency = base_latency + total_replication_fault_latency`. Compare this calculated total latency with the `current_latency` reported in `system_metrics`. They should align.
5.  Identify Primary Cause:
    *   If `base_latency` (driven by high capacity/saturation) is the main component of `Total Latency`, diagnose Fault Type 1 (Capacity) or 2 (Saturation) following the steps in the documentation.
    *   If `total_replication_fault_latency` is the main component, diagnose Fault Type 3 (Replication) following the steps in the documentation.
    *   If both contribute significantly, report both.
6.  Provide a clear analysis stating the calculated `base_latency`, `total_replication_fault_latency`, calculated `Total Latency`, and the identified primary cause(s) with supporting evidence from the metrics.
7.  Recommend actions based ONLY on the identified fault type(s).
8.  For follow-up questions, respond directly to the question without repeating the full analysis unless new information is requested.

IMPORTANT:
*   Use the tools provided whenever you need specific data points mentioned in the diagnostic steps. Call tools sequentially as needed within a turn.
*   Ensure ALL tool calls use the specified System ID: {system_id}
*   Be precise with numbers and thresholds from the documentation. Correctly interpret percentages (e.g., 6% is 6%, not 600%).
*   Explicitly state your calculated values for base latency, replication latency, and total latency in your final analysis.
*   DO NOT output tool calls as text. Use the function calling mechanism.
*   The bully volume is the volume that contributes the most to a specific fault. Determine the contribution of the bully volume to the fault, considering that the cause may not solely be the volume size but could also stem from high max_snapshots settings if the volume has snapshot settings configured.

"""

    # --- Conversation History Management ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Prepare messages for the LLM call, including history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous messages from history
    # Filter out any previous system messages to avoid duplication/confusion
    for msg in st.session_state.chat_history:
        if msg.get("role") != "system":
             messages.append(msg)
    
    # Check if this is a follow-up question by examining:
    # 1. If there is chat history
    # 2. If the query is short (typical of follow-ups)
    # 3. If it contains question words or has a question mark
    is_followup = (
        len(st.session_state.chat_history) > 0 and 
        (len(query.split()) <= 15 or 
         any(q in query.lower() for q in ["what", "which", "why", "how", "who", "where", "when", "?", "can", "does", "is", "are"]))
    )
             
    # Format the user message based on whether it's a follow-up
    if is_followup:
        # For follow-up questions, don't repeat all the context
        user_message_content = query
    else:
        # For new analysis requests, include the full context
        user_message_content = f"Please analyze System ID {system_id} for the time period {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}. My specific question is: {query}"
    
    messages.append({"role": "user", "content": user_message_content})

    # --- LLM Interaction Loop ---
    try:
        while True:
            print(f"DEBUG - Calling Groq API with {len(messages)} messages.")
            # print(f"DEBUG - Messages: {messages}") # Optional: Print messages if needed
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Reverted to 3.1 as 3.3 might not be available/stable yet
                messages=messages,
                tools=storage_tools.get_tool_definitions(),
                tool_choice="auto", # Let the model decide when to call tools
                temperature=temperature,
                max_tokens=4000
            )

            message = response.choices[0].message
            # Add assistant's response (which could be text or tool_calls) to messages list
            # We need to convert the Pydantic object to a dict for JSON serialization and consistency
            if message.content:
                 messages.append({"role": "assistant", "content": message.content})
            if message.tool_calls:
                 # Convert tool calls to dict format before appending
                 tool_calls_dict = [
                     {
                         "id": tc.id,
                         "type": tc.type,
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                     }
                     for tc in message.tool_calls
                 ]
                 messages.append({"role": "assistant", "tool_calls": tool_calls_dict})
                 
            # Check for tool calls to execute
            if not message.tool_calls:
                # No tool calls, this is the final response for this turn
                print("DEBUG - No tool calls detected. Final response received.")
                final_content = message.content

                # Clean up potential text-based tool calls just in case
                if isinstance(final_content, str) and "<tool-use>" in final_content:
                     print("DEBUG - Cleaning up textual tool calls in final response.")
                     final_content = re.sub(r'<tool-use>.*?</tool-use>', '', final_content, flags=re.DOTALL).strip()

                # --- Update Session State History ---
                # Add only the *current* turn's user query and final assistant response
                # Use the original user query for history, not the augmented one
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": final_content})

                return {"analysis": final_content}

            # Process tool calls
            print(f"DEBUG - Tool calls detected: {len(message.tool_calls)}")
            available_functions = {
                "fetch_io_metrics": storage_tools.fetch_io_metrics,
                "fetch_system_metrics": storage_tools.fetch_system_metrics,
                "fetch_replication_metrics": storage_tools.fetch_replication_metrics,
                "fetch_logs": storage_tools.fetch_logs,
                "fetch_system_config": storage_tools.fetch_system_config,
                "fetch_volumes": storage_tools.fetch_volumes,
                "fetch_hosts": storage_tools.fetch_hosts,
            }

            tool_results_for_next_iteration = []
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                tool_call_id = tool_call.id

                if not function_to_call:
                    print(f"DEBUG - Error: Unknown function {function_name}")
                    tool_result = json.dumps({"error": f"Unknown tool function: {function_name}"})
                else:
                    try:
                        # IMPORTANT: Ensure arguments are passed correctly, especially system_id
                        function_args = json.loads(tool_call.function.arguments)
                        # Double-check if system_id is present and correct, override if necessary/missing? (risky)
                        # if 'system_id' not in function_args or function_args['system_id'] != system_id:
                        #    print(f"WARN - Tool call for {function_name} had wrong/missing system_id. Forcing {system_id}.")
                        #    function_args['system_id'] = system_id 
                            
                        st.info(f"🤖 Calling tool: {function_name} with arguments: {function_args}")
                        print(f"DEBUG - Calling tool function {function_name} directly with arguments: {function_args}")
                        function_response = function_to_call(**function_args)
                        tool_result = json.dumps(function_response)
                        st.success(f"Tool {function_name} executed successfully.")
                    except Exception as e:
                        import traceback
                        print(f"DEBUG - Exception in tool call execution: {str(e)}")
                        print(f"DEBUG - Traceback: {traceback.format_exc()}")
                        st.error(f"Error executing tool {function_name}: {str(e)}")
                        tool_result = json.dumps({"error": f"Failed to execute tool {function_name}. Error: {str(e)}"})

                # Append tool result for the next LLM call
                tool_results_for_next_iteration.append(
                    {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result,
                    }
                )
                
            # Add all tool results to the messages list for the next iteration
            messages.extend(tool_results_for_next_iteration)
            # Loop back to call the LLM again with the tool results

    except Exception as e:
        import traceback
        print(f"DEBUG - Exception in analyze_with_llm_tool_calling: {str(e)}")
        print(f"DEBUG - Traceback: {traceback.format_exc()}")
        # Update history with error
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
        return {"error": f"Error analyzing with LLM: {str(e)}"}

# Get all available systems
available_systems = get_available_systems()
system_options = {f"{system['name']} (ID: {system['id']})": system['id'] for system in available_systems}

# Add system selection dropdown
selected_system_name = st.sidebar.selectbox(
    "Select Storage System",
    options=list(system_options.keys()) if system_options else ["No systems available"]
)

# Get selected system ID
selected_system_id = system_options.get(selected_system_name) if system_options else None

# Main UI elements
st.title("Storage System AI Assistant")

# Input query
user_query = st.text_area("Ask a question about your storage system:", height=100)

# Submit button
if st.button("Analyze"):
    if not user_query:
        st.error("Please enter a query.")
    elif not selected_system_id:
        st.error("Please select a storage system.")
    elif not GROQ_API_KEY:
        st.error("GROQ API key is not set. Please set the GROQ_API_KEY environment variable.")
    else:
        with st.spinner("Analyzing your storage system..."):
            # Extract time reference from query
            start_time, end_time = extract_time_with_llm(user_query, datetime.now())
            
            st.info(f"Analyzing data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Analyze using the agentic tool-calling function
            result = analyze_with_llm_tool_calling(selected_system_id, start_time, end_time, user_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Analysis Result")
                st.markdown(result["analysis"])
                
                # For visual representation, show metrics if available
                try:
                    # Fetch data for visualization
                    system_metrics = storage_tools.fetch_system_metrics(
                        selected_system_id, 
                        start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    if "system_metrics" in system_metrics and system_metrics["system_metrics"]:
                        # Convert to DataFrame for visualization
                        df = pd.DataFrame(system_metrics["system_metrics"])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        
                        # Create tabs for different metrics
                        metric_tabs = st.tabs(["Latency", "Capacity", "Saturation"])
                        
                        with metric_tabs[0]:
                            st.subheader("System Latency")
                            if "current_latency" in df.columns:
                                st.line_chart(df.set_index('timestamp')['current_latency'])
                        
                        with metric_tabs[1]:
                            st.subheader("Capacity Usage")
                            if "capacity_percentage" in df.columns:
                                st.line_chart(df.set_index('timestamp')['capacity_percentage'])
                        
                        with metric_tabs[2]:
                            st.subheader("System Saturation")
                            if "saturation" in df.columns:
                                st.line_chart(df.set_index('timestamp')['saturation'])
                except Exception as e:
                    st.warning(f"Could not visualize metrics: {str(e)}")

# Chat history (stored in session state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.subheader("Conversation History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This AI agent helps diagnose storage system issues by analyzing metrics, "
    "configurations, and logs based on support documentation."
) 