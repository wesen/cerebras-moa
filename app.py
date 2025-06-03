# -*- coding: utf-8 -*-
import copy
import json
import os
from typing import Iterable, Dict, Any, Generator

import streamlit as st
from streamlit_ace import st_ace
from cerebras.cloud.sdk import Cerebras

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Import competition system
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from competition_ui import render_competition_page
    COMPETITION_AVAILABLE = True
except ImportError as e:
    COMPETITION_AVAILABLE = False
    st.error(f"Failed to import competition system: {e}")
    st.stop()

# App configuration - must be first
st.set_page_config(
    page_title="Mixture-Of-Agents & AI Challenge",
    page_icon='static/favicon.ico',
    menu_items={
        'About': "## Cerebras Mixture-Of-Agents \n Powered by [Cerebras](https://cerebras.net)"
    },
    layout="wide"
)

# Default configuration
default_main_agent_config = {
    "main_model": "llama-3.3-70b",
    "cycles": 3,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-4-scout-17b-16e-instruct",
        "temperature": 0.3
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "qwen-3-32b",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3.1-8b",
        "temperature": 0.1
    },
}

# For Cerebras, we use a predefined list of models
valid_model_names = ["llama-3.3-70b", "llama3.1-8b", "llama-4-scout-17b-16e-instruct", "qwen-3-32b"]

# Display banner directly using st.image with absolute path
banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "banner.png")
if os.path.exists(banner_path):
    st.image(banner_path, width=500)
    st.markdown("[Powered by Cerebras](https://cerebras.net)")
else:
    st.error(f"Banner image not found at {banner_path}")
    st.markdown("# Mixture-Of-Agents Powered by Cerebras")
    st.markdown("[Powered by Cerebras](https://cerebras.net)")

st.write("---")

# Helper functions
def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            agent = message['metadata'].get('agent', '')
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append((agent, message['delta']))
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, (agent, output) in enumerate(outputs):
                    with cols[i]:
                        agent_label = f"Agent {i+1}" if not agent else f"{agent}"
                        st.expander(label=agent_label, expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def update_moa_config(update_main_only=False):
    """Auto-save the current configuration to the MOA agent"""
    try:
        # Update the main config with the current number of layers
        new_main_config = copy.deepcopy(st.session_state.moa_main_agent_config)
        new_main_config['cycles'] = st.session_state.num_layers
        
        if update_main_only:
            # Only update the main agent config, not the layer agents
            set_moa_agent(
                moa_main_agent_config=new_main_config,
                override=False
            )
        else:
            # Convert agent configs to layer agent config format
            new_layer_agent_config = {}
            for agent in st.session_state.agent_configs:
                new_layer_agent_config[agent["name"]] = {
                    "system_prompt": agent["system_prompt"],
                    "model_name": agent["model_name"],
                    "temperature": float(agent["temperature"]),
                    "max_tokens": int(agent["max_tokens"])
                }
            
            # Set the MOA agent with the new configuration
            set_moa_agent(
                moa_main_agent_config=new_main_config,
                moa_layer_agent_config=new_layer_agent_config,
                override=True
            )
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    # Only update the main agent config if explicitly provided or not initialized
    if moa_main_agent_config is not None and ("moa_main_agent_config" not in st.session_state or override):
        st.session_state.moa_main_agent_config = moa_main_agent_config

    # Only update the layer agent config if explicitly provided or not initialized
    if moa_layer_agent_config is not None and ("moa_layer_agent_config" not in st.session_state or override):
        st.session_state.moa_layer_agent_config = moa_layer_agent_config
        
        # When layer agent config is updated, also update the agent_configs list to stay in sync
        if "agent_configs" in st.session_state:
            if override:
                st.session_state.agent_configs = []
                for agent_name, agent_config in moa_layer_agent_config.items():
                    st.session_state.agent_configs.append({
                        "name": agent_name,
                        "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                        "model_name": agent_config.get("model_name", "llama3.1-8b"),
                        "temperature": float(agent_config.get("temperature", 0.7)),
                        "max_tokens": int(agent_config.get("max_tokens", 2048))
                    })

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        
        st.session_state.moa_agent = MOAgent.from_config(
            main_model=st_main_copy.pop('main_model'),
            system_prompt=st_main_copy.pop('system_prompt', SYSTEM_PROMPT),
            reference_system_prompt=st_main_copy.pop('reference_system_prompt', REFERENCE_SYSTEM_PROMPT),
            cycles=st_main_copy.pop('cycles', 1),
            temperature=st_main_copy.pop('temperature', 0.1),
            max_tokens=st_main_copy.pop('max_tokens', None),
            layer_agent_config=st_layer_copy,
            **st_main_copy
        )

# Initialize MOA system first
set_moa_agent()

def render_moa_chat_page():
    """Render the MOA chat interface"""
    # Sidebar for configuration
    with st.sidebar:
        st.title("MOA Configuration")

        # Initialize config in session state if not present
        if "agent_configs" not in st.session_state:
            st.session_state.agent_configs = []
            for agent_name, agent_config in st.session_state.moa_layer_agent_config.items():
                st.session_state.agent_configs.append({
                    "name": agent_name,
                    "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                    "model_name": agent_config.get("model_name", "llama3.1-8b"),
                    "temperature": agent_config.get("temperature", 0.7),
                    "max_tokens": agent_config.get("max_tokens", 2048)
                })
        
        # Initialize tracking variables in session state
        if "agent_to_remove" not in st.session_state:
            st.session_state.agent_to_remove = None
        
        if "num_layers" not in st.session_state:
            st.session_state.num_layers = st.session_state.moa_main_agent_config.get('cycles', 1)
        
        # Main Configuration
        st.markdown("### Main Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=20,
                value=st.session_state.num_layers,
                help="The number of processing layers in the MOA architecture."
            )
            if num_layers != st.session_state.num_layers:
                st.session_state.num_layers = num_layers
                update_moa_config(update_main_only=True)
        
        with col2:
            main_model = st.session_state.moa_main_agent_config['main_model']
            if main_model not in valid_model_names:
                default_index = 0
                st.warning(f"Model '{main_model}' not in valid models list.")
            else:
                default_index = valid_model_names.index(main_model)
                
            new_main_model = st.selectbox(
                "Main Model",
                options=valid_model_names,
                index=default_index,
                help="The model used for the final response generation"
            )
            
            if new_main_model != main_model:
                st.session_state.moa_main_agent_config['main_model'] = new_main_model
                update_moa_config(update_main_only=True)
        
        # Main Model Temperature
        main_temp = st.session_state.moa_main_agent_config.get('temperature', 0.1)
        new_main_temp = st.slider(
            label="Main Model Temperature",
            min_value=0.0,
            max_value=1.0,
            value=main_temp,
            step=0.1,
            help="Controls randomness in the main model's output."
        )
        
        if new_main_temp != main_temp:
            st.session_state.moa_main_agent_config['temperature'] = float(new_main_temp)
            update_moa_config(update_main_only=True)
        
        # Orchestration Agent System Prompt Editor
        st.markdown("### Orchestration Agent System Prompt")
        
        current_system_prompt = st.session_state.moa_main_agent_config.get('system_prompt', SYSTEM_PROMPT)
        
        if "orchestration_prompt_input" not in st.session_state:
            st.session_state.orchestration_prompt_input = current_system_prompt
        
        new_system_prompt = st_ace(
            value=st.session_state.orchestration_prompt_input,
            language='text',
            theme='github',
            height=120,
            auto_update=True,
            wrap=True,
            key="orchestration_system_prompt_input"
        )
        st.caption("Required: Your prompt must include {helper_response}")
        
        if new_system_prompt != st.session_state.orchestration_prompt_input:
            st.session_state.orchestration_prompt_input = new_system_prompt
            current_system_prompt = new_system_prompt
        
        # Validate {helper_response} placeholder
        if "{helper_response}" not in new_system_prompt:
            st.error("The system prompt MUST contain {helper_response} placeholder!")
        elif new_system_prompt.count("{helper_response}") > 1:
            st.warning(f"Found {new_system_prompt.count('{helper_response}')} placeholders. Only one is needed.")
        
        if new_system_prompt != current_system_prompt:
            st.session_state.moa_main_agent_config['system_prompt'] = new_system_prompt
            update_moa_config(update_main_only=True)

        # Agent Management
        st.markdown("### Agent Management")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Agents: {len(st.session_state.agent_configs)} | Layers: {st.session_state.num_layers}**")
        with col2:
            if st.button("+ Add Agent", key="add_agent_btn", use_container_width=True):
                agent_id = len(st.session_state.agent_configs) + 1
                agent_name = f"agent_{agent_id}"
                
                existing_names = [agent["name"] for agent in st.session_state.agent_configs]
                while agent_name in existing_names:
                    agent_id += 1
                    agent_name = f"agent_{agent_id}"
                
                st.session_state.agent_configs.append({
                    "name": agent_name,
                    "system_prompt": "Think through your response step by step. {helper_response}",
                    "model_name": "llama3.1-8b",
                    "temperature": 0.7,
                    "max_tokens": 2048
                })
                
                if update_moa_config():
                    st.success(f"Added new agent: {agent_name}")
                    st.rerun()
        
        # Process agent removal
        if st.session_state.agent_to_remove is not None:
            if 0 <= st.session_state.agent_to_remove < len(st.session_state.agent_configs):
                st.session_state.agent_configs.pop(st.session_state.agent_to_remove)
                update_moa_config()
            st.session_state.agent_to_remove = None
            st.rerun()
        
        # Display agents
        if not st.session_state.agent_configs:
            st.info("No agents configured. Add an agent above.")
        else:
            for i, agent in enumerate(st.session_state.agent_configs):
                with st.expander(f"Agent {agent['name']}", expanded=False):
                    # Agent name
                    agent_name = st.text_input(
                        "Agent Name",
                        value=agent["name"],
                        key=f"agent_name_{i}"
                    )
                    if agent_name != agent["name"]:
                        agent["name"] = agent_name
                        update_moa_config()
                    
                    # System prompt
                    prompt_key = f"agent_prompt_{i}_{agent['name']}"
                    if prompt_key not in st.session_state:
                        st.session_state[prompt_key] = agent["system_prompt"]
                    
                    system_prompt = st_ace(
                        value=st.session_state[prompt_key],
                        language='text',
                        theme='github',
                        height=100,
                        auto_update=True,
                        wrap=True,
                        key=f"system_prompt_input_{i}"
                    )
                    st.caption("Required: Must include {helper_response}")
                    
                    if system_prompt != st.session_state[prompt_key]:
                        st.session_state[prompt_key] = system_prompt
                    
                    # Validate placeholder
                    if "{helper_response}" not in system_prompt:
                        st.error(f"Agent '{agent['name']}' prompt MUST contain {{helper_response}} placeholder!")
                    elif system_prompt.count("{helper_response}") > 1:
                        st.warning(f"Agent '{agent['name']}' has multiple placeholders.")
                    
                    if system_prompt != agent["system_prompt"]:
                        agent["system_prompt"] = system_prompt
                        update_moa_config()
                    
                    # Model and settings
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        current_model = agent["model_name"]
                        if current_model not in valid_model_names:
                            default_model_index = 0
                        else:
                            default_model_index = valid_model_names.index(current_model)
                            
                        new_model = st.selectbox(
                            "Model",
                            options=valid_model_names,
                            index=default_model_index,
                            key=f"model_{i}"
                        )
                        if new_model != current_model:
                            agent["model_name"] = new_model
                            update_moa_config()
                    
                    with col2:
                        new_temp = st.slider(
                            "Temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(agent["temperature"]),
                            step=0.1,
                            key=f"temperature_{i}"
                        )
                        if new_temp != agent["temperature"]:
                            agent["temperature"] = new_temp
                            update_moa_config()
                    
                    # Max tokens
                    new_tokens = st.number_input(
                        "Max Tokens",
                        min_value=100,
                        max_value=4096,
                        value=min(int(agent["max_tokens"]), 4096),
                        step=100,
                        key=f"max_tokens_{i}"
                    )
                    if new_tokens != agent["max_tokens"]:
                        agent["max_tokens"] = int(new_tokens)
                        update_moa_config()
                    
                    # Remove button
                    if st.button("Remove Agent", key=f"remove_{i}", use_container_width=True):
                        st.session_state.agent_to_remove = i
                        st.rerun()

    # Credits section
    st.markdown("---")
    st.markdown("""
    ### Credits
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Cerebras](https://cerebras.ai/)
    - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    """)
    
    # Main app layout
    st.header("Mixture of Agents", anchor=False)
    st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Cerebras LLMs.")

    # Chat input
    query = st.chat_input("Ask a question")
    
    # Only show chat interface after user submits a query
    if query:
        try:
            debug_placeholder = st.empty()
            debug_placeholder.info("Processing your question...")
            
            with st.chat_message("user"):
                st.write(query)

            # Reinitialize MOA agent for each query to ensure no conversation history
            st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
            st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
            
            fresh_moa_agent = MOAgent.from_config(
                main_model=st_main_copy.pop('main_model'),
                system_prompt=st_main_copy.pop('system_prompt', SYSTEM_PROMPT),
                reference_system_prompt=st_main_copy.pop('reference_system_prompt', REFERENCE_SYSTEM_PROMPT),
                cycles=st_main_copy.pop('cycles', 1),
                temperature=st_main_copy.pop('temperature', 0.1),
                max_tokens=st_main_copy.pop('max_tokens', None),
                layer_agent_config=st_layer_copy,
                **st_main_copy
            )

            debug_placeholder.info(f"Using model: {fresh_moa_agent.main_model}")
            
            with st.chat_message("assistant"):
                try:
                    ast_mess = stream_response(fresh_moa_agent.chat(query, output_format='json'))
                    response = st.write_stream(ast_mess)
                    debug_placeholder.success("Response generated successfully!")
                except Exception as e:
                    error_str = str(e)
                    
                    if "400" in error_str or "invalid_request" in error_str:
                        st.error("API Request Error")
                        st.code(error_str, language="text")
                        response = "Request error. Please check your configuration."
                    elif "429" in error_str or "rate_limit" in error_str:
                        st.error("Rate Limited")
                        response = "Rate limited. Please wait and try again."
                    elif "Error code:" in error_str:
                        st.error("API Error")
                        st.code(error_str, language="text")
                        response = "API error occurred."
                    else:
                        st.error(f"Error: {error_str}")
                        response = "An error occurred."
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

def main():
    """Main function to control page navigation"""
    st.sidebar.title("Navigation")
    if COMPETITION_AVAILABLE:
        page = st.sidebar.selectbox(
            "Choose a page",
            ["MOA Chat", "AI Configuration Challenge"],
            index=0
        )
    else:
        page = "MOA Chat"
        st.sidebar.info("Competition system not available")

    if page == "AI Configuration Challenge" and COMPETITION_AVAILABLE:
        render_competition_page()
    else:
        render_moa_chat_page()

# Run the main function
main() 