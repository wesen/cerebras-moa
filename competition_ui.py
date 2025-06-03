import streamlit as st
import time
import asyncio
import threading
import json
from streamlit_ace import st_ace
from competitive_programming import CompetitiveProgrammingSystem, TEST_CASES, SCORING, validate_and_execute_code, SecureExecutionError
import pandas as pd
import sqlite3
import copy
from moa.agent import MOAgent
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Valid models for Cerebras with their context limits
MODEL_LIMITS = {
    "llama-3.3-70b": 8192,
    "llama3.1-8b": 8192, 
    "llama-4-scout-17b-16e-instruct": 8192,
    "qwen-3-32b": 16384
}

valid_model_names = list(MODEL_LIMITS.keys())

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 0.75 words)"""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def get_model_limit(model_name):
    """Get context limit for a model"""
    return MODEL_LIMITS.get(model_name, 8192)

def calculate_total_context_usage(ai_config, user_prompt):
    """Calculate estimated token usage for the entire MOA execution"""
    # Base user prompt
    user_tokens = estimate_tokens(user_prompt)
    
    # System prompts for agents  
    agent_prompt_tokens = sum(estimate_tokens(agent['prompt']) for agent in ai_config['layer_agents'])
    
    # Main system prompt
    main_prompt_tokens = estimate_tokens(ai_config.get('system_prompt', ''))
    
    # Estimate agent output accumulation over cycles - more realistic calculation
    num_agents = len(ai_config['layer_agents'])
    cycles = ai_config['cycles']
    
    # More realistic estimate based on actual MOA behavior:
    # - Each agent outputs ~300-800 tokens per cycle depending on task complexity
    # - Context grows as: cycle1_outputs + cycle2_outputs + ... 
    # - But there's also MOA system prompt overhead (~500 tokens)
    
    avg_agent_output_per_cycle = 500  # Conservative estimate
    moa_system_overhead = 600  # MOA adds its own system prompts
    
    accumulated_context = moa_system_overhead
    for cycle in range(1, cycles + 1):
        cycle_outputs = avg_agent_output_per_cycle * num_agents
        accumulated_context += cycle_outputs
    
    # Final main model call gets: user + main_prompt + all_accumulated_context
    total_estimated = user_tokens + main_prompt_tokens + accumulated_context
    
    return {
        'user_tokens': user_tokens,
        'agent_prompt_tokens': agent_prompt_tokens, 
        'main_prompt_tokens': main_prompt_tokens,
        'accumulated_context': accumulated_context,
        'total_estimated': total_estimated,
        'breakdown': f"User: {user_tokens} + Main: {main_prompt_tokens} + Accumulated: {accumulated_context} = {total_estimated}"
    }

def check_context_safety(ai_config, user_prompt):
    """Check if configuration is safe from context overflow"""
    if not ai_config.get('layer_agents'):
        return True, "No agents configured"
        
    usage = calculate_total_context_usage(ai_config, user_prompt)
    main_model_limit = get_model_limit(ai_config['main_model'])
    
    # Check main model (gets the full accumulated context)
    # Use 70% safety margin to ensure we have room for completion
    safety_threshold = main_model_limit * 0.7
    
    if usage['total_estimated'] > safety_threshold:
        overflow_amount = usage['total_estimated'] - safety_threshold
        return False, f"Estimated {usage['total_estimated']:,} tokens exceeds safe limit of {safety_threshold:,} tokens (70% of {main_model_limit:,}) by {overflow_amount:,} tokens"
    
    # Check if we have enough room for meaningful completion (at least 512 tokens)
    completion_space = main_model_limit - usage['total_estimated']
    if completion_space < 512:
        return False, f"Only {completion_space} tokens available for completion (need at least 512)"
    
    return True, f"Safe: ~{usage['total_estimated']:,} tokens, {completion_space:,} tokens available for completion"

def suggest_optimizations(ai_config, user_prompt):
    """Suggest optimizations to reduce context usage"""
    suggestions = []
    usage = calculate_total_context_usage(ai_config, user_prompt)
    
    if usage['total_estimated'] > 15000:
        suggestions.append("⚡ **Reduce cycles**: Try 2 cycles instead of 3+ to halve context growth")
    
    if len(ai_config['layer_agents']) > 2:
        suggestions.append("🤖 **Fewer agents**: Consider 2 specialized agents instead of 3+")
    
    if usage['agent_prompt_tokens'] > 1000:
        suggestions.append("📝 **Shorter prompts**: Keep agent prompts under 100 words each")
    
    if usage['user_tokens'] > 2000:
        suggestions.append("✂️ **Shorter input**: Consider reducing the user prompt length")
    
    return suggestions

def init_competition_session():
    """Initialize competition system in session state"""
    if 'competition_system' not in st.session_state:
        st.session_state.competition_system = CompetitiveProgrammingSystem()
    
    # AI agents no longer needed - using grader.py instead
    if 'specialized_agents' not in st.session_state:
        st.session_state.specialized_agents = {}  # Empty dict for backward compatibility
    
    if 'analysis_queue' not in st.session_state:
        st.session_state.analysis_queue = []
    
    # Initialize AI configuration defaults with auto-save
    if 'ai_config' not in st.session_state:
        st.session_state.ai_config = {
            'main_model': 'llama3.1-8b',
            'main_temperature': 0.7,
            'cycles': 1,
            'system_prompt': 'You are a helpful assistant. Please implement the requested function. {helper_response}',
            'layer_agents': [
                {
                    'name': 'assistant',
                    'model': 'llama3.1-8b',
                    'temperature': 0.5,
                    'prompt': 'Help with the coding task. {helper_response}'
                }
            ]
        }
    
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    
    if 'latest_analysis' not in st.session_state:
        st.session_state.latest_analysis = None
    
    # Auto-save tracking
    if 'config_last_saved' not in st.session_state:
        st.session_state.config_last_saved = time.time()

def save_config_to_session():
    """Auto-save configuration to session state"""
    st.session_state.config_last_saved = time.time()

def export_config_as_json():
    """Export current AI configuration as downloadable JSON"""
    config_export = {
        "ai_configuration": st.session_state.ai_config,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.0"
    }
    return json.dumps(config_export, indent=2)

def render_competition_page():
    """Main competition page with AI configuration challenge"""
    init_competition_session()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Import container styling */
    .block-container {
        max-width: 100%;
        padding-top: 2rem;
    }
    
    /* Button improvements - Updated selectors for Streamlit 1.40+ */
    .stButton > button, 
    div[data-testid="stButton"] > button {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #ee5a24, #d63031) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: linear-gradient(90deg, #74b9ff, #0984e3) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: linear-gradient(90deg, #0984e3, #2d3436) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    /* File uploader styling */
    .stFileUploader label,
    div[data-testid="stFileUploader"] label {
        font-weight: 500 !important;
        color: #2d3436 !important;
    }
    
    /* File uploader button */
    .stFileUploader button,
    div[data-testid="stFileUploader"] button {
        background: #f8f9fa !important;
        border: 2px dashed #dee2e6 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Success/error messages */
    .stAlert,
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border-left: 4px solid !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader,
    div[data-testid="stExpander"] > div > div {
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    
    /* Force button width */
    .row-widget.stButton {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    # Check if qwen is the orchestrator for title modification
    thinking_disabled = st.session_state.ai_config.get('main_model') == 'qwen-3-32b'
    thinking_suffix = " (thinking disabled)" if thinking_disabled else ""
    
    st.markdown(f"""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🎮 AI Configuration Challenge{thinking_suffix}</h1>
        <h3>Master Prompt Engineering & Multi-Agent Systems</h3>
        <p>Configure AI agents to generate perfect code • Maximum Score: 120 points • Be the AI whisperer! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs - removed leaderboard
    tab1, tab2, tab3 = st.tabs([
        "🎯 Challenge", "⚙️ Configure AI & Generate", "ℹ️ Scoring Guide"
    ])
    
    with tab1:
        render_challenge_tab()
    
    with tab2:
        render_ai_config_tab()
    
    with tab3:
        render_scoring_guide()

def render_challenge_tab():
    """Render the challenge description"""
    st.markdown("### 🎯 Your Mission: Master AI Configuration to Generate Perfect Code")
    st.markdown("""
    **🎮 This is an AI Configuration Game!** Instead of writing code yourself, you'll become an AI prompt engineer 
    and system architect. Your goal is to configure AI agents (models, prompts, temperatures, cycles) 
    to automatically generate code that scores the maximum 120 points.
    
    **🏆 The Challenge:**
    - 🤖 **Configure AI agents** - Choose models, craft prompts, set temperatures
    - 🔧 **Engineer the system** - Set cycles, agent specialization, and orchestration
    - 📊 **Optimize for scoring** - Use the grader feedback to improve your configuration
    - 🎯 **Achieve perfection** - Get your AI to generate 120/120 point solutions consistently
    
    **🔬 What makes this challenging:**
    - Different models have different strengths
    - Prompt engineering requires precision and creativity  
    - Temperature affects consistency vs creativity
    - Multi-agent coordination needs careful orchestration
    - Context length limits force strategic choices
    """)
    
    # Show the challenge description prominently
    st.markdown("#### 🐛 The Coding Challenge Your AI Must Solve")
    st.markdown("""
    **Function to implement:** `calculate_user_metrics(users, start_date, end_date)`
    
    **What your AI must figure out:**
    - Filter users by date range and activity status
    - Calculate engagement scores with proper formula
    - Handle all edge cases (empty inputs, missing keys, zero division)
    - Return top 5 performers sorted correctly
    - Provide accurate statistics
    
    **🧠 Your AI needs to discover:**
    - The exact engagement formula: `(posts * 2 + comments * 1.5 + likes * 0.1) / days_active`
    - How to handle `days_active = 0` without crashing
    - String date comparison logic for filtering
    - Proper error handling for missing dictionary keys
    - Efficient sorting and top-N selection
    """)
    
    st.info("👆 **The Game:** Go to **'Configure AI & Generate'** to build your AI system, then see the **'Prompt Inspector'** to understand exactly what your models receive!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Optimization Targets")
        st.markdown("""
        **🐛 Critical Bugs (65 pts):**
        - ❌ Division by zero when `days_active = 0`
        - ❌ Missing dictionary keys  
        - ❌ Wrong calculation logic
        - ❌ Incorrect averaging (all vs active users)
        
        **🔧 Logic Issues (28 pts):**
        - ❌ Wrong sorting direction
        - ❌ No active users in date range
        - ❌ Input data mutation
        
        **🎯 Edge Cases (17 pts):**
        - ❌ Less than 5 users available
        - ❌ Invalid date ranges
        - ❌ Malformed data handling
        
        **⚡ Performance (10 pts):**
        - ❌ Inefficient algorithms
        - ❌ Unnecessary operations
        """)
    
    with col2:
        st.markdown("### 🏅 AI Configuration Strategy")
        st.markdown("""
        **🤖 Model Selection:**
        - 🧠 **llama-3.3-70b**: Best overall reasoning
        - ⚡ **llama3.1-8b**: Fast, good for focused tasks
        - 🎯 **llama-4-scout-17b**: Strong at analysis
        - 🔍 **qwen-3-32b**: Good at edge case detection
        
        **🌡️ Temperature Tuning:**
        - **0.0-0.2**: Deterministic, consistent output
        - **0.3-0.5**: Balanced creativity/consistency  
        - **0.6-1.0**: Creative but potentially inconsistent
        
        **🔄 Multi-Agent Tactics:**
        - **Specialization**: Bug finder, edge case handler, optimizer
        - **Progressive refinement**: Multiple cycles for improvement
        - **Context management**: Balance depth vs token limits
        
        **📝 Prompt Engineering:**
        - **Specificity**: Mention exact issues to address
        - **Examples**: Reference the scoring criteria
        - **Structure**: Clear instructions and expectations
        """)
        
        st.markdown("### 📋 Test Cases Preview")
        with st.expander("📝 What Your AI Will Be Tested Against"):
            for i, test_case in enumerate(TEST_CASES[:3]):  # Show first 3
                st.markdown(f"**Test Case {i+1}: {test_case['name']}**")
                st.json(test_case)
                if i < 2:
                    st.markdown("---")
            if len(TEST_CASES) > 3:
                st.markdown(f"*...and {len(TEST_CASES) - 3} more test cases*")

    st.success("""
    🎯 **Success Metrics:**
    - **Grade A (90%+)**: 108+ points - Your AI is a coding master!
    - **Grade B (80%+)**: 96+ points - Excellent AI configuration skills
    - **Grade C (70%+)**: 84+ points - Good progress, keep optimizing
    - **Grade D (60%+)**: 72+ points - Needs work on edge cases
    - **Grade F (<60%)**: <72 points - Major bugs still present
    
    🏆 **Ultimate Goal:** Configure your AI to consistently generate 120/120 point solutions!
    """)

def render_ai_config_tab():
    """Render the AI configuration and code generation interface"""
    st.markdown("### ⚙️ Configure Your AI System")
    st.markdown("Set up the AI agents, models, and prompts to generate the perfect solution!")
    
    # Configuration management section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Download configuration button
        config_json = export_config_as_json()
        st.download_button(
            label="💾 Export Config",
            data=config_json,
            file_name="ai_config.json",
            mime="application/json",
            help="Download your current AI configuration"
        )
    
    with col2:
        # Import configuration
        uploaded_file = st.file_uploader(
            "📤 Upload Config", 
            type=['json'],
            help="Upload a previously saved AI configuration",
            label_visibility="visible"
        )
        
        # Process uploaded file when button is clicked
        if st.button("📤 Import Config", 
                    type="secondary", 
                    use_container_width=True,
                    help="Click to import configuration from JSON file"):
            if uploaded_file is not None:
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    config_data = json.load(uploaded_file)
                    success, message = import_config_from_json(json.dumps(config_data))
                    if success:
                        st.success(f"✅ {message}")
                        # Clear the uploaded file to prevent re-processing
                        st.session_state.pop('uploaded_file', None)
                        st.rerun()
                    else:
                        st.error(f"❌ Import failed: {message}")
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")
            else:
                st.warning("Please upload a JSON file first")
    
    st.divider()
    
    # Main Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Ensure main_temperature is a float
        current_temp = st.session_state.ai_config.get('main_temperature', 0.1)
        if isinstance(current_temp, list):
            current_temp = 0.1  # Reset to default if it's a list
        elif not isinstance(current_temp, (int, float)):
            current_temp = float(current_temp) if str(current_temp).replace('.', '').isdigit() else 0.1
        current_temp = max(0.0, min(1.0, float(current_temp)))
        
        main_temp = st.slider(
            "Main Temperature",
            min_value=0.0,
            max_value=1.0,
            value=current_temp,
            step=0.01,
            help="Lower = more focused, Higher = more creative",
            key="main_temp"
        )
        if main_temp != st.session_state.ai_config['main_temperature']:
            st.session_state.ai_config['main_temperature'] = main_temp
            save_config_to_session()
    
    with col2:
        # Ensure main_model is a string and in valid list
        current_model = st.session_state.ai_config.get('main_model', 'llama-3.3-70b')
        if isinstance(current_model, list):
            current_model = 'llama-3.3-70b'  # Reset to default
        elif current_model not in valid_model_names:
            current_model = 'llama-3.3-70b'  # Reset to default
            
        model = st.selectbox(
            "Main Model",
            options=valid_model_names,
            index=valid_model_names.index(current_model),
            help="Choose the primary AI model",
            key="main_model"
        )
        if model != st.session_state.ai_config['main_model']:
            st.session_state.ai_config['main_model'] = model
            save_config_to_session()
            st.rerun()  # Force page refresh to update title
    
    with col3:
        # Ensure cycles is an integer
        current_cycles = st.session_state.ai_config.get('cycles', 3)
        if isinstance(current_cycles, list):
            current_cycles = 3  # Reset to default
        elif not isinstance(current_cycles, int):
            current_cycles = int(current_cycles) if str(current_cycles).isdigit() else 3
        current_cycles = max(1, min(10, current_cycles))
        
        cycles = st.slider(
            "Cycles",
            min_value=1,
            max_value=10,
            value=current_cycles,
            help="Number of refinement cycles",
            key="cycles"
        )
        if cycles != st.session_state.ai_config['cycles']:
            st.session_state.ai_config['cycles'] = cycles
            save_config_to_session()
    
    # Main System Prompt Editor
    st.subheader("🎯 Main System Prompt")
    st.markdown("**This prompt is sent to your main model (final code generator). Optimize it for best results!**")
    
    main_system_prompt = st.text_area(
        "Main System Prompt Template",
        value=st.session_state.ai_config.get('system_prompt', ''),
        height=120,
        help="Use {helper_response} to include outputs from layer agents. This is your main instruction to the final code generator.",
        key="main_system_prompt",
        placeholder="You are an expert Python programmer. Fix all bugs in the given function while maintaining its original structure and purpose..."
    )
    if main_system_prompt != st.session_state.ai_config.get('system_prompt', ''):
        st.session_state.ai_config['system_prompt'] = main_system_prompt
        save_config_to_session()
    
    # Quick preset buttons for system prompt
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Bug-Focused Preset", 
                    type="secondary",
                    use_container_width=True,
                    help="Optimize for finding and fixing bugs"):
            preset_prompt = "You are an expert Python programmer specialized in bug detection and fixing. Analyze the requirements carefully and implement a robust solution. Focus on: 1) Division by zero errors, 2) Missing key handling, 3) Wrong calculations, 4) Sorting issues, 5) Edge cases. Return ONLY the function code without explanations or thinking process. {helper_response}"
            st.session_state.ai_config['system_prompt'] = preset_prompt
            save_config_to_session()
            st.rerun()
    
    with col2:
        if st.button("🚀 Performance Preset", 
                    type="secondary",
                    use_container_width=True,
                    help="Optimize for performance and efficiency"):
            preset_prompt = "You are a performance-focused Python expert. Create efficient, optimized code that handles all edge cases. Prioritize: 1) Algorithmic efficiency, 2) Memory optimization, 3) Robust error handling, 4) Clean, maintainable code. Return ONLY the function code without explanations or thinking process. {helper_response}"
            st.session_state.ai_config['system_prompt'] = preset_prompt
            save_config_to_session()
            st.rerun()
    
    with col3:
        if st.button("🧠 Comprehensive Preset", 
                    type="secondary",
                    use_container_width=True,
                    help="Balanced approach for overall quality"):
            preset_prompt = "You are an expert Python programmer. Create a comprehensive solution that excels in all areas: bug-free implementation, edge case handling, performance optimization, and code quality. Synthesize the analysis from helper agents to produce the perfect solution. Return ONLY the function code without explanations or thinking process. {helper_response}"
            st.session_state.ai_config['system_prompt'] = preset_prompt
            save_config_to_session()
            st.rerun()
    
    # Layer Agents section
    st.subheader("🤖 Layer Agents")
    
    # Add new agent button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Agent", 
                    key="add_agent", 
                    type="secondary",
                    use_container_width=True,
                    help="Add a new layer agent to your configuration"):
            new_agent = {
                'name': f'Agent {len(st.session_state.ai_config["layer_agents"]) + 1}',
                'model': 'llama-3.3-70b',
                'temperature': 0.1,
                'prompt': 'Analyze and improve the code. {helper_response}'
            }
            st.session_state.ai_config['layer_agents'].append(new_agent)
            save_config_to_session()
            st.rerun()
    
    with col2:
        if len(st.session_state.ai_config['layer_agents']) > 0:
            st.info(f"💡 **{len(st.session_state.ai_config['layer_agents'])} agents configured** • Edit each agent below")
        else:
            st.info("""
            🚀 **Get Started:**
            1. Click **"Add Agent"** to create your first AI agent
            2. Configure its model, temperature, and specialized prompt
            3. Add more agents with different specializations  
            4. Test your configuration with code generation!
            
            💡 **Pro Tip:** Start with a Bug Hunter, then add an Edge Case Expert and Performance Optimizer
            """)
    
    # Individual agent editors
    if len(st.session_state.ai_config['layer_agents']) > 0:
        st.markdown("---")
        
        for i, agent in enumerate(st.session_state.ai_config['layer_agents']):
            # Agent expander with summary info
            agent_summary = f"{agent['name']} • {agent['model']} • temp={agent['temperature']:.2f}"
            
            with st.expander(f"🤖 {agent_summary}", expanded=(i == 0)):  # First agent expanded by default
                
                # Agent configuration within the expander
                agent_col1, agent_col2 = st.columns(2)
                
                with agent_col1:
                    # Agent name
                    agent_name = st.text_input(
                        "Agent Name",
                        value=agent['name'],
                        key=f"agent_name_{i}",
                        help="Give your agent a descriptive name"
                    )
                    if agent_name != agent['name']:
                        agent['name'] = agent_name
                        save_config_to_session()
                    
                    # Model selection
                    agent_model = agent.get('model', 'llama-3.3-70b')
                    if agent_model not in valid_model_names:
                        agent_model = 'llama-3.3-70b'
                        
                    model = st.selectbox(
                        "Model",
                        options=valid_model_names,
                        index=valid_model_names.index(agent_model),
                        key=f"agent_model_{i}",
                        help="Choose the AI model for this agent"
                    )
                    if model != agent['model']:
                        agent['model'] = model
                        save_config_to_session()
                
                with agent_col2:
                    # Temperature slider
                    agent_temp = agent.get('temperature', 0.1)
                    if isinstance(agent_temp, list):
                        agent_temp = 0.1
                    elif not isinstance(agent_temp, (int, float)):
                        agent_temp = float(agent_temp) if str(agent_temp).replace('.', '').isdigit() else 0.1
                    agent_temp = max(0.0, min(1.0, float(agent_temp)))
                    
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=agent_temp,
                        step=0.01,
                        key=f"agent_temp_{i}",
                        help="Lower = more consistent, Higher = more creative"
                    )
                    if temperature != agent['temperature']:
                        agent['temperature'] = temperature
                        save_config_to_session()
                
                # Prompt configuration
                prompt = st.text_area(
                    "System Prompt Template", 
                    value=agent.get('prompt', 'Analyze and improve the code. {helper_response}'),
                    height=100,
                    help="Use {helper_response} to include outputs from previous agents/cycles",
                    key=f"agent_prompt_{i}"
                )
                if prompt != agent['prompt']:
                    agent['prompt'] = prompt
                    save_config_to_session()
                
                # Agent actions
                st.markdown("---")
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button("📋 Duplicate", key=f"duplicate_{i}", help="Create a copy of this agent"):
                        agents = st.session_state.ai_config['layer_agents']
                        duplicate = agents[i].copy()
                        duplicate['name'] = f"{duplicate['name']} Copy"
                        agents.insert(i + 1, duplicate)
                        save_config_to_session()
                        st.rerun()
                
                with action_col2:
                    if st.button("🗑️ Remove", key=f"remove_{i}", type="secondary", help="Delete this agent"):
                        st.session_state.ai_config['layer_agents'].pop(i)
                        save_config_to_session()
                        st.rerun()
                
                # Show what this agent receives
                st.markdown("---")
                show_preview = st.checkbox("🔍 Show Agent Input Preview", key=f"show_preview_{i}", help="Preview what this agent will receive")
                
                if show_preview:
                    cycle1_prompt = agent['prompt'].format(helper_response="")
                    st.markdown("**Cycle 1 (no previous context):**")
                    st.code(f"System: {cycle1_prompt}\nUser: [Your input prompt]", language='text')
                    
                    cycle2_prompt = agent['prompt'].format(helper_response="[Previous cycle outputs would be here...]")
                    st.markdown("**Cycle 2+ (with context):**")
                    st.code(f"System: {cycle2_prompt}\nUser: [Your input prompt]", language='text')
        
    else:
        # No agents configured
        st.info("""
        🚀 **Get Started:**
        1. Click **"Add Agent"** to create your first AI agent
        2. Configure its model, temperature, and specialized prompt
        3. Add more agents with different specializations  
        4. Test your configuration with code generation!
        
        💡 **Pro Tip:** Start with a Bug Hunter, then add an Edge Case Expert and Performance Optimizer
        """)

    st.divider()
    
    # Code Generation Section
    st.subheader("🚀 Generate Code")
    
    # Custom prompt input
    st.markdown("#### 📝 Custom Prompt Input")
    st.markdown("**Enter your own prompt to test your AI configuration, or use the default coding challenge:**")
    
    # Default coding challenge prompt
    default_prompt = """[Generation ID: {generation_id}]

You are tasked with implementing a Python function called `calculate_user_metrics` that analyzes user engagement data.

FUNCTION REQUIREMENTS:
- Function name: `calculate_user_metrics(users, start_date, end_date)`
- Purpose: Calculate engagement metrics for active users within a date range
- Input: list of user dictionaries, start_date string, end_date string
- Output: dictionary with 'average_engagement', 'top_performers', 'active_count'

KEY IMPLEMENTATION DETAILS:
1. Handle division by zero when days_active is 0
2. Validate that required dictionary keys exist before accessing them
3. Calculate average based on active users, not all users
4. Sort top performers correctly (highest engagement first)
5. Handle empty input gracefully
6. Add proper input validation for all parameters

ENGAGEMENT CALCULATION:
- engagement_score = (posts * 2 + comments * 1.5 + likes * 0.1) / days_active
- Only include users whose last_login is within the date range
- Required user keys: 'last_login', 'posts', 'comments', 'likes', 'days_active'

IMPORTANT CONSTRAINTS:
- Do NOT import datetime or any other modules  
- Work with string dates as-is (simple string comparison works for "YYYY-MM-DD" format)
- Do NOT use datetime.strptime() - it's not available
- Return ONLY the function code, no explanations or markdown
- Handle all edge cases properly

Return the complete function implementation."""

    # Custom prompt text area
    custom_prompt = st.text_area(
        "Your Prompt to AI Models",
        value=st.session_state.get('custom_prompt', default_prompt),
        height=200,
        help="This exact text will be sent to your AI models. Test different prompts to see how your configuration responds!",
        key="custom_prompt_input"
    )
    
    # Store in session state
    if custom_prompt != st.session_state.get('custom_prompt', ''):
        st.session_state.custom_prompt = custom_prompt
    
    # Quick preset buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎯 Default Challenge", help="Use the original coding challenge"):
            st.session_state.custom_prompt = default_prompt
            st.rerun()
    
    with col2:
        if st.button("🔢 Simple Test", help="Test with a simple request"):
            st.session_state.custom_prompt = "Return nothing but the number 5"
            st.rerun()
    
    with col3:
        if st.button("📝 Hello World", help="Test with basic code generation"):
            st.session_state.custom_prompt = "Write a Python function that prints 'Hello World'"
            st.rerun()
    
    with col4:
        if st.button("🧮 Math Test", help="Test with mathematical problem"):
            st.session_state.custom_prompt = "Write a Python function that calculates the factorial of a number"
            st.rerun()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **Test your configuration:** Try different prompts to see how your AI agents respond to various challenges!")
    
    with col2:
        # Check context safety before showing generate button
        user_prompt = st.session_state.get('custom_prompt', default_prompt)
        is_safe, safety_message = check_context_safety(st.session_state.ai_config, user_prompt)
        usage = calculate_total_context_usage(st.session_state.ai_config, user_prompt)
        
        # Show context usage info
        if len(st.session_state.ai_config['layer_agents']) > 0:
            # Context usage metric above the button
            main_limit = get_model_limit(st.session_state.ai_config['main_model'])
            usage_pct = (usage['total_estimated'] / main_limit) * 100
            
            if usage_pct > 80:
                st.error(f"🚨 Context: {usage['total_estimated']:,} / {main_limit:,} tokens ({usage_pct:.0f}%)")
            elif usage_pct > 60:
                st.warning(f"⚠️ Context: {usage['total_estimated']:,} / {main_limit:,} tokens ({usage_pct:.0f}%)")
            else:
                st.success(f"✅ Context: {usage['total_estimated']:,} / {main_limit:,} tokens ({usage_pct:.0f}%)")
        
        if st.button("⚡ Generate Code", type="primary", use_container_width=True, disabled=not is_safe):
            if not is_safe:
                st.error(f"❌ {safety_message}")
                suggestions = suggest_optimizations(st.session_state.ai_config, user_prompt)
                if suggestions:
                    st.markdown("**💡 Try these optimizations:**")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
            else:
                # Use the custom prompt from session state
                with st.spinner("🤖 AI system generating solution..."):
                    try:
                        generated_code = generate_code_with_ai(st.session_state.ai_config, user_prompt)
                        if generated_code:
                            st.session_state.generated_code = generated_code
                            st.session_state.last_used_prompt = user_prompt  # Store the prompt used
                            st.success("✅ Code generated successfully!")
                        else:
                            st.error("❌ Code generation failed. Please check your configuration.")
                    except Exception as e:
                        st.error(f"❌ Error generating code: {str(e)}")
    
    # Context overflow warning section
    if len(st.session_state.ai_config['layer_agents']) > 0:
        if not is_safe:
            st.markdown("---")
            st.error("🚨 **Context Overflow Risk Detected**")
            st.markdown(f"**Issue:** {safety_message}")
            
            with st.expander("📊 Detailed Token Usage Breakdown", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("User Prompt", f"{usage['user_tokens']:,} tokens")
                    st.metric("Agent Prompts", f"{usage['agent_prompt_tokens']:,} tokens") 
                    st.metric("Main Prompt", f"{usage['main_prompt_tokens']:,} tokens")
                
                with col2:
                    st.metric("Accumulated Context", f"{usage['accumulated_context']:,} tokens")
                    st.metric("**Total Estimated**", f"**{usage['total_estimated']:,} tokens**")
                    main_limit = get_model_limit(st.session_state.ai_config['main_model'])
                    st.metric("Main Model Limit", f"{main_limit:,} tokens")
                
                st.markdown("**💡 Quick Fixes:**")
                suggestions = suggest_optimizations(st.session_state.ai_config, user_prompt)
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
                
                # Quick fix buttons
                st.markdown("**⚡ One-Click Fixes:**")
                fix_col1, fix_col2 = st.columns(2)
                
                with fix_col1:
                    if st.session_state.ai_config['cycles'] > 2:
                        if st.button("⚡ Reduce to 2 Cycles", help="Reduce cycles to 2 to halve context growth"):
                            st.session_state.ai_config['cycles'] = 2
                            save_config_to_session()
                            st.rerun()
                
                with fix_col2:
                    if len(st.session_state.ai_config['layer_agents']) > 2:
                        if st.button("🤖 Remove Last Agent", help="Remove the last agent to reduce context usage"):
                            st.session_state.ai_config['layer_agents'].pop()
                            save_config_to_session()
                            st.rerun()
            
            st.info("🎯 **Fix the warnings above to enable code generation**")
        
        elif usage['total_estimated'] > 10000:  # Show optimization tips even when safe
            with st.expander("⚡ Performance Optimization Tips", expanded=False):
                st.markdown(f"**Current estimated usage:** {usage['total_estimated']:,} tokens")
                st.markdown("**Your configuration is safe, but here are some optimization tips:**")
                suggestions = suggest_optimizations(st.session_state.ai_config, user_prompt)
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")

    # NEW SECTION: Show exactly what gets sent to the models
    st.divider()
    st.subheader("🔍 Prompt Inspector - See What Your Models Receive")
    st.markdown("**🎯 Game Objective:** Optimize these prompts, models, and temperatures to generate perfect code!")
    
    with st.expander("📋 Exact Model Inputs & Prompts", expanded=False):
        # Show the user input (code generation task)
        st.markdown("#### 📝 User Input (The Code Generation Task)")
        st.markdown("*This is the same for all agents and cycles:*")
        
        generation_id = "preview-12345"  # Preview ID
        generation_prompt = f"""[Generation ID: {generation_id}]

You are tasked with implementing a Python function called `calculate_user_metrics` that analyzes user engagement data.

FUNCTION REQUIREMENTS:
- Function name: `calculate_user_metrics(users, start_date, end_date)`
- Purpose: Calculate engagement metrics for active users within a date range
- Input: list of user dictionaries, start_date string, end_date string
- Output: dictionary with 'average_engagement', 'top_performers', 'active_count'

KEY IMPLEMENTATION DETAILS:
1. Handle division by zero when days_active is 0
2. Validate that required dictionary keys exist before accessing them
3. Calculate average based on active users, not all users
4. Sort top performers correctly (highest engagement first)
5. Handle empty input gracefully
6. Add proper input validation for all parameters

ENGAGEMENT CALCULATION:
- engagement_score = (posts * 2 + comments * 1.5 + likes * 0.1) / days_active
- Only include users whose last_login is within the date range
- Required user keys: 'last_login', 'posts', 'comments', 'likes', 'days_active'

IMPORTANT CONSTRAINTS:
- Do NOT import datetime or any other modules  
- Work with string dates as-is (simple string comparison works for "YYYY-MM-DD" format)
- Do NOT use datetime.strptime() - it's not available
- Return ONLY the function code, no explanations or markdown
- Handle all edge cases properly

Return the complete function implementation."""
        
        st.code(generation_prompt, language='text')
        
        # Token count estimation
        estimated_tokens = len(generation_prompt.split()) * 1.3  # Rough estimate
        st.info(f"📊 **Estimated tokens:** ~{estimated_tokens:.0f} tokens")
        
        st.divider()
        
        # Show main system prompt
        st.markdown("#### 🎯 Main Model System Prompt")
        st.markdown("*This gets sent to your main model (final code generator):*")
        main_prompt = st.session_state.ai_config.get('system_prompt', '')
        st.code(main_prompt, language='text')
        
        # Show how it gets formatted with helper_response
        st.markdown("#### 🔄 Main Prompt With Helper Response")
        st.markdown("*After cycles complete, {helper_response} gets filled with agent outputs:*")
        sample_helper = """You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. 
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
0. [Agent 1 output would be here]
1. [Agent 2 output would be here]  
2. [Agent 3 output would be here]"""
        
        formatted_main = main_prompt.format(helper_response=sample_helper)
        st.code(formatted_main, language='text')
        
        st.divider()
        
        # Show layer agent prompts
        st.markdown("#### 🤖 Layer Agent System Prompts")
        st.markdown("*Each agent gets their specific prompt + user input:*")
        
        for i, agent in enumerate(st.session_state.ai_config['layer_agents']):
            st.markdown(f"**🤖 {agent['name']} ({agent['model']}, temp={agent['temperature']})**")
            
            # Show Cycle 1 (no helper_response)
            cycle1_prompt = agent['prompt'].format(helper_response="")
            st.code(f"Cycle 1: {cycle1_prompt}", language='text')
            
            # Show what it looks like with helper_response
            cycle2_prompt = agent['prompt'].format(helper_response="[Previous cycle outputs would be here...]")
            st.code(f"Cycle 2+: {cycle2_prompt}", language='text')
            
            if i < len(st.session_state.ai_config['layer_agents']) - 1:
                st.markdown("---")
        
        st.divider()
        
        # Token limit warning
        st.markdown("#### ⚠️ Context Length Management")
        
        # Calculate actual usage for current config
        current_usage = calculate_total_context_usage(st.session_state.ai_config, generation_prompt)
        main_limit = get_model_limit(st.session_state.ai_config['main_model'])
        usage_pct = (current_usage['total_estimated'] / main_limit) * 100
        
        if usage_pct > 80:
            warning_color = "error"
            warning_icon = "🚨"
        elif usage_pct > 60:
            warning_color = "warning" 
            warning_icon = "⚠️"
        else:
            warning_color = "info"
            warning_icon = "✅"
            
        getattr(st, warning_color)(f"""
        **{warning_icon} Current Configuration Analysis:**
        - **Estimated Usage:** {current_usage['total_estimated']:,} tokens ({usage_pct:.0f}% of {main_limit:,} limit)
        - **Model Limits:** qwen-3-32b (16k), llama models (8k)
        - **Breakdown:** {current_usage['breakdown']}
        
        **Context Growth Pattern:**
        - User input: ~{current_usage['user_tokens']:,} tokens (fixed)
        - System prompts: ~{current_usage['agent_prompt_tokens'] + current_usage['main_prompt_tokens']:,} tokens (fixed)
        - **Agent outputs accumulate:** Cycle 1 → Cycle 2 → Cycle 3 (exponential growth!)
        - Final main model gets ALL accumulated context: ~{current_usage['accumulated_context']:,} tokens
        
        **💡 Optimization Tips:**
        - **Switch to qwen-3-32b** as main model for 2x context capacity
        - **Reduce cycles** (2 instead of 3) to halve context accumulation
        - **Fewer agents** (2 instead of 3+) to reduce per-cycle growth
        - **Shorter prompts** to minimize baseline usage
        """)
        
        # Show optimization suggestions if needed
        if usage_pct > 60:
            suggestions = suggest_optimizations(st.session_state.ai_config, generation_prompt)
            if suggestions:
                st.markdown("**🔧 Recommended Fixes:**")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
        
        # Configuration summary
        st.markdown("#### 📊 Current Configuration Summary")
        config_summary = f"""
**Main Model:** {st.session_state.ai_config['main_model']} (temp: {st.session_state.ai_config['main_temperature']})
**Cycles:** {st.session_state.ai_config['cycles']}
**Layer Agents:** {len(st.session_state.ai_config['layer_agents'])}

**Execution Flow:**
1. Cycle 1: All {len(st.session_state.ai_config['layer_agents'])} agents run with just user input
2. Combine outputs → helper_response
3. Cycle 2: All agents run with previous outputs as context
4. Repeat for {st.session_state.ai_config['cycles']} cycles
5. Main model generates final code with ALL accumulated context
        """
        st.code(config_summary, language='text')
        
        st.success("""
        🎯 **Game Strategy:** 
        - Experiment with different prompt styles (detailed vs concise)
        - Try different model combinations for different strengths
        - Adjust temperatures: Lower for consistency, higher for creativity
        - Balance cycles vs context length limits
        - Specialize agents: one for bugs, one for edge cases, one for optimization
        """)
    
    # Display generated code
    if hasattr(st.session_state, 'generated_code') and st.session_state.generated_code:
        st.subheader("📝 Generated Response")
        
        # Show what prompt was used
        if hasattr(st.session_state, 'last_used_prompt'):
            with st.expander("📋 Prompt Used for Generation", expanded=False):
                st.code(st.session_state.last_used_prompt, language='text')
        
        # Display the generated output
        if "def " in st.session_state.generated_code:
            st.markdown("**🐍 Generated Python Code:**")
            st.code(st.session_state.generated_code, language='python')
        else:
            st.markdown("**🎯 Generated Response:**")
            st.code(st.session_state.generated_code, language='text')
        
        # Submit button (only show for the coding challenge)
        if hasattr(st.session_state, 'last_used_prompt') and "calculate_user_metrics" in st.session_state.last_used_prompt:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("🎯 Ready to submit? Your solution will be analyzed and scored automatically.")
            
            with col2:
                if st.button("🚀 Submit Solution", type="primary", use_container_width=True):
                    with st.spinner("📊 Analyzing your solution..."):
                        try:
                            submit_generated_solution("AI_Solution", st.session_state.generated_code)
                        except Exception as e:
                            st.error(f"❌ Submission failed: {str(e)}")
        else:
            st.info("💡 **This is a test response.** Use the 'Default Challenge' button to generate code that can be submitted for scoring.")
    
    # Display results if available
    if hasattr(st.session_state, 'latest_analysis') and st.session_state.latest_analysis:
        display_analysis_results(st.session_state.latest_analysis)

def generate_code_with_ai(ai_config, user_prompt):
    """Generate code using the configured AI system"""
    try:
        import random
        import uuid
        
        # Generate a unique ID for this generation
        generation_id = str(uuid.uuid4())[:8]
        
        # Check context safety first
        is_safe, safety_message = check_context_safety(ai_config, user_prompt)
        if not is_safe:
            st.error(f"❌ Context overflow prevented: {safety_message}")
            return None
        
        # Calculate dynamic max_tokens based on context usage and model limits
        usage = calculate_total_context_usage(ai_config, user_prompt)
        main_model_limit = get_model_limit(ai_config['main_model'])
        
        # Reserve tokens for context, leave remainder for generation
        # Use very conservative estimate: reserve much more than estimated usage
        context_safety_buffer = int(usage['total_estimated'] * 1.5)  # 50% safety buffer
        available_for_generation = main_model_limit - context_safety_buffer
        
        # Set completion limits very conservatively
        # Give qwen-3-32b more room since it's a reasoning model that needs space to think
        if ai_config['main_model'] == 'qwen-3-32b':
            max_generation_tokens = max(256, min(1024, available_for_generation))  # More space for qwen reasoning
        else:
            max_generation_tokens = max(128, min(512, available_for_generation))  # Conservative for llama models
        
        # Emergency fallback: if still too high, force very small completion
        if context_safety_buffer + max_generation_tokens > main_model_limit:
            max_generation_tokens = max(64, main_model_limit - usage['total_estimated'] - 200)  # Emergency 200 token buffer
        
        print(f"🎲 Generation ID: {generation_id}")
        print(f"📊 Context usage: {usage['total_estimated']} tokens (estimated)")
        print(f"🎯 Model limit: {main_model_limit} tokens")
        print(f"💾 Context buffer: {context_safety_buffer} tokens (150% of estimated)")
        print(f"✏️ Max completion tokens: {max_generation_tokens} tokens")
        
        # Validate that we're within bounds
        total_budget = usage['total_estimated'] + max_generation_tokens
        if total_budget > main_model_limit:
            print(f"⚠️ Still over budget: {total_budget} > {main_model_limit}, forcing minimal completion")
            max_generation_tokens = 64  # Force very small completion as last resort
        
        # Add some randomization to the temperature
        base_temp = ai_config['main_temperature']
        randomized_temp = max(0.0, min(1.0, base_temp + random.uniform(-0.05, 0.05)))
        
        print(f"🌡️ Randomized temperature: {randomized_temp}")
        
        # Create MOA agent with user configuration and dynamic limits
        layer_agent_config = {}
        for agent in ai_config['layer_agents']:
            # Calculate max_tokens for each agent based on their model - be very conservative
            agent_limit = get_model_limit(agent['model'])
            agent_context_estimate = usage['user_tokens'] + estimate_tokens(agent['prompt']) + (usage['accumulated_context'] // 4)
            agent_context_buffer = int(agent_context_estimate * 1.5)  # 50% buffer
            agent_max_tokens = max(128, min(512, agent_limit - agent_context_buffer))  # Very conservative limits
            
            layer_agent_config[agent['name']] = {
                'system_prompt': agent['prompt'],
                'model_name': agent['model'],
                'temperature': agent['temperature'],
                'max_tokens': agent_max_tokens  # Conservative limit per agent
            }
            
            print(f"🤖 Agent {agent['name']} ({agent['model']}): max_tokens={agent_max_tokens}")
        
        # Use the MOAgent.from_config properly with dynamic limits
        moa_agent = MOAgent.from_config(
            main_model=ai_config['main_model'],
            system_prompt=ai_config['system_prompt'],
            cycles=ai_config['cycles'],
            temperature=randomized_temp,  # Use randomized temperature
            max_tokens=max_generation_tokens,  # Conservative limit for main model
            layer_agent_config=layer_agent_config if layer_agent_config else None
        )
        
        # Use the user's custom prompt, with generation ID if it contains the placeholder
        if "{generation_id}" in user_prompt:
            final_prompt = user_prompt.format(generation_id=generation_id)
        else:
            final_prompt = user_prompt
        
        # If qwen-3-32b is the main model, append /no_think to disable reasoning
        if ai_config['main_model'] == 'qwen-3-32b':
            final_prompt = final_prompt + "\n\n/no_think"
        
        print(f"🔮 Generating with {ai_config['main_model']} (temp: {randomized_temp}, max_tokens: {max_generation_tokens})")
        print(f"📝 User prompt: {final_prompt[:100]}...")
        
        # Generate the response using the user's actual prompt
        response_chunks = moa_agent.chat(final_prompt)
        
        # Collect response
        full_response = ""
        for chunk in response_chunks:
            if isinstance(chunk, dict):
                # Handle dictionary chunks
                if chunk.get('response_type') == 'final':
                    full_response += chunk.get('delta', '')
                elif 'delta' in chunk:
                    full_response += chunk.get('delta', '')
            else:
                # Handle string chunks
                full_response += str(chunk)
                print(f"📝 Code chunk: {str(chunk)[:50]}...")
        
        # For the coding challenge, extract function code, otherwise return raw response
        if "calculate_user_metrics" in user_prompt or "function" in user_prompt.lower():
            generated_code = extract_function_code(full_response)
        else:
            # For simple prompts, return the raw response
            generated_code = full_response.strip()
        
        if generated_code:
            print(f"✅ Generated {len(generated_code)} characters")
            print(f"🔍 Preview: {generated_code[:200]}...")
        else:
            print("❌ No output generated")
            print(f"📄 Full response: {full_response[:500]}...")
            
        return generated_code
        
    except Exception as e:
        print(f"❌ Error generating code: {str(e)}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        st.error(f"Failed to generate code: {str(e)}")
        return None

def extract_function_code(response):
    """Extract the function code from AI response"""
    import re
    
    # If response is empty, return None
    if not response or not response.strip():
        return None
    
    # Remove any markdown code blocks
    code_pattern = r'```(?:python)?\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        # Take the first code block
        code = matches[0].strip()
    else:
        # If no code blocks, try to find function definition
        lines = response.split('\n')
        code_lines = []
        in_function = False
        function_indent = 0
        
        for line in lines:
            stripped_line = line.strip()
            
            # Start of function
            if stripped_line.startswith('def calculate_user_metrics'):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                code_lines.append(line)
            elif in_function:
                # Check if we're still in the function
                if line.strip() == '':
                    # Empty line, keep it
                    code_lines.append(line)
                elif len(line) - len(line.lstrip()) > function_indent or line.startswith(' ') or line.startswith('\t'):
                    # Indented line, still in function
                    code_lines.append(line)
                elif line.strip().startswith('#'):
                    # Comment line, keep it
                    code_lines.append(line)
                else:
                    # Non-indented, non-empty line that's not a comment - end of function
                    break
        
        code = '\n'.join(code_lines).strip()
    
    # Final validation - make sure we have a function
    if code and 'def calculate_user_metrics' in code:
        return code
    else:
        # Last resort - return the original response cleaned up
        return response.strip()

def submit_generated_solution(config_name: str, generated_code: str):
    """Submit the AI-generated solution for grading using the deterministic grader"""
    try:
        # Submit to the existing grading system
        result = st.session_state.competition_system.submit_solution(
            student_name=config_name,
            code=generated_code
        )
        
        if result.get('submission_id'):
            submission_id = result['submission_id']
            
            # Get analysis results using the grader (no AI agents needed)
            analysis_result = st.session_state.competition_system.analyze_submission(
                submission_id
            )
            
            st.session_state.latest_analysis = analysis_result
            
            if "error" not in analysis_result:
                total_score = analysis_result['total_score']
                max_score = analysis_result.get('max_score', 120)
                st.success(f"🎉 Solution scored: **{total_score}/{max_score} points**!")
            else:
                st.error("Analysis failed. Please try again.")
        else:
            st.error(f"Submission failed: {result.get('error', 'Unknown error')}")
                    
    except Exception as e:
        st.error(f"Error submitting solution: {str(e)}")

def display_analysis_results(analysis_result):
    """Display detailed analysis results from the grader"""
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Check if validation failed
    if analysis_result.get('validation_failed', False):
        st.error("❌ Code Validation Failed - Grading Skipped")
        
        # Display validation details
        validation_results = analysis_result.get('validation_results', {})
        
        with st.expander("🧪 Validation Details", expanded=True):
            st.markdown("#### 📝 Code Validation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                syntax_status = "✅ Pass" if validation_results.get('syntax_valid', False) else "❌ Fail"
                st.metric("Syntax Check", syntax_status)
                
                tests_passed = validation_results.get('passed_tests', 0)
                total_tests = validation_results.get('total_tests', 0)
                st.metric("Unit Tests", f"{tests_passed}/{total_tests}")
            
            with col2:
                overall_status = "✅ Pass" if validation_results.get('passes_tests', False) else "❌ Fail"
                st.metric("Overall Status", overall_status)
            
            # Show specific test results
            if validation_results.get('test_results'):
                st.markdown("#### 🔍 Test Case Results")
                for test in validation_results['test_results']:
                    status_icon = "✅" if test['passed'] else "❌"
                    st.write(f"{status_icon} **{test['test_name']}**")
                    if test['error']:
                        st.error(f"Error: {test['error']}")
                        if "_strptime" in str(test['error']):
                            st.info("""
                            💡 **Date Parsing Fix**: Your AI configuration generated code that uses restricted datetime functions.
                            Try updating your prompts to specify: "Use string comparison for dates (start_date <= date <= end_date) 
                            since dates are in YYYY-MM-DD format. Do not use datetime.strptime."
                            """)
                    elif test['output']:
                        with st.expander(f"Output for {test['test_name']}", expanded=False):
                            st.json(test['output'])
            
            # Show runtime errors if any
            if validation_results.get('runtime_errors'):
                st.markdown("#### ❌ Runtime Errors")
                for error in validation_results['runtime_errors']:
                    st.error(error)
        
        st.info("💡 **Adjust your AI configuration and regenerate to fix these issues!**")
        return
    
    # Display validation results if analysis succeeded
    validation_results = analysis_result.get('validation_results', {})
    if validation_results:
        with st.expander("🧪 Code Validation Results", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Syntax Check", "Pass")
            with col2:
                tests_passed = validation_results.get('passed_tests', 0)
                total_tests = validation_results.get('total_tests', 0)
                st.metric("🧪 Unit Tests", f"{tests_passed}/{total_tests}")
            with col3:
                st.metric("🎯 Validation", "✅ Pass")
            
            # Show grader processing info
            st.success("🎯 **Grader Analysis**: Comments automatically removed, code analyzed by deterministic grader")
    
    # Score summary using the grader results
    st.markdown("#### 🎯 Score Breakdown")
    
    # Get detailed scores from grader results
    detailed_scores = analysis_result.get('detailed_scores', {})
    
    # Create wider columns for better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        critical_bugs = detailed_scores.get('critical_bugs', 0)
        st.metric("🐛 Critical Bugs Fixed", f"{critical_bugs}/65", help="Division by zero, missing keys, wrong calculations")
        
        performance = detailed_scores.get('performance', 0)
        st.metric("⚡ Performance", f"{performance}/10", help="Efficient algorithms and optimizations")
    
    with col2:
        logic_issues = detailed_scores.get('logic_issues', 0)
        st.metric("🔧 Logic Issues", f"{logic_issues}/28", help="Sorting, no active users, input mutation")
        
        edge_cases = detailed_scores.get('edge_cases', 0)
        st.metric("🎯 Edge Cases", f"{edge_cases}/17", help="Handles fewer than 5 users, invalid dates, robust errors")
    
    # Total score - full width and prominent
    total_score = analysis_result['total_score']
    max_score = analysis_result.get('max_score', 135)
    percentage = round((total_score / max_score) * 100, 1)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;'>
        <h1>🏆 Total Score: {total_score}/{max_score} ({percentage}%)</h1>
        <p style='font-size: 1.2rem; margin: 0;'>Configuration: {analysis_result.get('student_name', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed grader analysis
    st.markdown("#### 🔍 Detailed Grader Analysis")
    
    grading_results = analysis_result.get('grading_results', {})
    if grading_results:
        with st.expander("📋 Complete Grader Report", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tests Passed", grading_results.get('tests_passed', 0))
                st.metric("Tests Failed", grading_results.get('tests_failed', 0))
            with col2:
                st.metric("Percentage Score", f"{grading_results.get('percentage', 0)}%")
                
            # Show detailed test results
            st.markdown("#### 📊 Individual Test Results")
            detailed_results = grading_results.get('detailed_results', {})
            for test_name, test_result in detailed_results.items():
                status = "✅ PASS" if test_result.get('passed', False) else "❌ FAIL"
                score = test_result.get('score', 0)
                max_score = test_result.get('max', 0)
                st.write(f"{status} **{test_name.replace('_', ' ').title()}**: {score}/{max_score}")
            
            # Show execution logs
            st.markdown("#### 📝 Execution Logs")
            execution_logs = grading_results.get('execution_logs', [])
            for log in execution_logs:
                if "✅ PASS" in log:
                    st.success(log)
                elif "❌ FAIL" in log:
                    st.error(log)
                elif "⚠️" in log:
                    st.warning(log)
                else:
                    st.info(log)

def render_scoring_guide():
    """Render detailed scoring guide based on grader.py"""
    st.markdown("### ℹ️ Detailed Scoring Guide")
    
    st.markdown("""
    This competition uses a **deterministic grader** that runs comprehensive tests 
    on your AI-generated code. The grader is more precise and consistent than AI-based scoring.
    """)
    
    # Import grader to get actual scoring
    from grader import FunctionQualityGrader
    grader = FunctionQualityGrader()
    
    # Critical Bugs
    st.markdown("#### 🐛 Critical Bugs (65 points)")
    critical_df = pd.DataFrame([
        {"Test": "Handles Division by Zero", "Points": 20, "Description": "When days_active = 0"},
        {"Test": "Handles Empty Users List", "Points": 15, "Description": "users = [] scenario"},
        {"Test": "Handles Missing Keys", "Points": 15, "Description": "Required dict keys missing"},
        {"Test": "Correct Average Calculation", "Points": 15, "Description": "Use active users, not all users"}
    ])
    st.dataframe(critical_df, use_container_width=True)
    
    # Logic Issues  
    st.markdown("#### 🔧 Logic Issues (28 points)")
    logic_df = pd.DataFrame([
        {"Test": "Correct Sorting Direction", "Points": 10, "Description": "Top performers (highest first)"},
        {"Test": "Handles No Active Users", "Points": 10, "Description": "No users in date range"},
        {"Test": "Doesn't Mutate Input", "Points": 8, "Description": "Original data unchanged"}
    ])
    st.dataframe(logic_df, use_container_width=True)
    
    # Edge Cases
    st.markdown("#### 🎯 Edge Cases (17 points)")
    edge_df = pd.DataFrame([
        {"Test": "Handles Less Than 5 Users", "Points": 5, "Description": "Return available users, not fail"},
        {"Test": "Handles Invalid Dates", "Points": 5, "Description": "end_date before start_date"},
        {"Test": "Robust Error Handling", "Points": 7, "Description": "None values, negatives, string numbers"}
    ])
    st.dataframe(edge_df, use_container_width=True)
    
    # Performance
    st.markdown("#### ⚡ Performance (10 points)")
    perf_df = pd.DataFrame([
        {"Test": "Efficient Implementation", "Points": 10, "Description": "Optimal algorithms, correct top 5 selection"}
    ])
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown(f"**Maximum Possible Score: {grader.max_possible_score} points**")
    
    # Show grading criteria
    st.markdown("#### 📊 Grading Criteria")
    st.markdown("""
    - **A (90%+)**: Excellent - Production Ready
    - **B (80%+)**: Good - Minor Issues  
    - **C (70%+)**: Fair - Several Bugs
    - **D (60%+)**: Poor - Major Issues
    - **F (<60%)**: Failing - Critical Bugs
    """)
    
    st.info("💡 **Pro Tips for AI Configuration:**")
    st.markdown("""
    **🎯 Optimization Strategies:**
    - **Lower temperatures** (0.1-0.3) for more consistent, bug-free code generation
    - **Multiple cycles** (3-5) for better refinement and error correction
    - **Specialized layer agents** with different focus areas (bugs, edge cases, performance)
    - **Clear, specific prompts** mentioning exact issues: division by zero, missing keys, sorting
    - **Progressive refinement** - start simple, add complexity gradually
    
    **📝 Prompt Engineering:**
    - Include specific examples of expected fixes in your prompts
    - Mention the grader's focus areas: critical bugs worth 65 points!
    - Use structured prompts: "First identify bugs, then fix systematically"
    - Reference string date comparison (YYYY-MM-DD format works directly)
    
    **🔄 Iteration Tips:**
    - Download your best configs for easy sharing and backup
    - Import successful configurations and modify incrementally  
    - Focus on critical bugs first - they're worth the most points!
    - Use agent specialization: one for bugs, one for edge cases, one for performance
    """)
    
    st.success("""
    🎯 **Key Advantage**: The grader is deterministic and fair - the same code will always get 
    the same score, making this a true test of your AI configuration skills! 
    
    🏆 **Goal**: Configure your AI to achieve 97+ points (80%+) for Grade A performance!
    """)

def import_config_from_json(config_json):
    """Import AI configuration from JSON with bulletproof type handling"""
    try:
        # Parse the JSON
        config_data = json.loads(config_json)
        
        # Basic structure check
        if "ai_configuration" not in config_data:
            return False, "Invalid format: missing 'ai_configuration'"
        
        imported_config = config_data["ai_configuration"]
        
        # Validate required keys exist
        required_keys = ['main_model', 'main_temperature', 'cycles', 'system_prompt', 'layer_agents']
        for key in required_keys:
            if key not in imported_config:
                return False, f"Missing required field: {key}"
        
        # Validate main model is valid
        main_model = str(imported_config['main_model'])
        if main_model not in valid_model_names:
            return False, f"Invalid main model: {main_model}. Must be one of: {', '.join(valid_model_names)}"
        
        # Validate and convert numeric values
        try:
            main_temperature = float(imported_config['main_temperature'])
            if not (0.0 <= main_temperature <= 1.0):
                return False, "main_temperature must be between 0.0 and 1.0"
        except (ValueError, TypeError):
            return False, "main_temperature must be a valid number"
            
        try:
            cycles = int(imported_config['cycles'])
            if not (1 <= cycles <= 10):
                return False, "cycles must be between 1 and 10"
        except (ValueError, TypeError):
            return False, "cycles must be a valid integer"
        
        # Validate layer agents structure
        if not isinstance(imported_config['layer_agents'], list):
            return False, "layer_agents must be a list"
        
        processed_agents = []
        for i, agent in enumerate(imported_config['layer_agents']):
            if not isinstance(agent, dict):
                return False, f"layer_agents[{i}] must be a dictionary"
            
            # Validate agent model
            agent_model = str(agent.get('model', 'llama3.1-8b'))
            if agent_model not in valid_model_names:
                return False, f"Invalid model in layer_agents[{i}]: {agent_model}"
            
            # Validate agent temperature
            try:
                agent_temp = float(agent.get('temperature', 0.5))
                if not (0.0 <= agent_temp <= 1.0):
                    return False, f"layer_agents[{i}] temperature must be between 0.0 and 1.0"
            except (ValueError, TypeError):
                return False, f"layer_agents[{i}] temperature must be a valid number"
            
            processed_agent = {
                'name': str(agent.get('name', f'agent_{i+1}')),
                'model': agent_model,
                'temperature': agent_temp,
                'prompt': str(agent.get('prompt', 'Analyze and improve the code. {helper_response}'))
            }
            processed_agents.append(processed_agent)
        
        # Clear potentially conflicting widget states more selectively
        keys_to_clear = []
        
        # Only clear keys that might conflict with imported values
        for key in list(st.session_state.keys()):
            if (key.startswith('agent_') or 
                key in ['main_model', 'main_temp', 'cycles', 'main_system_prompt']):
                keys_to_clear.append(key)
        
        # Clear the identified keys
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Now assign the clean configuration
        st.session_state.ai_config = {
            'main_model': main_model,
            'main_temperature': main_temperature,
            'cycles': cycles,
            'system_prompt': str(imported_config['system_prompt']),
            'layer_agents': processed_agents
        }
        
        # Update the last saved time
        save_config_to_session()
        
        return True, f"Configuration imported! Main model: {main_model}, {len(processed_agents)} agents loaded"
        
    except json.JSONDecodeError:
        return False, "Invalid JSON format - please check the file"
    except Exception as e:
        return False, f"Import error: {str(e)}"