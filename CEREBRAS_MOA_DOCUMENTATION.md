# Cerebras Mixture-of-Agents (MOA) - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Agent System](#agent-system)
5. [Layer Processing](#layer-processing)
6. [Execution Flow](#execution-flow)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Usage Examples](#usage-examples)
10. [Advanced Features](#advanced-features)

## Overview

The Cerebras Mixture-of-Agents (MOA) is an advanced AI system that implements the MOA architecture proposed by Together AI, powered by Cerebras' high-performance inference infrastructure. The system uses multiple specialized AI agents working in layers to provide enhanced, more accurate responses than single-model approaches.

### Key Features
- **Multi-layer agent processing**: Multiple cycles of agent collaboration
- **Parallel agent execution**: Layer agents process queries simultaneously
- **Streaming responses**: Real-time response generation
- **Configurable architecture**: Customizable agents, models, and parameters
- **Memory management**: Conversation history tracking
- **Web interface**: Streamlit-based UI for easy interaction

## Architecture

The MOA system follows a hierarchical architecture with two main types of agents:

```
User Query
    ↓
┌─────────────────────────────────────────┐
│              LAYER AGENTS               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Agent 1  │ │Agent 2  │ │Agent 3  │   │
│  │Step-by- │ │Thought  │ │Logic &  │   │
│  │Step     │ │Response │ │Reasoning│   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
    ↓ (Responses Combined)
┌─────────────────────────────────────────┐
│            MAIN AGENT                   │
│  Synthesizes layer responses into       │
│  final, high-quality answer             │
└─────────────────────────────────────────┘
    ↓
Final Response to User
```

### Processing Layers

1. **Layer Agents**: Specialized agents that analyze the query from different perspectives
2. **Response Synthesis**: Combining layer agent outputs into structured input
3. **Main Agent**: Final processing agent that generates the ultimate response

## Core Components

### 1. MOAgent Class (`moa/agent/moa.py`)

The central orchestrator that manages the entire MOA workflow.

**Key Responsibilities:**
- Manages layer agents and main agent
- Coordinates multi-layer processing
- Handles streaming responses
- Maintains conversation memory
- Interfaces with Cerebras API

### 2. ConversationMemory

Manages conversation history for context-aware responses.

```python
class ConversationMemory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str)
    def get_messages(self)
    def clear(self)
```

### 3. Response Types

The system generates two types of responses:

- **Intermediate**: Layer agent outputs during processing
- **Output**: Final response from the main agent

## Agent System

### Layer Agents

Layer agents are specialized AI models that process the user query from different perspectives:

#### Default Layer Agent Configuration

```python
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
    }
}
```

#### Agent Specializations

1. **Step-by-Step Agent**: Breaks down problems methodically
2. **Thought-Response Agent**: Provides reasoning followed by answers
3. **Logic & Reasoning Agent**: Focuses on logical analysis
4. **Planning Agent**: Creates structured approaches to problems

### Main Agent

The main agent synthesizes all layer agent responses into a final answer:

**System Prompt:**
```
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. 
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.
```

## Layer Processing

### Multi-Cycle Processing

The system supports multiple processing cycles (layers), where each cycle involves:

1. **Parallel Agent Execution**: All layer agents process the query simultaneously
2. **Response Collection**: Gathering outputs from all layer agents
3. **Response Formatting**: Combining agent outputs into structured format
4. **Context Building**: Creating enhanced context for the next cycle or main agent

### Layer Agent Execution Flow

```python
def process_layer(self, cycle_number, user_input, helper_response=""):
    layer_responses = {}
    
    # Process each agent in the layer
    for agent_name, agent_config in self.layer_agent_config.items():
        # Create specialized messages for this agent
        messages = self._create_chat_messages(
            system_prompt=agent_config['system_prompt'],
            user_input=user_input,
            helper_response=helper_response
        )
        
        # Get response from Cerebras API
        response = self._generate_completion(
            model=agent_config['model_name'],
            messages=messages,
            temperature=agent_config['temperature']
        )
        
        layer_responses[agent_name] = response
    
    return layer_responses
```

## Execution Flow

### Complete Processing Pipeline

When you submit a prompt to the MOA system, here's what happens:

#### 1. Input Processing
```python
user_query = "Explain quantum computing"
```

#### 2. Layer Processing (for each cycle)
```python
# Cycle 1: Initial Analysis
layer_agents = {
    "step_by_step": "1. Quantum computing uses quantum bits...",
    "thought_response": "I think quantum computing is revolutionary because...",
    "logic_reasoning": "Logically, quantum computing differs from classical..."
}
```

#### 3. Response Synthesis
```python
combined_context = """
You have been provided with responses from various models:
0. 1. Quantum computing uses quantum bits...
1. I think quantum computing is revolutionary because...
2. Logically, quantum computing differs from classical...
"""
```

#### 4. Main Agent Processing
```python
main_agent_prompt = system_prompt + combined_context + user_query
final_response = main_agent.generate(main_agent_prompt)
```

#### 5. Streaming Output
The final response is streamed back to the user in real-time.

### Detailed Execution Steps

1. **Initialization**
   - Load configuration
   - Initialize Cerebras client
   - Set up conversation memory

2. **Layer Processing** (repeated for each cycle)
   - Distribute query to all layer agents
   - Execute agents in parallel
   - Collect and format responses
   - Build enhanced context

3. **Main Agent Synthesis**
   - Combine all layer insights
   - Generate final response
   - Stream output to user

4. **Memory Management**
   - Save user query
   - Save assistant response
   - Update conversation history

## Configuration

### Main Configuration Structure

```json
{
  "main_config": {
    "main_model": "llama-3.3-70b",
    "cycles": 1,
    "temperature": 0.0,
    "system_prompt": "You are a personal assistant...",
    "reference_system_prompt": "You have been provided with responses..."
  },
  "layer_config": {
    "layer_agent_1": {
      "system_prompt": "Think through your response step by step. {helper_response}",
      "model_name": "llama-4-scout-17b-16e-instruct",
      "temperature": 0.1
    }
  }
}
```

### Available Models

The system supports these Cerebras models:
- `llama-3.3-70b` - Large, high-capability model
- `llama3.1-8b` - Fast, efficient model
- `llama-4-scout-17b-16e-instruct` - Balanced performance model
- `qwen-3-32b` - Alternative architecture model

### Configuration Parameters

#### Main Agent Parameters
- `main_model`: Primary model for final response generation
- `cycles`: Number of layer processing cycles
- `temperature`: Randomness in response generation (0.0-1.0)
- `max_tokens`: Maximum response length
- `system_prompt`: Instructions for the main agent

#### Layer Agent Parameters
- `model_name`: Cerebras model to use
- `system_prompt`: Specialized instructions for this agent
- `temperature`: Response randomness for this agent
- `max_tokens`: Maximum response length for this agent

## API Reference

### MOAgent Class

#### Constructor
```python
MOAgent(
    main_model: str,
    layer_agent_config: Dict[str, Dict[str, Any]],
    reference_system_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    cycles: Optional[int] = None,
    temperature: Optional[float] = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs
)
```

#### Key Methods

##### `from_config()`
```python
@classmethod
def from_config(
    cls,
    main_model: str = 'llama3.3-70b',
    system_prompt: Optional[str] = None,
    cycles: int = 1,
    layer_agent_config: Optional[Dict] = None,
    reference_system_prompt: Optional[str] = None,
    temperature: Optional[float] = 0.1,
    max_tokens: Optional[int] = None,
    **main_model_kwargs
)
```

##### `chat()`
```python
def chat(
    self, 
    input: str,
    messages: Optional[List[Dict[str, str]]] = None,
    cycles: Optional[int] = None,
    save: bool = True,
    output_format: Literal['string', 'json'] = 'string'
) -> Generator[Union[str, ResponseChunk], None, None]
```

### Response Types

#### ResponseChunk
```python
class ResponseChunk(TypedDict):
    delta: str  # Incremental response content
    response_type: Literal['intermediate', 'output']  # Type of response
    metadata: Dict[str, Any]  # Additional information
```

## Usage Examples

### Basic Usage

```python
from moa.agent import MOAgent

# Create agent with default configuration
agent = MOAgent.from_config(
    main_model='llama-3.3-70b',
    cycles=2
)

# Chat with the agent
response_stream = agent.chat("Explain machine learning")
for chunk in response_stream:
    print(chunk, end='')
```

### Custom Configuration

```python
# Custom layer agent configuration
custom_layer_config = {
    'creative_agent': {
        'system_prompt': 'Provide creative and innovative responses. {helper_response}',
        'model_name': 'qwen-3-32b',
        'temperature': 0.8
    },
    'analytical_agent': {
        'system_prompt': 'Provide analytical and data-driven responses. {helper_response}',
        'model_name': 'llama3.1-8b',
        'temperature': 0.2
    }
}

agent = MOAgent.from_config(
    main_model='llama-3.3-70b',
    layer_agent_config=custom_layer_config,
    cycles=3,
    temperature=0.1
)
```

### Streaming with Metadata

```python
response_stream = agent.chat("Solve this math problem: 2x + 5 = 15", output_format='json')

for chunk in response_stream:
    if chunk['response_type'] == 'intermediate':
        print(f"Layer {chunk['metadata']['layer']}: {chunk['delta']}")
    else:
        print(f"Final: {chunk['delta']}", end='')
```

### Web Interface Usage

```bash
# Start the Streamlit application
streamlit run app.py

# Navigate to http://localhost:8501
# Configure agents in the sidebar
# Chat in the main interface
```

## Advanced Features

### 1. Dynamic Configuration

The web interface allows real-time configuration changes:
- Add/remove layer agents
- Modify system prompts
- Adjust model parameters
- Change processing cycles

### 2. Response Visualization

The Streamlit interface provides:
- Layer-by-layer response visualization
- Agent-specific output display
- Real-time streaming
- Configuration export/import

### 3. Memory Management

```python
# Access conversation history
history = agent.memory.get_messages()

# Clear conversation memory
agent.memory.clear()

# Add custom messages
agent.memory.add_message("user", "Previous question")
agent.memory.add_message("assistant", "Previous response")
```

### 4. Error Handling

The system includes robust error handling:
- API connection failures
- Model availability issues
- Configuration validation
- Graceful degradation

### 5. Performance Optimization

- Parallel layer agent execution
- Streaming response generation
- Efficient memory usage
- Configurable token limits

## Best Practices

### 1. Agent Configuration
- Use diverse models for layer agents
- Balance temperature settings
- Craft specific system prompts
- Limit max_tokens appropriately

### 2. Cycle Management
- Start with 1-2 cycles for simple queries
- Use 3+ cycles for complex problems
- Monitor response quality vs. processing time

### 3. Model Selection
- Use larger models for main agent
- Use faster models for layer agents
- Consider task-specific model strengths

### 4. Prompt Engineering
- Make system prompts specific and clear
- Use the `{helper_response}` placeholder effectively
- Test different prompt variations

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   export CEREBRAS_API_KEY="your-api-key-here"
   ```

2. **Model Availability**
   - Check model names against available models
   - Verify API access permissions

3. **Configuration Errors**
   - Validate JSON configuration syntax
   - Ensure required fields are present

4. **Performance Issues**
   - Reduce number of cycles
   - Lower max_tokens limits
   - Use faster models for layer agents

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export DEBUG=1
export CEREBRAS_DEBUG=1
```

This comprehensive documentation covers the complete Cerebras MOA system architecture, from basic usage to advanced configuration and troubleshooting. The system provides a powerful framework for leveraging multiple AI agents to generate enhanced, more accurate responses than traditional single-model approaches. 