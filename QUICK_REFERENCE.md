# Cerebras MOA - Quick Reference Guide

## 🚀 Quick Start

### 1. Basic Setup
```python
from moa.agent import MOAgent

# Simple setup with defaults
agent = MOAgent.from_config(main_model='llama-3.3-70b')

# Chat
for chunk in agent.chat("Your question here"):
    print(chunk, end='')
```

### 2. Web Interface
```bash
streamlit run app.py
# Open http://localhost:8501
```

## 🏗️ Architecture Overview

```
User Query → Layer Agents (Parallel) → Response Synthesis → Main Agent → Final Response
```

**Layer Agents**: Specialized AI models analyzing from different perspectives
**Main Agent**: Synthesizes layer outputs into final response

## ⚙️ Configuration

### Default Layer Agents
- **Agent 1**: Step-by-step analysis (`llama-4-scout-17b-16e-instruct`)
- **Agent 2**: Thought + response (`qwen-3-32b`)
- **Agent 3**: Logic & reasoning (`llama3.1-8b`)

### Custom Configuration
```python
custom_config = {
    'creative_agent': {
        'system_prompt': 'Be creative and innovative. {helper_response}',
        'model_name': 'qwen-3-32b',
        'temperature': 0.8
    },
    'analytical_agent': {
        'system_prompt': 'Be analytical and precise. {helper_response}',
        'model_name': 'llama3.1-8b',
        'temperature': 0.2
    }
}

agent = MOAgent.from_config(
    main_model='llama-3.3-70b',
    layer_agent_config=custom_config,
    cycles=2
)
```

## 🔧 Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `main_model` | Primary model for final response | `llama-3.3-70b` | See available models |
| `cycles` | Number of processing layers | `1` | `1-5` |
| `temperature` | Response randomness | `0.1` | `0.0-1.0` |
| `max_tokens` | Maximum response length | `None` | `1-4096` |

## 🤖 Available Models

- `llama-3.3-70b` - Large, high-capability
- `llama3.1-8b` - Fast, efficient  
- `llama-4-scout-17b-16e-instruct` - Balanced
- `qwen-3-32b` - Alternative architecture

## 📊 Response Formats

### String Format (Default)
```python
for chunk in agent.chat("Question"):
    print(chunk, end='')
```

### JSON Format (with metadata)
```python
for chunk in agent.chat("Question", output_format='json'):
    if chunk['response_type'] == 'intermediate':
        print(f"Layer: {chunk['delta']}")
    else:
        print(f"Final: {chunk['delta']}")
```

## 🔄 Execution Flow

1. **Input Processing**: User query received
2. **Layer Processing**: All layer agents process in parallel
3. **Response Synthesis**: Combine layer outputs
4. **Main Agent**: Generate final response
5. **Streaming**: Real-time output to user
6. **Memory**: Save conversation history

## 💡 Best Practices

### Agent Configuration
- Use diverse models for different perspectives
- Balance temperature: low for factual, high for creative
- Craft specific system prompts for each agent
- Start with 1-2 cycles, increase for complex queries

### Performance
- Use faster models (`llama3.1-8b`) for layer agents
- Use larger models (`llama-3.3-70b`) for main agent
- Limit `max_tokens` to control response length
- Monitor API usage and costs

### Prompt Engineering
- Include `{helper_response}` in system prompts
- Be specific about agent roles and expectations
- Test different prompt variations
- Use clear, actionable instructions

## 🐛 Common Issues & Solutions

### API Key
```bash
export CEREBRAS_API_KEY="your-key-here"
# or create .env file
```

### Model Not Available
- Check model name spelling
- Verify API access permissions
- Use alternative models

### Slow Performance
- Reduce number of cycles
- Use faster models for layer agents
- Lower max_tokens limits

### Configuration Errors
- Validate JSON syntax
- Ensure all required fields present
- Check temperature ranges (0.0-1.0)

## 📝 Example Use Cases

### Research Assistant
```python
research_config = {
    'fact_checker': {
        'system_prompt': 'Verify facts and provide sources. {helper_response}',
        'model_name': 'llama3.1-8b',
        'temperature': 0.1
    },
    'synthesizer': {
        'system_prompt': 'Synthesize information clearly. {helper_response}',
        'model_name': 'qwen-3-32b',
        'temperature': 0.3
    }
}
```

### Creative Writing
```python
creative_config = {
    'storyteller': {
        'system_prompt': 'Create engaging narratives. {helper_response}',
        'model_name': 'qwen-3-32b',
        'temperature': 0.8
    },
    'editor': {
        'system_prompt': 'Improve clarity and flow. {helper_response}',
        'model_name': 'llama-4-scout-17b-16e-instruct',
        'temperature': 0.4
    }
}
```

### Technical Analysis
```python
technical_config = {
    'analyzer': {
        'system_prompt': 'Provide technical analysis. {helper_response}',
        'model_name': 'llama3.1-8b',
        'temperature': 0.2
    },
    'validator': {
        'system_prompt': 'Validate technical accuracy. {helper_response}',
        'model_name': 'llama-4-scout-17b-16e-instruct',
        'temperature': 0.1
    }
}
```

## 🔍 Debug Mode

```bash
export DEBUG=1
export CEREBRAS_DEBUG=1
python your_script.py
```

## 📚 Key Files

- `moa/agent/moa.py` - Core MOA implementation
- `moa/agent/prompts.py` - System prompts
- `app.py` - Streamlit web interface
- `config/moa_config.json` - Configuration examples

## 🌐 Web Interface Features

- **Real-time configuration**: Modify agents on-the-fly
- **Response visualization**: See layer-by-layer outputs
- **Configuration export/import**: Save and load setups
- **Streaming responses**: Real-time output display

This quick reference provides everything you need to get started with the Cerebras MOA system efficiently! 