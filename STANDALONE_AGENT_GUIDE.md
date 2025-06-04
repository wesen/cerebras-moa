# Building Standalone Problem-Solving Agents with Cerebras MOA

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture Extraction](#core-architecture-extraction)
3. [Standalone Agent Components](#standalone-agent-components)
4. [Implementation Guide](#implementation-guide)
5. [Specialized Agent Types](#specialized-agent-types)
6. [Configuration Patterns](#configuration-patterns)
7. [Deployment Strategies](#deployment-strategies)
8. [Examples and Use Cases](#examples-and-use-cases)

## Overview

This guide shows how to extract and adapt the Cerebras MOA (Mixture-of-Agents) architecture to build standalone problem-solving agents. The MOA system's multi-layer agent approach can be simplified and specialized for specific problem domains while maintaining its core benefits of enhanced reasoning and accuracy.

### Why Build Standalone Agents?

- **Focused Problem Solving**: Target specific domains (math, coding, analysis)
- **Reduced Complexity**: Simpler deployment and maintenance
- **Cost Optimization**: Use only necessary models and cycles
- **Custom Workflows**: Tailor agent behavior to specific use cases
- **Performance**: Faster execution for specialized tasks

## Core Architecture Extraction

### Key Components from MOA

The Cerebras MOA system has several extractable components that can be used independently:

1. **Multi-Agent Processing**: Multiple specialized agents working on the same problem
2. **Response Synthesis**: Combining multiple perspectives into a final answer
3. **Configurable Models**: Different models for different reasoning styles
4. **Streaming Responses**: Real-time output generation
5. **Memory Management**: Conversation context handling

### Minimal Standalone Architecture

```
User Problem
    ↓
┌─────────────────────────────────────────┐
│         SPECIALIZED AGENTS              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Agent A  │ │Agent B  │ │Agent C  │   │
│  │Domain   │ │Method   │ │Quality  │   │
│  │Expert   │ │Expert   │ │Checker  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
    ↓ (Responses Combined)
┌─────────────────────────────────────────┐
│          SYNTHESIS AGENT                │
│  Combines insights into final solution  │
└─────────────────────────────────────────┘
    ↓
Final Solution
```

## Standalone Agent Components

### 1. Base Agent Class

Extract the core agent functionality:

```python
import os
from typing import Dict, List, Optional, Generator, Any
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

class StandaloneAgent:
    """Base class for standalone problem-solving agents"""
    
    def __init__(
        self,
        model: str = "llama3.1-8b",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_prompt: str = "You are a helpful assistant."
    ):
        load_dotenv()
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
    
    def generate(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """Generate a response for the given prompt"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            if stream:
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                full_response = ""
                for chunk in response_stream:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content
                return full_response
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
```

### 2. Multi-Agent Coordinator

Extract the coordination logic:

```python
class MultiAgentCoordinator:
    """Coordinates multiple specialized agents"""
    
    def __init__(self, agents: Dict[str, StandaloneAgent]):
        self.agents = agents
        self.synthesis_prompt = """
        You have received responses from multiple specialized agents.
        Analyze their outputs and synthesize them into a single, high-quality solution.
        Consider the strengths of each response and resolve any conflicts.
        
        Agent Responses:
        {responses}
        
        Provide a comprehensive, accurate final answer.
        """
    
    def solve(self, problem: str, use_synthesis: bool = True) -> Dict[str, Any]:
        """Solve a problem using multiple agents"""
        agent_responses = {}
        
        # Get responses from all agents
        for agent_name, agent in self.agents.items():
            response = agent.generate(problem)
            agent_responses[agent_name] = response
        
        if not use_synthesis:
            return {"agent_responses": agent_responses}
        
        # Synthesize responses
        responses_text = "\n".join([
            f"{name}: {response}" 
            for name, response in agent_responses.items()
        ])
        
        synthesis_agent = StandaloneAgent(
            model="llama-3.3-70b",
            temperature=0.0,
            system_prompt="You are an expert at synthesizing multiple perspectives into optimal solutions."
        )
        
        final_response = synthesis_agent.generate(
            self.synthesis_prompt.format(responses=responses_text)
        )
        
        return {
            "agent_responses": agent_responses,
            "final_solution": final_response
        }
```

### 3. Memory and Context Manager

Extract memory management:

```python
class AgentMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.messages = []
        self.max_history = max_history
        self.context = {}
    
    def add_interaction(self, problem: str, solution: str, metadata: Dict = None):
        """Add a problem-solution pair to memory"""
        self.messages.append({
            "problem": problem,
            "solution": solution,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        
        # Keep only recent interactions
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_relevant_context(self, current_problem: str, similarity_threshold: float = 0.7) -> str:
        """Get relevant past interactions for context"""
        # Simple keyword-based similarity (can be enhanced with embeddings)
        relevant_interactions = []
        current_keywords = set(current_problem.lower().split())
        
        for interaction in self.messages:
            past_keywords = set(interaction["problem"].lower().split())
            similarity = len(current_keywords & past_keywords) / len(current_keywords | past_keywords)
            
            if similarity >= similarity_threshold:
                relevant_interactions.append(interaction)
        
        if not relevant_interactions:
            return ""
        
        context = "Relevant past interactions:\n"
        for interaction in relevant_interactions[-3:]:  # Last 3 relevant
            context += f"Problem: {interaction['problem']}\n"
            context += f"Solution: {interaction['solution']}\n\n"
        
        return context
```

## Implementation Guide

### Step 1: Basic Standalone Agent

Create a minimal working agent:

```python
# standalone_agent.py
import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

class ProblemSolver:
    def __init__(self, domain: str = "general"):
        load_dotenv()
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        self.domain = domain
        self.model = "llama3.1-8b"
        
        # Domain-specific system prompts
        self.prompts = {
            "math": "You are a mathematics expert. Solve problems step-by-step with clear explanations.",
            "coding": "You are a programming expert. Write clean, efficient code with explanations.",
            "analysis": "You are a data analyst. Provide thorough analysis with insights.",
            "general": "You are a helpful problem-solving assistant."
        }
    
    def solve(self, problem: str) -> str:
        system_prompt = self.prompts.get(self.domain, self.prompts["general"])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        
        return response.choices[0].message.content

# Usage
solver = ProblemSolver(domain="math")
result = solver.solve("Solve the quadratic equation: x² + 5x + 6 = 0")
print(result)
```

### Step 2: Multi-Agent Problem Solver

Build a system with multiple specialized agents:

```python
# multi_agent_solver.py
from typing import Dict, List
import json

class SpecializedProblemSolver:
    def __init__(self):
        load_dotenv()
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        
        # Define specialized agents
        self.agents = {
            "analyzer": {
                "model": "llama3.1-8b",
                "prompt": "Analyze the problem thoroughly. Break it down into components and identify key requirements.",
                "temperature": 0.1
            },
            "solver": {
                "model": "llama-4-scout-17b-16e-instruct", 
                "prompt": "Solve the problem step-by-step. Show your work and reasoning clearly.",
                "temperature": 0.3
            },
            "validator": {
                "model": "qwen-3-32b",
                "prompt": "Review the solution for accuracy and completeness. Check for errors or improvements.",
                "temperature": 0.1
            }
        }
    
    def _get_agent_response(self, agent_config: Dict, problem: str, context: str = "") -> str:
        system_prompt = agent_config["prompt"]
        if context:
            system_prompt += f"\n\nContext from other agents:\n{context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ]
        
        response = self.client.chat.completions.create(
            model=agent_config["model"],
            messages=messages,
            temperature=agent_config["temperature"]
        )
        
        return response.choices[0].message.content
    
    def solve_collaborative(self, problem: str) -> Dict[str, str]:
        results = {}
        context = ""
        
        # Sequential processing with context building
        for agent_name, agent_config in self.agents.items():
            response = self._get_agent_response(agent_config, problem, context)
            results[agent_name] = response
            context += f"\n{agent_name.title()}: {response}\n"
        
        # Final synthesis
        synthesis_prompt = """
        Based on the analysis, solution, and validation provided by the specialized agents,
        provide a final, comprehensive answer to the problem.
        
        Problem: {problem}
        
        Agent Outputs:
        {context}
        
        Final Answer:
        """
        
        final_response = self._get_agent_response(
            {
                "model": "llama-3.3-70b",
                "prompt": "You are an expert synthesizer. Combine multiple perspectives into optimal solutions.",
                "temperature": 0.0
            },
            synthesis_prompt.format(problem=problem, context=context)
        )
        
        results["final_solution"] = final_response
        return results

# Usage
solver = SpecializedProblemSolver()
result = solver.solve_collaborative("Design a simple web scraper in Python")
print(json.dumps(result, indent=2))
```

### Step 3: Configurable Agent System

Create a flexible, configuration-driven system:

```python
# configurable_agent.py
import json
from typing import Dict, Any, Optional

class ConfigurableAgentSystem:
    def __init__(self, config_file: str = None, config_dict: Dict = None):
        load_dotenv()
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "agents": {
                "primary": {
                    "model": "llama3.1-8b",
                    "temperature": 0.1,
                    "system_prompt": "You are a helpful problem-solving assistant.",
                    "max_tokens": 1000
                }
            },
            "synthesis": {
                "enabled": False,
                "model": "llama-3.3-70b",
                "temperature": 0.0
            },
            "memory": {
                "enabled": True,
                "max_history": 5
            }
        }
    
    def solve(self, problem: str, agent_name: str = None) -> Dict[str, Any]:
        if agent_name and agent_name in self.config["agents"]:
            # Use specific agent
            agent_config = self.config["agents"][agent_name]
            response = self._generate_response(agent_config, problem)
            return {"agent": agent_name, "response": response}
        else:
            # Use all agents
            responses = {}
            for name, config in self.config["agents"].items():
                responses[name] = self._generate_response(config, problem)
            
            if self.config["synthesis"]["enabled"]:
                final_response = self._synthesize_responses(problem, responses)
                return {"agent_responses": responses, "final_solution": final_response}
            else:
                return {"agent_responses": responses}
    
    def _generate_response(self, agent_config: Dict, problem: str) -> str:
        messages = [
            {"role": "system", "content": agent_config["system_prompt"]},
            {"role": "user", "content": problem}
        ]
        
        response = self.client.chat.completions.create(
            model=agent_config["model"],
            messages=messages,
            temperature=agent_config["temperature"],
            max_tokens=agent_config.get("max_tokens")
        )
        
        return response.choices[0].message.content
    
    def _synthesize_responses(self, problem: str, responses: Dict[str, str]) -> str:
        synthesis_config = self.config["synthesis"]
        
        responses_text = "\n".join([
            f"{name}: {response}" for name, response in responses.items()
        ])
        
        synthesis_prompt = f"""
        Problem: {problem}
        
        Agent Responses:
        {responses_text}
        
        Synthesize these responses into a single, optimal solution.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at combining multiple perspectives."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=synthesis_config["model"],
            messages=messages,
            temperature=synthesis_config["temperature"]
        )
        
        return response.choices[0].message.content

# Example configuration file (config.json)
example_config = {
    "agents": {
        "creative": {
            "model": "qwen-3-32b",
            "temperature": 0.8,
            "system_prompt": "You are a creative problem solver. Think outside the box.",
            "max_tokens": 800
        },
        "analytical": {
            "model": "llama3.1-8b", 
            "temperature": 0.1,
            "system_prompt": "You are an analytical thinker. Use logic and data.",
            "max_tokens": 1000
        },
        "practical": {
            "model": "llama-4-scout-17b-16e-instruct",
            "temperature": 0.3,
            "system_prompt": "You are focused on practical, implementable solutions.",
            "max_tokens": 900
        }
    },
    "synthesis": {
        "enabled": True,
        "model": "llama-3.3-70b",
        "temperature": 0.0
    }
}

# Usage
solver = ConfigurableAgentSystem(config_dict=example_config)
result = solver.solve("How can I improve team productivity?")
print(json.dumps(result, indent=2))
```

## Specialized Agent Types

### 1. Code Problem Solver

```python
class CodeProblemSolver(StandaloneAgent):
    def __init__(self):
        super().__init__(
            model="llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            system_prompt="""You are an expert programmer. For each coding problem:
            1. Understand the requirements
            2. Plan the solution approach
            3. Write clean, efficient code
            4. Explain your solution
            5. Consider edge cases"""
        )
    
    def solve_coding_problem(self, problem: str, language: str = "python") -> Dict[str, str]:
        prompt = f"""
        Programming Language: {language}
        Problem: {problem}
        
        Please provide:
        1. Solution approach
        2. Complete code implementation
        3. Explanation of the solution
        4. Time/space complexity analysis
        """
        
        response = self.generate(prompt)
        
        # Parse response into sections (simplified)
        sections = {
            "full_response": response,
            "approach": self._extract_section(response, "approach"),
            "code": self._extract_section(response, "code"),
            "explanation": self._extract_section(response, "explanation")
        }
        
        return sections
    
    def _extract_section(self, text: str, section: str) -> str:
        # Simple section extraction (can be enhanced)
        lines = text.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if section.lower() in line.lower():
                in_section = True
                continue
            elif in_section and any(keyword in line.lower() for keyword in ['approach', 'code', 'explanation', 'complexity']):
                break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
```

### 2. Math Problem Solver

```python
class MathProblemSolver(StandaloneAgent):
    def __init__(self):
        super().__init__(
            model="llama3.1-8b",
            temperature=0.0,
            system_prompt="""You are a mathematics expert. For each problem:
            1. Identify the type of problem
            2. Show step-by-step solution
            3. Verify your answer
            4. Explain the mathematical concepts used"""
        )
    
    def solve_math_problem(self, problem: str, show_work: bool = True) -> Dict[str, str]:
        if show_work:
            prompt = f"""
            Math Problem: {problem}
            
            Please solve this step-by-step:
            1. Identify what type of problem this is
            2. Show each step of your solution
            3. Provide the final answer
            4. Verify your solution
            """
        else:
            prompt = f"Solve this math problem: {problem}"
        
        response = self.generate(prompt)
        
        return {
            "problem": problem,
            "solution": response,
            "final_answer": self._extract_final_answer(response)
        }
    
    def _extract_final_answer(self, solution: str) -> str:
        # Extract final answer (simplified pattern matching)
        import re
        patterns = [
            r"final answer[:\s]*([^\n]+)",
            r"answer[:\s]*([^\n]+)",
            r"solution[:\s]*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Answer not clearly identified"
```

### 3. Research and Analysis Agent

```python
class ResearchAnalysisAgent(StandaloneAgent):
    def __init__(self):
        super().__init__(
            model="llama-3.3-70b",
            temperature=0.2,
            system_prompt="""You are a research analyst. For each query:
            1. Break down the research question
            2. Identify key areas to investigate
            3. Provide comprehensive analysis
            4. Draw evidence-based conclusions
            5. Suggest further research directions"""
        )
    
    def analyze_topic(self, topic: str, depth: str = "medium") -> Dict[str, str]:
        depth_prompts = {
            "shallow": "Provide a brief overview and key points about: {topic}",
            "medium": "Provide a comprehensive analysis of: {topic}. Include background, current state, key issues, and implications.",
            "deep": "Conduct a thorough research analysis of: {topic}. Include historical context, current developments, multiple perspectives, evidence, and detailed conclusions."
        }
        
        prompt = depth_prompts.get(depth, depth_prompts["medium"]).format(topic=topic)
        
        response = self.generate(prompt)
        
        return {
            "topic": topic,
            "depth": depth,
            "analysis": response,
            "key_points": self._extract_key_points(response)
        }
    
    def _extract_key_points(self, analysis: str) -> List[str]:
        # Extract key points (simplified)
        lines = analysis.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 'key' in line.lower():
                key_points.append(line)
        
        return key_points[:5]  # Top 5 key points
```

## Configuration Patterns

### 1. Domain-Specific Configurations

```python
# Domain configurations for different problem types
DOMAIN_CONFIGS = {
    "software_debugging": {
        "agents": {
            "bug_hunter": {
                "model": "llama-4-scout-17b-16e-instruct",
                "temperature": 0.1,
                "system_prompt": "You are a bug detection expert. Find and explain code issues systematically."
            },
            "solution_provider": {
                "model": "qwen-3-32b",
                "temperature": 0.3,
                "system_prompt": "You provide practical solutions to code problems with working examples."
            },
            "code_reviewer": {
                "model": "llama3.1-8b",
                "temperature": 0.1,
                "system_prompt": "You review code solutions for best practices and potential improvements."
            }
        },
        "synthesis": {"enabled": True, "model": "llama-3.3-70b"}
    },
    
    "business_analysis": {
        "agents": {
            "market_analyst": {
                "model": "llama-3.3-70b",
                "temperature": 0.2,
                "system_prompt": "You analyze market conditions and business opportunities."
            },
            "financial_analyst": {
                "model": "llama3.1-8b",
                "temperature": 0.1,
                "system_prompt": "You focus on financial implications and cost-benefit analysis."
            },
            "strategy_consultant": {
                "model": "qwen-3-32b",
                "temperature": 0.4,
                "system_prompt": "You provide strategic recommendations and implementation plans."
            }
        },
        "synthesis": {"enabled": True, "model": "llama-3.3-70b"}
    },
    
    "scientific_research": {
        "agents": {
            "literature_reviewer": {
                "model": "llama-3.3-70b",
                "temperature": 0.1,
                "system_prompt": "You review and synthesize scientific literature and research."
            },
            "methodology_expert": {
                "model": "llama3.1-8b",
                "temperature": 0.2,
                "system_prompt": "You design research methodologies and experimental approaches."
            },
            "data_analyst": {
                "model": "qwen-3-32b",
                "temperature": 0.1,
                "system_prompt": "You analyze data patterns and statistical significance."
            }
        },
        "synthesis": {"enabled": True, "model": "llama-3.3-70b"}
    }
}
```

### 2. Performance-Optimized Configurations

```python
# Fast response configuration
FAST_CONFIG = {
    "agents": {
        "quick_solver": {
            "model": "llama3.1-8b",
            "temperature": 0.1,
            "max_tokens": 500,
            "system_prompt": "Provide quick, accurate solutions. Be concise but complete."
        }
    },
    "synthesis": {"enabled": False}
}

# High-quality configuration
QUALITY_CONFIG = {
    "agents": {
        "deep_thinker": {
            "model": "llama-3.3-70b",
            "temperature": 0.1,
            "max_tokens": 2000,
            "system_prompt": "Provide thorough, high-quality analysis and solutions."
        },
        "validator": {
            "model": "qwen-3-32b",
            "temperature": 0.0,
            "max_tokens": 1000,
            "system_prompt": "Validate and improve the solution quality."
        }
    },
    "synthesis": {"enabled": True, "model": "llama-3.3-70b"}
}

# Balanced configuration
BALANCED_CONFIG = {
    "agents": {
        "primary": {
            "model": "llama-4-scout-17b-16e-instruct",
            "temperature": 0.2,
            "max_tokens": 1000,
            "system_prompt": "Provide balanced solutions with good quality and reasonable speed."
        },
        "reviewer": {
            "model": "llama3.1-8b",
            "temperature": 0.1,
            "max_tokens": 500,
            "system_prompt": "Review and refine the solution."
        }
    },
    "synthesis": {"enabled": True, "model": "llama-4-scout-17b-16e-instruct"}
}
```

## Deployment Strategies

### 1. Standalone Script Deployment

```python
#!/usr/bin/env python3
# standalone_solver.py

import sys
import argparse
from typing import Dict, Any

def create_solver(domain: str) -> ConfigurableAgentSystem:
    """Create a solver for the specified domain"""
    config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["software_debugging"])
    return ConfigurableAgentSystem(config_dict=config)

def main():
    parser = argparse.ArgumentParser(description="Standalone Problem Solver")
    parser.add_argument("problem", help="Problem to solve")
    parser.add_argument("--domain", default="software_debugging", 
                       choices=list(DOMAIN_CONFIGS.keys()),
                       help="Problem domain")
    parser.add_argument("--output", choices=["simple", "detailed", "json"], 
                       default="simple", help="Output format")
    
    args = parser.parse_args()
    
    solver = create_solver(args.domain)
    result = solver.solve(args.problem)
    
    if args.output == "json":
        print(json.dumps(result, indent=2))
    elif args.output == "detailed":
        print(f"Problem: {args.problem}")
        print(f"Domain: {args.domain}")
        print("\nSolution:")
        if "final_solution" in result:
            print(result["final_solution"])
        else:
            for agent, response in result["agent_responses"].items():
                print(f"\n{agent.title()}:")
                print(response)
    else:
        # Simple output
        if "final_solution" in result:
            print(result["final_solution"])
        else:
            print(list(result["agent_responses"].values())[0])

if __name__ == "__main__":
    main()

# Usage:
# python standalone_solver.py "Fix this Python code: def divide(a, b): return a/b" --domain software_debugging
# python standalone_solver.py "Analyze the market for electric vehicles" --domain business_analysis --output detailed
```

### 2. API Service Deployment

```python
# api_service.py
from flask import Flask, request, jsonify, stream_template
import json

app = Flask(__name__)

# Global solver instances
solvers = {
    domain: ConfigurableAgentSystem(config_dict=config)
    for domain, config in DOMAIN_CONFIGS.items()
}

@app.route('/solve', methods=['POST'])
def solve_problem():
    data = request.json
    problem = data.get('problem')
    domain = data.get('domain', 'software_debugging')
    
    if not problem:
        return jsonify({"error": "Problem is required"}), 400
    
    if domain not in solvers:
        return jsonify({"error": f"Unknown domain: {domain}"}), 400
    
    try:
        result = solvers[domain].solve(problem)
        return jsonify({
            "problem": problem,
            "domain": domain,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/domains', methods=['GET'])
def get_domains():
    return jsonify({"domains": list(DOMAIN_CONFIGS.keys())})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# Usage:
# curl -X POST http://localhost:5000/solve \
#   -H "Content-Type: application/json" \
#   -d '{"problem": "Debug this code", "domain": "software_debugging"}'
```

### 3. CLI Tool Deployment

```python
# cli_tool.py
import click
import json
from pathlib import Path

@click.group()
def cli():
    """Standalone Problem Solver CLI"""
    pass

@cli.command()
@click.argument('problem')
@click.option('--domain', default='software_debugging', help='Problem domain')
@click.option('--config', help='Custom configuration file')
@click.option('--output', type=click.Choice(['simple', 'detailed', 'json']), default='simple')
def solve(problem, domain, config, output):
    """Solve a problem using AI agents"""
    if config and Path(config).exists():
        solver = ConfigurableAgentSystem(config_file=config)
    else:
        config_dict = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["software_debugging"])
        solver = ConfigurableAgentSystem(config_dict=config_dict)
    
    result = solver.solve(problem)
    
    if output == 'json':
        click.echo(json.dumps(result, indent=2))
    elif output == 'detailed':
        click.echo(f"Problem: {problem}")
        click.echo(f"Domain: {domain}")
        click.echo("\nSolution:")
        if "final_solution" in result:
            click.echo(result["final_solution"])
        else:
            for agent, response in result["agent_responses"].items():
                click.echo(f"\n{agent.title()}:")
                click.echo(response)
    else:
        if "final_solution" in result:
            click.echo(result["final_solution"])
        else:
            click.echo(list(result["agent_responses"].values())[0])

@cli.command()
def domains():
    """List available problem domains"""
    for domain in DOMAIN_CONFIGS.keys():
        click.echo(domain)

@cli.command()
@click.argument('domain')
def describe(domain):
    """Describe a problem domain configuration"""
    if domain in DOMAIN_CONFIGS:
        config = DOMAIN_CONFIGS[domain]
        click.echo(f"Domain: {domain}")
        click.echo(f"Agents: {list(config['agents'].keys())}")
        click.echo(f"Synthesis enabled: {config['synthesis']['enabled']}")
    else:
        click.echo(f"Unknown domain: {domain}")

if __name__ == '__main__':
    cli()

# Usage:
# python cli_tool.py solve "Fix this bug" --domain software_debugging
# python cli_tool.py domains
# python cli_tool.py describe business_analysis
```

## Examples and Use Cases

### 1. Code Debugging Agent

```python
# Example: Debugging a Python function
problem = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This function crashes with empty lists
"""

solver = ConfigurableAgentSystem(config_dict=DOMAIN_CONFIGS["software_debugging"])
result = solver.solve(problem)
print(result["final_solution"])
```

### 2. Business Strategy Agent

```python
# Example: Market analysis
problem = """
Our company is considering entering the electric vehicle charging station market.
What are the key factors we should consider and what strategy would you recommend?
"""

solver = ConfigurableAgentSystem(config_dict=DOMAIN_CONFIGS["business_analysis"])
result = solver.solve(problem)
print(result["final_solution"])
```

### 3. Research Analysis Agent

```python
# Example: Scientific research question
problem = """
What are the current challenges and opportunities in quantum computing for 
practical applications, and what research directions show the most promise?
"""

solver = ConfigurableAgentSystem(config_dict=DOMAIN_CONFIGS["scientific_research"])
result = solver.solve(problem)
print(result["final_solution"])
```

### 4. Custom Configuration Example

```python
# Create a custom agent for creative writing
creative_writing_config = {
    "agents": {
        "idea_generator": {
            "model": "qwen-3-32b",
            "temperature": 0.8,
            "system_prompt": "Generate creative ideas and concepts. Think outside the box."
        },
        "structure_expert": {
            "model": "llama3.1-8b",
            "temperature": 0.3,
            "system_prompt": "Organize ideas into coherent structures and narratives."
        },
        "style_refiner": {
            "model": "llama-4-scout-17b-16e-instruct",
            "temperature": 0.5,
            "system_prompt": "Refine writing style and improve clarity and engagement."
        }
    },
    "synthesis": {
        "enabled": True,
        "model": "llama-3.3-70b",
        "temperature": 0.4
    }
}

# Use the custom configuration
creative_solver = ConfigurableAgentSystem(config_dict=creative_writing_config)
story_result = creative_solver.solve("Write a short story about AI and human collaboration")
print(story_result["final_solution"])
```

## Best Practices for Standalone Agents

### 1. Agent Specialization
- Design agents with specific expertise areas
- Use appropriate models for different reasoning types
- Adjust temperature based on creativity vs. accuracy needs

### 2. Configuration Management
- Use configuration files for easy customization
- Implement validation for configuration parameters
- Provide sensible defaults for common use cases

### 3. Error Handling
- Implement robust error handling for API failures
- Provide fallback mechanisms for model unavailability
- Log errors for debugging and monitoring

### 4. Performance Optimization
- Cache frequently used configurations
- Implement request batching where possible
- Monitor token usage and costs

### 5. Testing and Validation
- Test agents with diverse problem types
- Validate output quality and consistency
- Implement automated testing for critical paths

This guide provides a comprehensive foundation for extracting and adapting the Cerebras MOA architecture to build specialized, standalone problem-solving agents. The modular approach allows for easy customization and deployment across different domains and use cases. 