# Multi-Agent Testing and Grading Framework

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration Management](#configuration-management)
4. [Testing Pipeline](#testing-pipeline)
5. [Grading System Integration](#grading-system-integration)
6. [Streaming Application Design](#streaming-application-design)
7. [Batch Testing Implementation](#batch-testing-implementation)
8. [Results Analysis and Visualization](#results-analysis-and-visualization)
9. [Example Configurations](#example-configurations)
10. [Performance Optimization](#performance-optimization)

## Overview

This framework enables systematic testing and grading of multiple AI agent configurations against coding challenges. It builds upon the existing Cerebras MOA architecture, competition UI, and grading system to create a comprehensive testing pipeline.

### Key Features
- **Batch Configuration Testing**: Test multiple JSON configurations automatically
- **Streaming Execution**: Real-time progress updates and results
- **Automated Grading**: Integration with `grader.py` for consistent scoring
- **Comparative Analysis**: Compare performance across configurations
- **Result Persistence**: Store results in SQLite for analysis
- **Visualization Dashboard**: Streamlit-based UI for insights

### System Components
```
┌─────────────────────────────────────────────────────┐
│                Configuration Manager                 │
│  • Load JSON configs from directory                 │
│  • Validate configurations                          │
│  • Queue management                                 │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                  Testing Pipeline                    │
│  • MOAgent initialization (moa/agent/moa.py)       │
│  • Code generation via AI                          │
│  • Function extraction                              │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                  Grading System                      │
│  • FunctionQualityGrader (grader.py)               │
│  • Score calculation                                │
│  • Detailed feedback                                │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              Results & Visualization                 │
│  • SQLite storage (competition.db)                 │
│  • Streamlit dashboard                             │
│  • Comparative analysis                            │
└─────────────────────────────────────────────────────┘
```

## Architecture

### Core Dependencies

```python
# From existing codebase
from moa.agent import MOAgent  # Multi-agent orchestration
from grader import FunctionQualityGrader  # Code quality grading
from competitive_programming import extract_function_code  # Function extraction
from competition_ui import generate_code_with_ai  # AI code generation

# Additional requirements
import streamlit as st
import pandas as pd
import sqlite3
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
```

### Database Schema

Extend the existing `competition.db` with new tables:

```sql
-- Configuration test runs
CREATE TABLE IF NOT EXISTS config_test_runs (
    run_id TEXT PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_configs INTEGER,
    completed_configs INTEGER,
    status TEXT DEFAULT 'running'
);

-- Individual configuration results
CREATE TABLE IF NOT EXISTS config_test_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    config_name TEXT,
    config_json TEXT,
    generated_code TEXT,
    total_score INTEGER,
    max_score INTEGER,
    percentage REAL,
    grade TEXT,
    detailed_results TEXT,
    execution_logs TEXT,
    generation_time REAL,
    test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES config_test_runs(run_id)
);

-- Configuration performance metrics
CREATE TABLE IF NOT EXISTS config_performance (
    config_name TEXT PRIMARY KEY,
    avg_score REAL,
    max_score_achieved INTEGER,
    min_score_achieved INTEGER,
    test_count INTEGER,
    success_rate REAL,
    avg_generation_time REAL
);
```

## Configuration Management

### Configuration Directory Structure

```
configs/
├── baseline/
│   ├── single_agent_basic.json
│   ├── single_agent_advanced.json
│   └── multi_cycle_basic.json
├── specialized/
│   ├── bug_focused.json
│   ├── edge_case_master.json
│   └── performance_optimizer.json
├── experimental/
│   ├── high_temperature_creative.json
│   ├── low_temperature_precise.json
│   └── mixed_model_ensemble.json
└── competition_winners/
    ├── top_scorer_v1.json
    ├── most_consistent.json
    └── fastest_solver.json
```

### Configuration Loader

```python
class ConfigurationManager:
    """Manages loading and validation of agent configurations"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.validation_errors = {}
    
    def load_all_configs(self) -> Dict[str, Dict]:
        """Load all JSON configurations from directory tree"""
        for json_file in self.config_dir.rglob("*.json"):
            relative_path = json_file.relative_to(self.config_dir)
            config_name = str(relative_path).replace('.json', '')
            
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                    
                # Validate configuration structure
                if self._validate_config(config):
                    self.configs[config_name] = config
                else:
                    self.validation_errors[config_name] = "Invalid structure"
                    
            except Exception as e:
                self.validation_errors[config_name] = str(e)
        
        return self.configs
    
    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration has required fields"""
        required_fields = ['ai_configuration']
        ai_config_fields = ['main_model', 'cycles', 'layer_agents']
        
        if not all(field in config for field in required_fields):
            return False
            
        ai_config = config.get('ai_configuration', {})
        return all(field in ai_config for field in ai_config_fields)
```

## Testing Pipeline

### Test Orchestrator

```python
class MultiAgentTestOrchestrator:
    """Orchestrates testing of multiple configurations"""
    
    def __init__(self, db_path: str = "competition.db"):
        self.db_path = db_path
        self.grader = FunctionQualityGrader()
        self.current_run_id = None
        
    async def run_batch_test(
        self, 
        configs: Dict[str, Dict],
        challenge_prompt: str,
        progress_callback=None
    ) -> str:
        """Run batch test on multiple configurations"""
        
        # Initialize test run
        self.current_run_id = self._create_test_run(len(configs))
        
        # Process configurations concurrently with limit
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for config_name, config in configs.items():
                future = executor.submit(
                    self._test_single_config,
                    config_name,
                    config,
                    challenge_prompt
                )
                futures.append((config_name, future))
            
            # Process results as they complete
            for config_name, future in futures:
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(config_name, result)
                        
                except Exception as e:
                    error_result = {
                        'config_name': config_name,
                        'error': str(e),
                        'total_score': 0
                    }
                    results.append(error_result)
        
        # Finalize test run
        self._finalize_test_run(self.current_run_id)
        return self.current_run_id
    
    def _test_single_config(
        self, 
        config_name: str,
        config: Dict,
        challenge_prompt: str
    ) -> Dict:
        """Test a single configuration"""
        
        start_time = time.time()
        
        try:
            # Extract AI configuration
            ai_config = config['ai_configuration']
            
            # Generate code using MOAgent
            generated_code = self._generate_code_with_config(
                ai_config, 
                challenge_prompt
            )
            
            # Extract function from response
            function_code = extract_function_code(generated_code)
            
            # Grade the function
            grading_result = self._grade_function(function_code)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Store result
            result = {
                'config_name': config_name,
                'config': config,
                'generated_code': function_code,
                'grading_result': grading_result,
                'generation_time': generation_time
            }
            
            self._store_result(result)
            return result
            
        except Exception as e:
            return {
                'config_name': config_name,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    def _generate_code_with_config(
        self, 
        ai_config: Dict,
        prompt: str
    ) -> str:
        """Generate code using MOAgent with specific configuration"""
        
        # Create MOAgent instance
        agent = MOAgent.from_config(
            main_model=ai_config.get('main_model', 'llama-3.3-70b'),
            system_prompt=ai_config.get('system_prompt'),
            cycles=ai_config.get('cycles', 1),
            layer_agent_config=self._convert_layer_agents(ai_config),
            temperature=ai_config.get('main_temperature', 0.1)
        )
        
        # Generate response
        response = ""
        for chunk in agent.chat(prompt):
            response += chunk
            
        return response
    
    def _convert_layer_agents(self, ai_config: Dict) -> Dict:
        """Convert layer agents from config format to MOAgent format"""
        layer_config = {}
        
        for i, agent in enumerate(ai_config.get('layer_agents', [])):
            layer_config[f'layer_agent_{i+1}'] = {
                'system_prompt': agent['prompt'],
                'model_name': agent['model'],
                'temperature': agent['temperature']
            }
            
        return layer_config
```

## Grading System Integration

### Enhanced Grader Wrapper

```python
class EnhancedGrader:
    """Wrapper around FunctionQualityGrader with additional metrics"""
    
    def __init__(self):
        self.base_grader = FunctionQualityGrader()
        
    def grade_with_metadata(
        self, 
        function_code: str,
        config_name: str
    ) -> Dict:
        """Grade function and add metadata"""
        
        try:
            # Create function from code string
            exec_globals = {}
            exec(function_code, exec_globals)
            
            # Find the function
            func = None
            for name, obj in exec_globals.items():
                if callable(obj) and name == 'calculate_user_metrics':
                    func = obj
                    break
            
            if not func:
                return {
                    'error': 'Function not found',
                    'total_score': 0,
                    'config_name': config_name
                }
            
            # Grade the function
            result = self.base_grader.test_function(func, config_name)
            
            # Add additional metrics
            result['config_name'] = config_name
            result['timestamp'] = time.time()
            
            # Calculate consistency score
            result['consistency_score'] = self._calculate_consistency(result)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'total_score': 0,
                'config_name': config_name
            }
    
    def _calculate_consistency(self, result: Dict) -> float:
        """Calculate how consistent the solution is across test categories"""
        scores = []
        for test_result in result.get('detailed_results', {}).values():
            if test_result['max'] > 0:
                scores.append(test_result['score'] / test_result['max'])
        
        if not scores:
            return 0.0
            
        # Lower variance = higher consistency
        import numpy as np
        variance = np.var(scores)
        consistency = 1.0 - min(variance, 1.0)
        return round(consistency * 100, 2)
```

## Streaming Application Design

### Main Testing Application

```python
# test_multi_agent_configs.py

import streamlit as st
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="Multi-Agent Configuration Tester",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    if 'current_run' not in st.session_state:
        st.session_state.current_run = None
    
    # Header
    st.title("🤖 Multi-Agent Configuration Testing Framework")
    st.markdown("Test and compare multiple AI agent configurations for code generation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Test Configuration")
        
        # Config directory selection
        config_dir = st.text_input(
            "Configuration Directory",
            value="configs/",
            help="Directory containing JSON configuration files"
        )
        
        # Challenge selection
        challenge_type = st.selectbox(
            "Challenge Type",
            ["User Metrics Calculator", "Custom Challenge"]
        )
        
        if challenge_type == "Custom Challenge":
            challenge_prompt = st.text_area(
                "Challenge Prompt",
                height=200
            )
        else:
            challenge_prompt = get_default_challenge_prompt()
        
        # Test controls
        if st.button("🚀 Start Batch Test", type="primary"):
            run_batch_test(config_dir, challenge_prompt)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Results", 
        "📈 Analysis", 
        "🏆 Leaderboard",
        "📝 Configuration Details"
    ])
    
    with tab1:
        render_live_results()
    
    with tab2:
        render_analysis()
    
    with tab3:
        render_leaderboard()
    
    with tab4:
        render_config_details()

def run_batch_test(config_dir: str, challenge_prompt: str):
    """Run batch test with streaming updates"""
    
    # Load configurations
    config_manager = ConfigurationManager(Path(config_dir))
    configs = config_manager.load_all_configs()
    
    if not configs:
        st.error("No valid configurations found!")
        return
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # Initialize orchestrator
    orchestrator = MultiAgentTestOrchestrator()
    
    # Progress callback
    completed = 0
    def update_progress(config_name, result):
        nonlocal completed
        completed += 1
        progress = completed / len(configs)
        progress_bar.progress(progress)
        status_text.text(f"Testing {config_name}... ({completed}/{len(configs)})")
        
        # Display result immediately
        with results_container:
            display_single_result(result)
    
    # Run tests
    with st.spinner("Running batch tests..."):
        run_id = orchestrator.run_batch_test(
            configs,
            challenge_prompt,
            progress_callback=update_progress
        )
    
    st.success(f"✅ Batch test completed! Run ID: {run_id}")
    st.balloons()

def render_live_results():
    """Display live testing results"""
    
    if not st.session_state.test_results:
        st.info("No test results yet. Start a batch test from the sidebar.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    results_df = pd.DataFrame(st.session_state.test_results)
    
    with col1:
        st.metric(
            "Configurations Tested",
            len(results_df)
        )
    
    with col2:
        avg_score = results_df['total_score'].mean()
        st.metric(
            "Average Score",
            f"{avg_score:.1f}"
        )
    
    with col3:
        max_score = results_df['total_score'].max()
        st.metric(
            "Best Score",
            f"{max_score}"
        )
    
    with col4:
        success_rate = (results_df['total_score'] > 72).mean() * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%"
        )
    
    # Results table
    st.subheader("📋 Test Results")
    
    # Format results for display
    display_df = results_df[[
        'config_name', 'total_score', 'percentage', 
        'grade', 'generation_time', 'consistency_score'
    ]].copy()
    
    display_df['generation_time'] = display_df['generation_time'].round(2)
    
    # Color code by grade
    def color_grade(val):
        colors = {
            'A': 'background-color: #28a745',
            'B': 'background-color: #17a2b8',
            'C': 'background-color: #ffc107',
            'D': 'background-color: #fd7e14',
            'F': 'background-color: #dc3545'
        }
        return colors.get(val, '')
    
    styled_df = display_df.style.applymap(
        color_grade, 
        subset=['grade']
    )
    
    st.dataframe(styled_df, use_container_width=True)

def render_analysis():
    """Render detailed analysis visualizations"""
    
    if not st.session_state.test_results:
        st.info("No results to analyze yet.")
        return
    
    results_df = pd.DataFrame(st.session_state.test_results)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Score Distribution")
        fig = px.histogram(
            results_df, 
            x='total_score',
            nbins=20,
            title="Distribution of Total Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Score vs Generation Time")
        fig = px.scatter(
            results_df,
            x='generation_time',
            y='total_score',
            color='grade',
            size='consistency_score',
            hover_data=['config_name'],
            title="Performance vs Speed Trade-off"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    st.subheader("📈 Category Performance")
    
    # Extract category scores
    category_data = []
    for _, row in results_df.iterrows():
        if 'detailed_results' in row and row['detailed_results']:
            for category, scores in row['detailed_results'].items():
                category_data.append({
                    'config_name': row['config_name'],
                    'category': category,
                    'score': scores['score'],
                    'max_score': scores['max']
                })
    
    if category_data:
        cat_df = pd.DataFrame(category_data)
        cat_pivot = cat_df.pivot_table(
            index='config_name',
            columns='category',
            values='score'
        )
        
        fig = px.imshow(
            cat_pivot,
            title="Category Scores Heatmap",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance comparison
    st.subheader("🤖 Model Performance")
    
    # Extract model usage
    model_scores = {}
    for _, row in results_df.iterrows():
        if 'config' in row:
            model = row['config']['ai_configuration']['main_model']
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(row['total_score'])
    
    model_df = pd.DataFrame([
        {'model': model, 'avg_score': np.mean(scores), 'count': len(scores)}
        for model, scores in model_scores.items()
    ])
    
    fig = px.bar(
        model_df,
        x='model',
        y='avg_score',
        title="Average Score by Model",
        text='count'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_leaderboard():
    """Display configuration leaderboard"""
    
    conn = sqlite3.connect("competition.db")
    
    # Get aggregated performance data
    query = """
    SELECT 
        config_name,
        AVG(total_score) as avg_score,
        MAX(total_score) as best_score,
        COUNT(*) as attempts,
        AVG(generation_time) as avg_time
    FROM config_test_results
    GROUP BY config_name
    ORDER BY avg_score DESC
    LIMIT 20
    """
    
    leaderboard_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if leaderboard_df.empty:
        st.info("No leaderboard data available yet.")
        return
    
    st.subheader("🏆 Top Performing Configurations")
    
    # Add rank
    leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
    
    # Display with custom formatting
    for idx, row in leaderboard_df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        
        with col1:
            if row['rank'] <= 3:
                medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                st.markdown(f"# {medals[row['rank']]}")
            else:
                st.markdown(f"### #{row['rank']}")
        
        with col2:
            st.markdown(f"**{row['config_name']}**")
            st.caption(f"Best: {row['best_score']} | Attempts: {row['attempts']}")
        
        with col3:
            st.metric("Average Score", f"{row['avg_score']:.1f}")
        
        with col4:
            st.metric("Avg Time", f"{row['avg_time']:.1f}s")
        
        st.divider()

def render_config_details():
    """Show detailed configuration information"""
    
    if not st.session_state.test_results:
        st.info("No configurations tested yet.")
        return
    
    results_df = pd.DataFrame(st.session_state.test_results)
    
    # Configuration selector
    config_name = st.selectbox(
        "Select Configuration",
        results_df['config_name'].unique()
    )
    
    # Get configuration details
    config_data = results_df[results_df['config_name'] == config_name].iloc[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Configuration")
        st.json(config_data['config'])
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        metrics = {
            "Total Score": config_data['total_score'],
            "Percentage": f"{config_data['percentage']}%",
            "Grade": config_data['grade'],
            "Generation Time": f"{config_data['generation_time']:.2f}s",
            "Consistency Score": f"{config_data['consistency_score']}%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    # Generated code
    st.subheader("💻 Generated Code")
    st.code(config_data['generated_code'], language='python')
    
    # Detailed test results
    st.subheader("🔍 Detailed Test Results")
    
    if 'execution_logs' in config_data:
        with st.expander("Execution Logs"):
            for log in config_data['execution_logs']:
                st.text(log)

if __name__ == "__main__":
    main()
```

## Batch Testing Implementation

### Configuration Examples Directory

Create a set of test configurations to demonstrate different strategies:

```python
# generate_test_configs.py

def generate_test_configurations():
    """Generate a variety of test configurations"""
    
    configs = {
        # Baseline configurations
        "baseline/single_agent_simple": {
            "ai_configuration": {
                "main_model": "llama3.1-8b",
                "main_temperature": 0.1,
                "cycles": 1,
                "system_prompt": "You are a Python expert. Fix all bugs in the code.",
                "layer_agents": [{
                    "name": "debugger",
                    "model": "llama3.1-8b",
                    "temperature": 0.1,
                    "prompt": "Debug the Python code and fix all issues."
                }]
            }
        },
        
        # Specialized bug hunter
        "specialized/bug_hunter_focused": {
            "ai_configuration": {
                "main_model": "llama-3.3-70b",
                "main_temperature": 0.0,
                "cycles": 2,
                "system_prompt": "You are a bug detection expert. Fix: division by zero, KeyError, wrong calculations, sorting issues.",
                "layer_agents": [
                    {
                        "name": "critical_bugs",
                        "model": "llama-4-scout-17b-16e-instruct",
                        "temperature": 0.0,
                        "prompt": "Find and fix critical bugs: division by zero, missing keys, wrong denominators."
                    },
                    {
                        "name": "edge_cases",
                        "model": "qwen-3-32b",
                        "temperature": 0.1,
                        "prompt": "Handle edge cases: empty lists, no active users, invalid inputs."
                    }
                ]
            }
        },
        
        # High-performance configuration
        "optimized/performance_master": {
            "ai_configuration": {
                "main_model": "llama-3.3-70b",
                "main_temperature": 0.1,
                "cycles": 3,
                "system_prompt": "Create efficient, bug-free Python code with optimal performance.",
                "layer_agents": [
                    {
                        "name": "analyzer",
                        "model": "llama-4-scout-17b-16e-instruct",
                        "temperature": 0.1,
                        "prompt": "Analyze the requirements and identify all potential issues."
                    },
                    {
                        "name": "implementer",
                        "model": "llama-3.3-70b",
                        "temperature": 0.2,
                        "prompt": "Implement a robust solution handling all edge cases."
                    },
                    {
                        "name": "optimizer",
                        "model": "qwen-3-32b",
                        "temperature": 0.0,
                        "prompt": "Optimize the code for performance and correctness."
                    }
                ]
            }
        },
        
        # Experimental high-temperature
        "experimental/creative_solver": {
            "ai_configuration": {
                "main_model": "qwen-3-32b",
                "main_temperature": 0.7,
                "cycles": 2,
                "system_prompt": "Think creatively to solve all bugs and edge cases.",
                "layer_agents": [
                    {
                        "name": "creative_thinker",
                        "model": "qwen-3-32b",
                        "temperature": 0.8,
                        "prompt": "Think outside the box to identify and fix issues."
                    },
                    {
                        "name": "validator",
                        "model": "llama3.1-8b",
                        "temperature": 0.0,
                        "prompt": "Validate the solution and ensure correctness."
                    }
                ]
            }
        }
    }
    
    # Save configurations
    import os
    for path, config in configs.items():
        dir_path = os.path.dirname(f"configs/{path}.json")
        os.makedirs(dir_path, exist_ok=True)
        
        with open(f"configs/{path}.json", 'w') as f:
            json.dump(config, f, indent=2)
```

## Results Analysis and Visualization

### Analysis Dashboard Components

```python
class ResultsAnalyzer:
    """Analyze and visualize test results"""
    
    def __init__(self, db_path: str = "competition.db"):
        self.db_path = db_path
    
    def get_performance_trends(self, config_name: str) -> pd.DataFrame:
        """Get performance trends over time for a configuration"""
        
        query = """
        SELECT 
            test_timestamp,
            total_score,
            percentage,
            generation_time
        FROM config_test_results
        WHERE config_name = ?
        ORDER BY test_timestamp
        """
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=[config_name])
        conn.close()
        
        return df
    
    def get_category_breakdown(self, run_id: str) -> pd.DataFrame:
        """Get category score breakdown for a test run"""
        
        query = """
        SELECT 
            config_name,
            detailed_results
        FROM config_test_results
        WHERE run_id = ?
        """
        
        conn = sqlite3.connect(self.db_path)
        results = pd.read_sql_query(query, conn, params=[run_id])
        conn.close()
        
        # Parse detailed results
        category_data = []
        for _, row in results.iterrows():
            details = json.loads(row['detailed_results'])
            for category, scores in details.items():
                category_data.append({
                    'config_name': row['config_name'],
                    'category': category,
                    'score': scores['score'],
                    'max_score': scores['max'],
                    'percentage': (scores['score'] / scores['max'] * 100) if scores['max'] > 0 else 0
                })
        
        return pd.DataFrame(category_data)
    
    def generate_report(self, run_id: str) -> Dict:
        """Generate comprehensive test report"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get run summary
        run_query = """
        SELECT * FROM config_test_runs WHERE run_id = ?
        """
        run_data = pd.read_sql_query(run_query, conn, params=[run_id]).iloc[0]
        
        # Get results
        results_query = """
        SELECT 
            config_name,
            total_score,
            max_score,
            percentage,
            grade,
            generation_time
        FROM config_test_results
        WHERE run_id = ?
        ORDER BY total_score DESC
        """
        results = pd.read_sql_query(results_query, conn, params=[run_id])
        
        conn.close()
        
        report = {
            'run_id': run_id,
            'timestamp': run_data['run_timestamp'],
            'total_configs': run_data['total_configs'],
            'summary_stats': {
                'avg_score': results['total_score'].mean(),
                'max_score': results['total_score'].max(),
                'min_score': results['total_score'].min(),
                'std_dev': results['total_score'].std(),
                'success_rate': (results['percentage'] >= 60).mean() * 100
            },
            'grade_distribution': results['grade'].value_counts().to_dict(),
            'top_performers': results.head(5).to_dict('records'),
            'fastest_solvers': results.nsmallest(5, 'generation_time')[['config_name', 'generation_time', 'total_score']].to_dict('records')
        }
        
        return report
```

### Visualization Components

```python
def create_performance_dashboard(run_id: str):
    """Create comprehensive performance dashboard"""
    
    analyzer = ResultsAnalyzer()
    
    # Get data
    category_df = analyzer.get_category_breakdown(run_id)
    report = analyzer.generate_report(run_id)
    
    # Create visualizations
    figs = {}
    
    # 1. Category performance radar chart
    categories = category_df['category'].unique()
    
    fig_radar = go.Figure()
    
    for config in category_df['config_name'].unique():
        config_data = category_df[category_df['config_name'] == config]
        
        values = []
        for cat in categories:
            cat_data = config_data[config_data['category'] == cat]
            if not cat_data.empty:
                values.append(cat_data['percentage'].iloc[0])
            else:
                values.append(0)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=config
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Category Performance Comparison"
    )
    
    figs['radar'] = fig_radar
    
    # 2. Score distribution violin plot
    conn = sqlite3.connect("competition.db")
    scores_df = pd.read_sql_query(
        "SELECT config_name, total_score FROM config_test_results WHERE run_id = ?",
        conn,
        params=[run_id]
    )
    conn.close()
    
    fig_violin = px.violin(
        scores_df,
        y='total_score',
        x='config_name',
        title="Score Distribution by Configuration",
        box=True
    )
    
    figs['violin'] = fig_violin
    
    # 3. Performance vs Time scatter
    fig_scatter = px.scatter(
        x=[r['generation_time'] for r in report['fastest_solvers']],
        y=[r['total_score'] for r in report['fastest_solvers']],
        text=[r['config_name'] for r in report['fastest_solvers']],
        title="Performance vs Generation Time"
    )
    
    fig_scatter.update_traces(textposition='top center')
    figs['scatter'] = fig_scatter
    
    return figs, report
```

## Performance Optimization

### Optimization Strategies

```python
class PerformanceOptimizer:
    """Optimize configuration testing performance"""
    
    @staticmethod
    def optimize_context_usage(config: Dict) -> Dict:
        """Optimize configuration to reduce context usage"""
        
        optimized = copy.deepcopy(config)
        ai_config = optimized['ai_configuration']
        
        # Reduce cycles if too high
        if ai_config.get('cycles', 1) > 3:
            ai_config['cycles'] = 3
        
        # Shorten prompts
        for agent in ai_config.get('layer_agents', []):
            if len(agent['prompt']) > 200:
                # Summarize long prompts
                agent['prompt'] = agent['prompt'][:200] + "..."
        
        # Use smaller models for layer agents if main model is large
        if ai_config['main_model'] == 'llama-3.3-70b':
            for agent in ai_config.get('layer_agents', []):
                if agent['model'] == 'llama-3.3-70b':
                    agent['model'] = 'llama3.1-8b'
        
        return optimized
    
    @staticmethod
    def batch_configs_by_similarity(configs: Dict[str, Dict]) -> List[List[str]]:
        """Group similar configurations for efficient testing"""
        
        # Group by main model and cycles
        groups = {}
        for name, config in configs.items():
            ai_config = config['ai_configuration']
            key = (ai_config['main_model'], ai_config['cycles'])
            
            if key not in groups:
                groups[key] = []
            groups[key].append(name)
        
        # Convert to list of batches
        batches = list(groups.values())
        return batches
```

### Caching and Memoization

```python
from functools import lru_cache
import hashlib

class CachedTester:
    """Cache test results for identical configurations"""
    
    def __init__(self):
        self.cache = {}
    
    def _get_config_hash(self, config: Dict) -> str:
        """Generate hash for configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_cached_result(self, config: Dict) -> Optional[Dict]:
        """Get cached result if available"""
        config_hash = self._get_config_hash(config)
        return self.cache.get(config_hash)
    
    def cache_result(self, config: Dict, result: Dict):
        """Cache test result"""
        config_hash = self._get_config_hash(config)
        self.cache[config_hash] = result
```

## Example Configurations

### High-Scoring Configuration Examples

```json
// configs/competition_winners/top_scorer_v1.json
{
  "ai_configuration": {
    "main_model": "llama-3.3-70b",
    "main_temperature": 0.1,
    "cycles": 2,
    "system_prompt": "You are a Python debugging expert. Fix all bugs while maintaining code structure. Focus on: division by zero, missing keys, wrong calculations, sorting issues, edge cases. {helper_response}",
    "layer_agents": [
      {
        "name": "bug_hunter",
        "model": "llama-4-scout-17b-16e-instruct",
        "temperature": 0.0,
        "prompt": "Identify and fix critical bugs: division by zero (check days_active), KeyError (validate dict keys), wrong denominator (use active_users not all users). {helper_response}"
      },
      {
        "name": "edge_case_specialist",
        "model": "qwen-3-32b",
        "temperature": 0.1,
        "prompt": "Handle edge cases: empty users list, no active users in date range, less than 5 users for top performers, invalid date ranges. {helper_response}"
      },
      {
        "name": "performance_optimizer",
        "model": "llama3.1-8b",
        "temperature": 0.0,
        "prompt": "Optimize performance: efficient sorting (use sorted with reverse=True), proper list slicing [:5], avoid mutations. {helper_response}"
      }
    ]
  },
  "metadata": {
    "author": "competition_winner",
    "score_achieved": 120,
    "consistency": "high",
    "notes": "Achieves perfect score by addressing all test categories systematically"
  }
}
```

### Fast Solver Configuration

```json
// configs/optimized/speed_focused.json
{
  "ai_configuration": {
    "main_model": "llama3.1-8b",
    "main_temperature": 0.0,
    "cycles": 1,
    "system_prompt": "Fix this Python function. Handle: division by zero (days_active), missing keys, empty lists, wrong average calculation (use active users count), correct sorting (descending). Return exactly the calculate_user_metrics function. {helper_response}",
    "layer_agents": [
      {
        "name": "fixer",
        "model": "llama3.1-8b",
        "temperature": 0.0,
        "prompt": "Fix all bugs in one pass. Check days_active > 0, use .get() for dict access, handle empty lists, fix denominator, sort descending. {helper_response}"
      }
    ]
  },
  "metadata": {
    "optimization": "speed",
    "avg_generation_time": "8.5s",
    "avg_score": 95
  }
}
```

## Best Practices

### Configuration Design

1. **Layer Agent Specialization**
   - Each agent should have a specific focus
   - Avoid overlapping responsibilities
   - Use appropriate models for each task

2. **Temperature Settings**
   - Use low temperatures (0.0-0.2) for consistency
   - Higher temperatures only for creative problem-solving
   - Main model should typically have lower temperature

3. **Prompt Engineering**
   - Be specific about bugs to fix
   - Reference exact issues from test cases
   - Use the `{helper_response}` placeholder effectively

4. **Cycle Management**
   - 1-2 cycles for simple fixes
   - 3+ cycles for complex problems
   - Balance quality vs. generation time

### Testing Strategy

1. **Batch Organization**
   - Group similar configurations
   - Test variations systematically
   - Use control configurations for baseline

2. **Performance Monitoring**
   - Track generation times
   - Monitor token usage
   - Identify bottlenecks

3. **Result Analysis**
   - Compare across multiple runs
   - Look for consistency patterns
   - Identify optimal trade-offs

This framework provides a comprehensive system for testing and comparing multiple AI agent configurations, enabling systematic optimization of code generation quality while maintaining performance and efficiency. 