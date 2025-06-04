#!/usr/bin/env python3
"""
JSON Scoring Application - Textual UI for MOA Competition
Batch-style scorer that evaluates multiple JSON agent-configuration files 
against the Competition goal.md specification.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Import MOA components
from moa.agent import MOAgent
from grader import FunctionQualityGrader
from competitive_programming import validate_and_execute_code, SecureExecutionError
import re

# Configuration for the scoring mission
MISSION_PROMPT = """
Write a Python function called `calculate_user_metrics` that processes user engagement data.

The function should:
1. Take parameters: users (list of dicts), start_date (str), end_date (str)
2. Filter users by date range (last_login between start_date and end_date)
3. Calculate engagement scores using: (posts * 2 + comments * 1.5 + likes * 0.1) / days_active
4. Handle edge cases: empty lists, missing keys, division by zero
5. Return dict with: average_engagement, top_performers (top 5), active_count

Your function must handle all edge cases gracefully and return proper data types.
"""

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class BatchScorer:
    """Main batch scoring controller"""
    
    def __init__(self, config_dir: str = "configs", output_file: str = "scores.csv"):
        self.config_dir = Path(config_dir)
        self.output_file = output_file
        self.grader = FunctionQualityGrader()
        self.results = []
        
        print(f"{Colors.BLUE}{Colors.BOLD}🚀 MOA Competition Batch Scorer{Colors.END}")
        print(f"📁 Config directory: {self.config_dir}")
        print(f"📊 Output file: {self.output_file}")
        print(f"🎯 Max possible score: {self.grader.max_possible_score}")
        
    def load_json_configs(self) -> List[Dict[str, Any]]:
        """Load all JSON configuration files from the config directory"""
        configs = []
        
        if not self.config_dir.exists():
            print(f"{Colors.RED}❌ Config directory '{self.config_dir}' not found{Colors.END}")
            return configs
            
        json_files = list(self.config_dir.glob("*.json"))
        if not json_files:
            print(f"{Colors.YELLOW}⚠️ No JSON files found in '{self.config_dir}'{Colors.END}")
            return configs
            
        print(f"\n📁 Found {len(json_files)} configuration files:")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                    configs.append({
                        'file_path': str(json_file),
                        'file_name': json_file.name,
                        'config': config
                    })
                print(f"  ✅ {json_file.name}")
            except Exception as e:
                print(f"  ❌ {json_file.name} - Error: {str(e)}")
                
        return configs
    
    def normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize arbitrary JSON config to MOAgent format"""
        # Default values
        normalized = {
            'main_model': 'llama3.3-70b',
            'system_prompt': None,
            'cycles': 1,
            'layer_agent_config': {},
            'reference_system_prompt': None,
            'temperature': 0.1,
            'max_tokens': None
        }
        
        # Handle ai_configuration wrapper structure
        if 'ai_configuration' in config:
            ai_config = config['ai_configuration']
            
            # Map fields from ai_configuration
            if 'main_model' in ai_config:
                normalized['main_model'] = ai_config['main_model']
            if 'main_temperature' in ai_config:
                normalized['temperature'] = ai_config['main_temperature']
            if 'cycles' in ai_config:
                normalized['cycles'] = ai_config['cycles']
            if 'system_prompt' in ai_config:
                normalized['system_prompt'] = ai_config['system_prompt']
                
            # Handle layer_agents array format
            if 'layer_agents' in ai_config:
                layer_agent_config = {}
                for agent in ai_config['layer_agents']:
                    agent_name = agent.get('name', f'agent_{len(layer_agent_config) + 1}')
                    layer_agent_config[agent_name] = {
                        'model_name': agent.get('model', 'llama3.1-8b'),
                        'temperature': agent.get('temperature', 0.1),
                        'system_prompt': agent.get('prompt', 'You are a helpful assistant. {helper_response}')
                    }
                normalized['layer_agent_config'] = layer_agent_config
        else:
            # Handle original format - Map common field variations
            field_mappings = {
                'model': 'main_model',
                'primary_model': 'main_model',
                'main': 'main_model',
                'temp': 'temperature',
                'layers': 'layer_agent_config',
                'agents': 'layer_agent_config',
                'prompt': 'system_prompt',
                'system': 'system_prompt',
                'max_length': 'max_tokens',
                'context_length': 'max_tokens'
            }
            
            # Apply mappings
            for old_key, new_key in field_mappings.items():
                if old_key in config:
                    normalized[new_key] = config[old_key]
            
            # Direct field updates
            for key in normalized.keys():
                if key in config:
                    normalized[key] = config[key]
                
        return normalized
    
    def extract_function_code(self, text: str) -> str:
        """Extract Python function code from model response"""
        # Try to find code blocks first
        code_block_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`([^`]*calculate_user_metrics[^`]*)`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if 'def calculate_user_metrics' in code:
                    return code
        
        # Try to find function definition directly
        lines = text.split('\n')
        in_function = False
        function_lines = []
        indent_level = 0
        
        for line in lines:
            if 'def calculate_user_metrics' in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                function_lines.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() == '':
                    function_lines.append(line)
                elif current_indent > indent_level or (current_indent == indent_level and line.strip().startswith(('return', 'pass', 'raise', 'break', 'continue'))):
                    function_lines.append(line)
                elif current_indent <= indent_level and line.strip():
                    break
                else:
                    function_lines.append(line)
        
        if function_lines:
            return '\n'.join(function_lines)
        
        # Fallback: return the original text
        return text
    
    def execute_moa_config(self, config: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """Execute a single MOA configuration and collect the answer"""
        start_time = time.perf_counter()
        result = {
            'file_name': file_name,
            'status': 'unknown',
            'score': 0,
            'max_score': self.grader.max_possible_score,
            'duration_s': 0,
            'error': None,
            'function_extracted': False,
            'grading_details': {}
        }
        
        try:
            print(f"\n{Colors.BLUE}🔄 Processing {file_name}...{Colors.END}")
            
            # Normalize configuration
            normalized_config = self.normalize_config(config)
            print(f"  📋 Model: {normalized_config['main_model']}")
            print(f"  🔄 Cycles: {normalized_config['cycles']}")
            print(f"  🌡️ Temperature: {normalized_config['temperature']}")
            
            # Create MOAgent
            moa = MOAgent.from_config(**normalized_config)
            
            # Execute the mission
            print(f"  🤖 Executing MOA stack...")
            response_chunks = []
            for chunk in moa.chat(MISSION_PROMPT):
                response_chunks.append(chunk)
            
            full_response = ''.join(response_chunks)
            print(f"  📝 Generated {len(full_response)} characters")
            
            # Extract function from response
            try:
                # First try to extract just the function code
                extracted_code = self.extract_function_code(full_response)
                print(f"  🔍 Extracted {len(extracted_code)} characters of code")
                
                user_function, _ = validate_and_execute_code(extracted_code)
                result['function_extracted'] = True
                print(f"  ✅ Function extracted successfully")
                
                # Grade the function
                grading_results = self.grader.test_function(user_function, file_name)
                result['score'] = grading_results['total_score']
                result['grading_details'] = grading_results
                result['status'] = 'completed'
                
                print(f"  🎯 Score: {result['score']}/{result['max_score']} ({result['score']/result['max_score']*100:.1f}%)")
                
            except SecureExecutionError as e:
                result['error'] = f"Function extraction failed: {str(e)}"
                result['status'] = 'extraction_failed'
                print(f"  ❌ Function extraction failed: {str(e)}")
                
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result['status'] = 'execution_failed'
            print(f"  ❌ Execution error: {str(e)}")
            
        finally:
            result['duration_s'] = time.perf_counter() - start_time
            print(f"  ⏱️ Completed in {result['duration_s']:.2f}s")
            
        return result
    
    def run_batch_scoring(self, max_workers: int = 2) -> List[Dict[str, Any]]:
        """Run batch scoring on all configurations"""
        configs = self.load_json_configs()
        
        if not configs:
            print(f"{Colors.RED}❌ No configurations to process{Colors.END}")
            return []
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 Starting batch execution with {max_workers} workers{Colors.END}")
        
        if max_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(self.execute_moa_config, cfg['config'], cfg['file_name']): cfg
                    for cfg in configs
                }
                
                for future in as_completed(future_to_config):
                    result = future.result()
                    self.results.append(result)
        else:
            # Sequential execution
            for cfg in configs:
                result = self.execute_moa_config(cfg['config'], cfg['file_name'])
                self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed scoring report"""
        if not self.results:
            print(f"{Colors.YELLOW}⚠️ No results to report{Colors.END}")
            return pd.DataFrame()
        
        # Create DataFrame
        df_data = []
        for result in self.results:
            row = {
                'file_name': result['file_name'],
                'status': result['status'],
                'score': result['score'],
                'max_score': result['max_score'],
                'percentage': round((result['score'] / result['max_score']) * 100, 1),
                'duration_s': round(result['duration_s'], 2),
                'function_extracted': result['function_extracted'],
                'error': result.get('error', ''),
            }
            
            # Add detailed scoring breakdown if available
            if 'grading_details' in result and result['grading_details']:
                details = result['grading_details'].get('detailed_results', {})
                row.update({
                    'tests_passed': result['grading_details'].get('tests_passed', 0),
                    'tests_failed': result['grading_details'].get('tests_failed', 0),
                    'critical_bugs_score': sum([
                        details.get('handles_division_by_zero', {}).get('score', 0),
                        details.get('handles_empty_users_list', {}).get('score', 0),
                        details.get('handles_missing_keys', {}).get('score', 0),
                        details.get('correct_average_calculation', {}).get('score', 0),
                    ]),
                    'logic_bugs_score': sum([
                        details.get('correct_sorting_direction', {}).get('score', 0),
                        details.get('handles_no_active_users', {}).get('score', 0),
                        details.get('doesnt_mutate_input', {}).get('score', 0),
                    ]),
                    'edge_cases_score': sum([
                        details.get('handles_less_than_5_users', {}).get('score', 0),
                        details.get('handles_invalid_dates', {}).get('score', 0),
                        details.get('robust_error_handling', {}).get('score', 0),
                    ]),
                    'performance_score': details.get('efficient_implementation', {}).get('score', 0),
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by score (descending)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        return df
    
    def print_leaderboard(self, df: pd.DataFrame):
        """Print a formatted leaderboard to console"""
        if df.empty:
            return
            
        print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 COMPETITION LEADERBOARD{Colors.END}")
        print("=" * 80)
        
        # Header
        print(f"{'Rank':<4} {'File Name':<25} {'Score':<8} {'%':<6} {'Status':<12} {'Time(s)':<8}")
        print("-" * 80)
        
        # Results
        for _, row in df.head(10).iterrows():  # Top 10
            rank = row['rank']
            name = row['file_name'][:24]  # Truncate long names
            score = f"{row['score']}/{row['max_score']}"
            percentage = f"{row['percentage']}%"
            status = row['status'][:11]  # Truncate long status
            duration = f"{row['duration_s']}"
            
            # Color coding based on performance
            if row['percentage'] >= 90:
                color = Colors.GREEN
            elif row['percentage'] >= 70:
                color = Colors.YELLOW
            else:
                color = Colors.RED
                
            print(f"{color}{rank:<4} {name:<25} {score:<8} {percentage:<6} {status:<12} {duration:<8}{Colors.END}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        total_configs = len(df)
        successful = len(df[df['status'] == 'completed'])
        avg_score = df['score'].mean()
        best_score = df['score'].max()
        
        print(f"📊 {Colors.BOLD}Summary:{Colors.END}")
        print(f"   • Total configurations: {total_configs}")
        print(f"   • Successful executions: {successful}")
        print(f"   • Average score: {avg_score:.1f}/{df['max_score'].iloc[0]}")
        print(f"   • Best score: {best_score}/{df['max_score'].iloc[0]}")
        print(f"   • Success rate: {(successful/total_configs)*100:.1f}%")
    
    def save_results(self, df: pd.DataFrame):
        """Save results to CSV file"""
        if df.empty:
            return
            
        try:
            df.to_csv(self.output_file, index=False)
            print(f"\n{Colors.GREEN}✅ Results saved to {self.output_file}{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Failed to save results: {str(e)}{Colors.END}")

def create_sample_configs():
    """Create sample configuration files for testing"""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    sample_configs = [
        {
            "name": "simple_single_agent.json",
            "config": {
                "main_model": "llama3.3-70b",
                "temperature": 0.1,
                "cycles": 1,
                "system_prompt": "You are an expert Python programmer. Write clean, robust code.",
                "layer_agent_config": {}
            }
        },
        {
            "name": "multi_agent_diverse.json", 
            "config": {
                "main_model": "llama3.3-70b",
                "temperature": 0.2,
                "cycles": 2,
                "layer_agent_config": {
                    "planner": {
                        "system_prompt": "Plan the solution step by step. {helper_response}",
                        "model_name": "llama3.1-8b",
                        "temperature": 0.3
                    },
                    "critic": {
                        "system_prompt": "Find potential bugs and edge cases. {helper_response}",
                        "model_name": "llama-4-scout-17b-16e-instruct", 
                        "temperature": 0.1
                    },
                    "optimizer": {
                        "system_prompt": "Optimize for performance and robustness. {helper_response}",
                        "model_name": "qwen-3-32b",
                        "temperature": 0.2
                    }
                }
            }
        },
        {
            "name": "high_temperature_creative.json",
            "config": {
                "main_model": "llama3.3-70b",
                "temperature": 0.7,
                "cycles": 1,
                "system_prompt": "Think creatively about edge cases and robust solutions."
            }
        }
    ]
    
    for sample in sample_configs:
        file_path = configs_dir / sample["name"]
        if not file_path.exists():
            with open(file_path, 'w') as f:
                json.dump(sample["config"], f, indent=2)
            print(f"✅ Created sample config: {file_path}")

def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="MOA Competition Batch Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_score.py --config_dir configs/ --output scores.csv
  python batch_score.py --workers 4 --create-samples
  python batch_score.py --config_dir my_configs/ --workers 1
        """
    )
    
    parser.add_argument(
        '--config_dir', 
        default='configs',
        help='Directory containing JSON configuration files (default: configs)'
    )
    parser.add_argument(
        '--output',
        default='scores.csv', 
        help='Output CSV file for results (default: scores.csv)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel workers (default: 2, use 1 for sequential)'
    )
    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample configuration files in the config directory'
    )
    
    args = parser.parse_args()
    
    # Create sample configs if requested
    if args.create_samples:
        print(f"{Colors.BLUE}📁 Creating sample configuration files...{Colors.END}")
        create_sample_configs()
        print()
    
    # Check for required environment variable
    if not os.getenv('CEREBRAS_API_KEY'):
        print(f"{Colors.RED}❌ CEREBRAS_API_KEY environment variable not set{Colors.END}")
        print("   Please export your Cerebras API key:")
        print("   export CEREBRAS_API_KEY=your_key_here")
        sys.exit(1)
    
    # Initialize scorer
    scorer = BatchScorer(config_dir=args.config_dir, output_file=args.output)
    
    try:
        # Run batch scoring
        results = scorer.run_batch_scoring(max_workers=args.workers)
        
        # Generate and display report
        df = scorer.generate_report()
        scorer.print_leaderboard(df)
        scorer.save_results(df)
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Batch scoring completed!{Colors.END}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {str(e)}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
