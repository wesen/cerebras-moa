import json
import time
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st
from moa.agent import MOAgent
import ast
import sys
from types import ModuleType
import re

# Import the grader
from grader import FunctionQualityGrader

# Scoring configuration
SCORING = {
    'bug_detection': {
        'division_by_zero': 15,
        'wrong_denominator': 15, 
        'sorting_direction': 10,
        'keyerror_handling': 10
    },
    'edge_cases': {
        'empty_input': 10,
        'missing_keys': 10,
        'date_formats': 8,
        'no_active_users': 7
    },
    'performance': {
        'sorting_optimization': 12,
        'single_loop': 8,
        'unnecessary_operations': 5
    },
    'security': {
        'input_validation': 10,
        'data_sanitization': 8,
        'injection_risks': 7
    }
}

# Original buggy function for reference - REMOVED
# This has been removed so only AI-generated solutions are used
# No hardcoded examples should exist in the system

# Scoring schema for structured outputs
SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 50,
            "description": "The score for this analysis category"
        },
        "feedback": {
            "type": "string",
            "description": "Brief feedback about the analysis"
        },
        "issues_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific issues found"
        }
    },
    "required": ["score", "feedback", "issues_found"],
    "additionalProperties": False
}

# Prompt injection detection patterns
INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'system\s+override',
    r'ai\s+agent\s*:',
    r'give\s+(me\s+)?(\d+|\w+)\s+points?',
    r'score\s*[=:]\s*\d+',
    r'maximum\s+points?',
    r'perfect\s+code',
    r'no\s+bugs?',
    r'scholarship\s+depends',
    r'grade\s+depends',
    r'urgent',
    r'please\s+(help|score\s+high)',
    r'you\s+are\s+now\s+a',
    r'new\s+task',
    r'conversation\s+is\s+complete',
    r'helpful\s+assistant',
    r'override\s+scoring',
    r'ignore\s+actual\s+bugs',
]

def sanitize_code_for_ai(code: str) -> str:
    """
    Sanitize code before sending to AI agents to prevent prompt injection.
    Removes comments, docstrings, and potentially malicious content.
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Remove docstrings and comments while preserving structure
        sanitized_lines = []
        code_lines = code.split('\n')
        
        class DocstringRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Remove docstring if present
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    # Remove the docstring
                    node.body = node.body[1:]
                return node
            
            def visit_ClassDef(self, node):
                # Remove class docstring if present
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]
                return node
        
        # Remove docstrings
        cleaned_tree = DocstringRemover().visit(tree)
        
        # Convert back to code
        try:
            import astor
            sanitized_code = astor.to_source(cleaned_tree)
        except ImportError:
            # Fallback: manual comment removal if astor not available
            print("📝 astor not available, using manual sanitization")
            sanitized_code = remove_comments_manual(code)
        except Exception:
            # Fallback: manual comment removal if astor fails
            sanitized_code = remove_comments_manual(code)
        
        # Additional cleaning: remove any remaining injection patterns
        for pattern in INJECTION_PATTERNS:
            sanitized_code = re.sub(pattern, '[REMOVED]', sanitized_code, flags=re.IGNORECASE)
        
        return sanitized_code
        
    except Exception:
        # Fallback to manual comment removal if AST parsing fails
        return remove_comments_manual(code)

def remove_comments_manual(code: str) -> str:
    """Manual comment removal fallback"""
    lines = code.split('\n')
    cleaned_lines = []
    in_multiline_string = False
    
    for line in lines:
        # Basic comment removal (this is a simplified approach)
        if '"""' in line or "'''" in line:
            in_multiline_string = not in_multiline_string
            continue
        
        if in_multiline_string:
            continue
            
        # Remove single-line comments but preserve strings
        if '#' in line:
            # Simple approach: remove everything after # 
            # (This could break if # is in a string, but it's a security tradeoff)
            line = line.split('#')[0].rstrip()
        
        # Check for injection patterns in this line
        line_lower = line.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, line_lower):
                line = '[POTENTIALLY MALICIOUS CONTENT REMOVED]'
                break
        
        if line.strip():  # Only add non-empty lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def detect_prompt_injection(code: str) -> List[str]:
    """Detect potential prompt injection attempts in code"""
    violations = []
    code_lower = code.lower()
    
    for pattern in INJECTION_PATTERNS:
        matches = re.findall(pattern, code_lower, re.IGNORECASE)
        if matches:
            violations.append(f"Potential prompt injection detected: pattern '{pattern}' found")
    
    return violations

# AI Analysis Prompts (COMPLETELY REMOVED - Now using FunctionQualityGrader only)
# All AI agent functionality has been replaced with the deterministic grader from grader.py
# for consistent, reliable scoring without the variability of language models.

# Secure execution environment
ALLOWED_BUILTINS = {
    'len', 'max', 'min', 'sum', 'abs', 'round', 'sorted', 'reversed', 'enumerate', 'zip',
    'range', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'type',
    'isinstance', 'hasattr', 'getattr', 'setattr', 'all', 'any', 'filter', 'map',
    # Exception types for proper error handling
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError',
    'ZeroDivisionError', 'OverflowError', 'RuntimeError'
}

ALLOWED_MODULES = {
    'datetime': ['datetime', 'date', 'time', 'timedelta'],
    'heapq': ['nlargest', 'nsmallest', 'heappush', 'heappop'],
    'math': ['sqrt', 'ceil', 'floor', 'log', 'exp', 'sin', 'cos', 'tan', 'pi', 'e'],
    'json': ['loads', 'dumps'],
    'logging': ['getLogger', 'info', 'debug', 'warning', 'error', 'critical', 'basicConfig', 'Logger', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    'typing': ['List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any', 'Callable', 'Iterator', 'Iterable'],
    'sys': ['version', 'version_info', 'maxsize'],  # Safe read-only sys attributes only
    'dateutil': ['parser', 'parse'],  # dateutil.parser for date parsing
}

class SecureExecutionError(Exception):
    """Raised when code contains dangerous operations"""
    pass

class SecurityValidator(ast.NodeVisitor):
    """AST visitor to check for dangerous operations"""
    
    def __init__(self):
        self.errors = []
        self.imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                self.errors.append(f"Import '{alias.name}' not allowed")
            else:
                self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module not in ALLOWED_MODULES:
            # Special case for dateutil submodules
            if node.module and node.module.startswith('dateutil.'):
                parent_module = 'dateutil'
                if parent_module in ALLOWED_MODULES:
                    # Allow dateutil submodules like dateutil.parser
                    for alias in node.names:
                        if alias.name not in ALLOWED_MODULES[parent_module]:
                            self.errors.append(f"Import '{alias.name}' from '{node.module}' not allowed")
                else:
                    self.errors.append(f"Import from '{node.module}' not allowed")
            else:
                self.errors.append(f"Import from '{node.module}' not allowed")
        else:
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES[node.module]:
                    self.errors.append(f"Import '{alias.name}' from '{node.module}' not allowed")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Check for dangerous function calls
        if hasattr(node.func, 'id'):
            if node.func.id in ['exec', 'eval', 'compile', '__import__', 'open', 'input']:
                self.errors.append(f"Function '{node.func.id}' not allowed")
        elif hasattr(node.func, 'attr'):
            if node.func.attr in ['system', 'popen', 'spawn', 'call', 'run']:
                self.errors.append(f"Method '{node.func.attr}' not allowed")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        # Check for dangerous attribute access
        dangerous_attrs = ['__globals__', '__locals__', '__builtins__', '__import__', '__eval__']
        if hasattr(node, 'attr') and node.attr in dangerous_attrs:
            self.errors.append(f"Attribute '{node.attr}' not allowed")
        self.generic_visit(node)

def create_safe_environment():
    """Create a restricted execution environment"""
    safe_builtins = {}
    
    # Add only safe builtins
    for name in ALLOWED_BUILTINS:
        if name in __builtins__:
            safe_builtins[name] = __builtins__[name]
    
    # Add safe __import__ function
    def safe_import(name, *args, **kwargs):
        if name in ALLOWED_MODULES:
            return __import__(name, *args, **kwargs)
        elif name.startswith('dateutil.'):
            # Special handling for dateutil submodules
            return __import__(name, *args, **kwargs)
        else:
            raise ImportError(f"Import of '{name}' is not allowed")
    
    safe_builtins['__import__'] = safe_import
    
    # Add safe modules
    safe_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        '__doc__': None,
    }
    
    # Import allowed modules safely
    for module_name, allowed_items in ALLOWED_MODULES.items():
        try:
            if module_name == 'dateutil':
                # Special handling for dateutil - it's usually imported as submodules
                try:
                    import dateutil.parser
                    safe_module = ModuleType('dateutil')
                    safe_module.parser = dateutil.parser
                    safe_globals['dateutil'] = safe_module
                except ImportError:
                    pass  # dateutil not available
            else:
                module = __import__(module_name)
                safe_module = ModuleType(module_name)
                for item in allowed_items:
                    if hasattr(module, item):
                        setattr(safe_module, item, getattr(module, item))
                safe_globals[module_name] = safe_module
        except ImportError:
            pass
    
    return safe_globals

def validate_and_execute_code(code: str, test_input: Dict = None):
    """Safely validate and execute user code with security checks"""
    
    # Step 1: Parse and validate AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecureExecutionError(f"Syntax Error: {str(e)}")
    
    # Step 2: Security validation
    validator = SecurityValidator()
    validator.visit(tree)
    
    if validator.errors:
        raise SecureExecutionError(f"Security violations: {'; '.join(validator.errors)}")
    
    # Step 3: Create safe execution environment
    safe_globals = create_safe_environment()
    safe_locals = {}
    
    # Step 4: Execute code in sandbox
    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        raise SecureExecutionError(f"Execution Error: {str(e)}")
    
    # Step 5: Extract and validate function
    if 'calculate_user_metrics' not in safe_locals:
        raise SecureExecutionError("Function 'calculate_user_metrics' not found")
    
    func = safe_locals['calculate_user_metrics']
    
    # Step 6: Test with provided input if given
    if test_input:
        try:
            result = func(**test_input)
            return func, result
        except Exception as e:
            raise SecureExecutionError(f"Function execution error: {str(e)}")
    
    return func, None

class CompetitiveProgrammingSystem:
    def __init__(self):
        self.db_path = "competition.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for submissions and leaderboard with device tracking"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Enable WAL mode for better concurrent access
                conn.execute('PRAGMA journal_mode=WAL;')
                conn.execute('PRAGMA synchronous=NORMAL;')
                conn.execute('PRAGMA cache_size=10000;')
                conn.execute('PRAGMA temp_store=memory;')
                
                cursor = conn.cursor()
                
                # Enhanced submissions table with device tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS submissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        device_fingerprint TEXT NOT NULL,
                        code TEXT NOT NULL,
                        code_hash TEXT,
                        submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        bug_score INTEGER DEFAULT 0,
                        edge_case_score INTEGER DEFAULT 0,
                        performance_score INTEGER DEFAULT 0,
                        security_score INTEGER DEFAULT 0,
                        total_score INTEGER DEFAULT 0,
                        analysis_complete BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Enhanced leaderboard table with device tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS leaderboard (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        device_fingerprint TEXT NOT NULL,
                        submission_id INTEGER NOT NULL,
                        best_score INTEGER DEFAULT 0,
                        submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        position INTEGER DEFAULT 0,
                        is_best_for_user BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (submission_id) REFERENCES submissions (id)
                    )
                ''')
                
                # Add device tracking columns to existing tables if they don't exist
                try:
                    cursor.execute('ALTER TABLE submissions ADD COLUMN device_fingerprint TEXT')
                    print("✅ Added device_fingerprint to submissions table")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    cursor.execute('ALTER TABLE leaderboard ADD COLUMN device_fingerprint TEXT')
                    print("✅ Added device_fingerprint to leaderboard table")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    cursor.execute('ALTER TABLE leaderboard ADD COLUMN is_best_for_user BOOLEAN DEFAULT FALSE')
                    print("✅ Added is_best_for_user to leaderboard table")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                # Create device tracking table for user-device relationships
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_devices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        device_fingerprint TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        submission_count INTEGER DEFAULT 0,
                        is_verified BOOLEAN DEFAULT TRUE,
                        UNIQUE(student_name, device_fingerprint)
                    )
                ''')
                
                conn.commit()
                print("✅ Database initialized successfully with device tracking and WAL mode")
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
    
    def create_specialized_agents(self) -> Dict[str, MOAgent]:
        """DEPRECATED: This method is no longer used. Scoring is handled by FunctionQualityGrader."""
        print("⚠️ Warning: create_specialized_agents is deprecated. Using FunctionQualityGrader instead.")
        return {}  # Return empty dict for backward compatibility
    
    def submit_solution(self, student_name: str, code: str, device_fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Submit a solution for analysis with device tracking for fairness"""
        # Get device fingerprint
        if device_fingerprint is None:
            device_fingerprint = get_device_fingerprint()
        
        # Validate device consistency for this user
        device_validation = self.validate_user_device(student_name, device_fingerprint)
        if not device_validation["valid"]:
            return {
                "error": f"Device validation failed: {device_validation['reason']}",
                "device_fingerprint": device_fingerprint,
                "allowed_device": device_validation.get("expected_device")
            }
        
        # Strip comments from code for security and fairness
        print("🧹 Stripping comments from submitted code...")
        original_length = len(code)
        cleaned_code = strip_comments_from_code(code)
        print(f"📝 Code cleaned: {original_length} -> {len(cleaned_code)} characters")
        
        # Generate hash for tracking
        code_hash = hashlib.md5(cleaned_code.encode()).hexdigest()
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Insert new submission with device tracking
                cursor.execute('''
                    INSERT INTO submissions (student_name, device_fingerprint, code, code_hash)
                    VALUES (?, ?, ?, ?)
                ''', (student_name, device_fingerprint, cleaned_code, code_hash))
                
                submission_id = cursor.lastrowid
                
                # Update user device tracking
                self.update_user_device_tracking(student_name, device_fingerprint, conn)
                
                conn.commit()
                
                return {
                    "submission_id": submission_id, 
                    "status": "submitted", 
                    "code_hash": code_hash,
                    "device_fingerprint": device_fingerprint,
                    "device_status": device_validation["status"]
                }
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
    
    def validate_user_device(self, student_name: str, device_fingerprint: str) -> Dict[str, Any]:
        """Validate that the user is submitting from a consistent device"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Check if user has submitted before
                cursor.execute('''
                    SELECT device_fingerprint, submission_count 
                    FROM user_devices 
                    WHERE student_name = ?
                    ORDER BY first_seen ASC
                ''', (student_name,))
                
                existing_devices = cursor.fetchall()
                
                if not existing_devices:
                    # First submission from this user - allow any device
                    print(f"✅ New user '{student_name}' - device {device_fingerprint} approved")
                    return {
                        "valid": True,
                        "status": "new_user",
                        "reason": "First submission from user"
                    }
                
                # Check if this device is already registered for this user
                user_devices = [device[0] for device in existing_devices]
                if device_fingerprint in user_devices:
                    print(f"✅ Known device for '{student_name}' - device {device_fingerprint} approved")
                    return {
                        "valid": True,
                        "status": "known_device",
                        "reason": "Device previously used by this user"
                    }
                
                # User trying to submit from a different device - block
                expected_device = user_devices[0]  # Use the first registered device
                print(f"❌ Device mismatch for '{student_name}' - expected {expected_device}, got {device_fingerprint}")
                return {
                    "valid": False,
                    "status": "device_mismatch",
                    "reason": "Submissions must come from the same device",
                    "expected_device": expected_device
                }
                
        except Exception as e:
            print(f"❌ Error validating device: {str(e)}")
            return {
                "valid": False,
                "status": "validation_error",
                "reason": f"Device validation error: {str(e)}"
            }
    
    def update_user_device_tracking(self, student_name: str, device_fingerprint: str, conn):
        """Update the user device tracking table"""
        cursor = conn.cursor()
        
        # Insert or update device tracking
        cursor.execute('''
            INSERT OR REPLACE INTO user_devices 
            (student_name, device_fingerprint, first_seen, last_seen, submission_count, is_verified)
            VALUES (
                ?, ?, 
                COALESCE((SELECT first_seen FROM user_devices WHERE student_name = ? AND device_fingerprint = ?), CURRENT_TIMESTAMP),
                CURRENT_TIMESTAMP,
                COALESCE((SELECT submission_count FROM user_devices WHERE student_name = ? AND device_fingerprint = ?), 0) + 1,
                TRUE
            )
        ''', (student_name, device_fingerprint, student_name, device_fingerprint, student_name, device_fingerprint))
        
        print(f"📱 Updated device tracking: {student_name} @ {device_fingerprint}")
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate that submitted code can run and pass basic unit tests.
        Returns validation results with pass/fail status and details.
        NOTE: Does not block submissions for prompt injection - only logs them.
        """
        print("🧪 Starting secure code validation...")
        validation_results = {
            "passes_tests": False,
            "syntax_valid": False,
            "runtime_errors": [],
            "test_results": [],
            "total_tests": 0,
            "passed_tests": 0,
            "security_errors": [],
            "injection_attempts": []
        }
        
        try:
            # Step 1: Check for prompt injection attempts (LOG ONLY - don't block)
            print("🔍 Checking for prompt injection attempts...")
            injection_violations = detect_prompt_injection(code)
            if injection_violations:
                validation_results["injection_attempts"] = injection_violations
                print(f"⚠️ Potential prompt injection detected: {len(injection_violations)} violations (logged but not blocking)")
                # Continue with validation - don't block submission
            
            # Step 2: Secure validation and extraction
            print("🔒 Checking security and extracting function...")
            try:
                user_function, _ = validate_and_execute_code(code)
                validation_results["syntax_valid"] = True
                print("✅ Security check and function extraction passed")
            except SecureExecutionError as e:
                validation_results["security_errors"].append(str(e))
                print(f"❌ Security/Execution Error: {str(e)}")
                return validation_results
            
            # Step 3: Run unit tests with secure execution (use ORIGINAL code)
            print("🧪 Running unit tests with secure execution...")
            test_cases = [
                {
                    "name": "Normal case",
                    "input": {
                        "users": [
                            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
                            {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
                        ],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    },
                    "expected_keys": ["average_engagement", "top_performers", "active_count"]
                },
                {
                    "name": "Empty users list",
                    "input": {
                        "users": [],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    },
                    "expected_keys": ["average_engagement", "top_performers", "active_count"]
                },
                {
                    "name": "Zero days active",
                    "input": {
                        "users": [
                            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 0}
                        ],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    },
                    "expected_keys": ["average_engagement", "top_performers", "active_count"]
                },
                {
                    "name": "Missing keys",
                    "input": {
                        "users": [
                            {"last_login": "2024-01-15", "posts": 10}  # Missing comments, likes, days_active
                        ],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    },
                    "expected_keys": ["average_engagement", "top_performers", "active_count"]
                }
            ]
            
            validation_results["total_tests"] = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                test_result = {
                    "test_name": test_case["name"],
                    "passed": False,
                    "error": None,
                    "output": None
                }
                
                try:
                    print(f"  🔍 Test {i+1}: {test_case['name']}")
                    
                    # Run the function with test input in secure environment
                    result = user_function(**test_case["input"])
                    test_result["output"] = result
                    
                    # Check that result is a dictionary
                    if not isinstance(result, dict):
                        test_result["error"] = f"Expected dict output, got {type(result)}"
                        print(f"    ❌ Wrong return type: {type(result)}")
                        continue
                    
                    # Check that all expected keys are present
                    missing_keys = set(test_case["expected_keys"]) - set(result.keys())
                    if missing_keys:
                        test_result["error"] = f"Missing keys: {missing_keys}"
                        print(f"    ❌ Missing keys: {missing_keys}")
                        continue
                    
                    # Additional validation for specific test cases
                    if test_case["name"] == "Empty users list":
                        if result["active_count"] != 0:
                            test_result["error"] = f"Expected active_count=0 for empty users, got {result['active_count']}"
                            print(f"    ❌ Expected active_count=0, got {result['active_count']}")
                            continue
                    
                    # If we get here, test passed
                    test_result["passed"] = True
                    validation_results["passed_tests"] += 1
                    print(f"    ✅ Test passed")
                    
                except Exception as e:
                    test_result["error"] = str(e)
                    print(f"    ❌ Test failed: {str(e)}")
                
                validation_results["test_results"].append(test_result)
            
            # Determine if code passes overall validation
            validation_results["passes_tests"] = (
                validation_results["syntax_valid"] and 
                validation_results["passed_tests"] >= 2 and  # Must pass at least 2 out of 4 tests
                len(validation_results["security_errors"]) == 0  # No security violations
                # NOTE: injection_attempts no longer blocks validation
            )
            
            if validation_results["passes_tests"]:
                print(f"✅ Secure code validation passed! ({validation_results['passed_tests']}/{validation_results['total_tests']} tests)")
                if injection_violations:
                    print(f"⚠️ Note: {len(injection_violations)} potential prompt injection attempts detected (will be sanitized for AI analysis)")
            else:
                print(f"❌ Code validation failed! ({validation_results['passed_tests']}/{validation_results['total_tests']} tests)")
            
            return validation_results
            
        except Exception as e:
            validation_results["runtime_errors"].append(f"Validation system error: {str(e)}")
            print(f"❌ Validation system error: {str(e)}")
            return validation_results
    
    def analyze_submission(self, submission_id: int, agents: Dict[str, MOAgent] = None) -> Dict[str, Any]:
        """Analyze a submission using the deterministic grader - no AI agents needed"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Get submission
                cursor.execute('SELECT code, student_name FROM submissions WHERE id = ?', (submission_id,))
                result = cursor.fetchone()
                if not result:
                    return {"error": "Submission not found"}
                
                code, student_name = result
                
                # STEP 1: Validate code before analysis
                print("🧪 Pre-analysis validation...")
                validation_results = self.validate_code(code)
                
                if not validation_results["passes_tests"]:
                    print("❌ Code failed validation - using low score")
                    
                    # Update submission with validation failure
                    cursor.execute('''
                        UPDATE submissions 
                        SET bug_score = 0, edge_case_score = 0, performance_score = 0, security_score = 0,
                            total_score = 0, analysis_complete = TRUE
                        WHERE id = ?
                    ''', (submission_id,))
                    conn.commit()
                    
                    return {
                        "validation_failed": True,
                        "validation_results": validation_results,
                        "total_score": 0,
                        "student_name": student_name,
                        "message": "Code failed unit tests"
                    }
                
                print("✅ Code validation passed - proceeding to grader analysis")
                
                # STEP 2: Use the deterministic grader
                print("🎯 Using FunctionQualityGrader for scoring...")
                grader = FunctionQualityGrader()
                
                # Extract the function from the user's code
                try:
                    user_function, _ = validate_and_execute_code(code)
                except Exception as e:
                    print(f"❌ Function extraction failed: {str(e)}")
                    # Use fallback score if function extraction fails
                    cursor.execute('''
                        UPDATE submissions 
                        SET bug_score = 0, edge_case_score = 0, performance_score = 0, security_score = 0,
                            total_score = 0, analysis_complete = TRUE
                        WHERE id = ?
                    ''', (submission_id,))
                    conn.commit()
                    
                    return {
                        "extraction_failed": True,
                        "validation_results": validation_results,
                        "total_score": 0,
                        "student_name": student_name,
                        "message": f"Function extraction failed: {str(e)}"
                    }
                
                # Grade the function
                grading_results = grader.test_function(user_function, f"{student_name}'s implementation")
                
                # Convert grader results to our scoring format
                total_score = grading_results['total_score']
                
                # Map detailed results to our categories (approximate mapping)
                critical_bugs_score = sum([
                    grading_results['detailed_results'].get('handles_division_by_zero', {}).get('score', 0),
                    grading_results['detailed_results'].get('handles_empty_users_list', {}).get('score', 0),
                    grading_results['detailed_results'].get('handles_missing_keys', {}).get('score', 0),
                    grading_results['detailed_results'].get('correct_average_calculation', {}).get('score', 0),
                ])
                
                logic_bugs_score = sum([
                    grading_results['detailed_results'].get('correct_sorting_direction', {}).get('score', 0),
                    grading_results['detailed_results'].get('handles_no_active_users', {}).get('score', 0),
                    grading_results['detailed_results'].get('doesnt_mutate_input', {}).get('score', 0),
                ])
                
                edge_cases_score = sum([
                    grading_results['detailed_results'].get('handles_less_than_5_users', {}).get('score', 0),
                    grading_results['detailed_results'].get('handles_invalid_dates', {}).get('score', 0),
                    grading_results['detailed_results'].get('robust_error_handling', {}).get('score', 0),
                ])
                
                performance_score = grading_results['detailed_results'].get('efficient_implementation', {}).get('score', 0)
                
                print(f"🎯 Grader Results:")
                print(f"   • Total Score: {total_score}/{grader.max_possible_score}")
                print(f"   • Critical Bugs: {critical_bugs_score}")
                print(f"   • Logic Issues: {logic_bugs_score}")
                print(f"   • Edge Cases: {edge_cases_score}")
                print(f"   • Performance: {performance_score}")
                print(f"   • Tests Passed: {grading_results['tests_passed']}")
                print(f"   • Tests Failed: {grading_results['tests_failed']}")
                
                # Update submission with results
                cursor.execute('''
                    UPDATE submissions 
                    SET bug_score = ?, edge_case_score = ?, performance_score = ?, security_score = ?,
                        total_score = ?, analysis_complete = TRUE
                    WHERE id = ?
                ''', (
                    critical_bugs_score,  # Map critical bugs to bug_score
                    edge_cases_score,     # Edge cases to edge_case_score
                    performance_score,    # Performance to performance_score
                    logic_bugs_score,     # Logic bugs to security_score (repurposed)
                    total_score,
                    submission_id
                ))
                
                conn.commit()
                
                # Update leaderboard
                try:
                    self.update_leaderboard(student_name, total_score, submission_id)
                except Exception as leaderboard_error:
                    print(f"Warning: Leaderboard update failed: {str(leaderboard_error)}")
                
                return {
                    "validation_results": validation_results,
                    "grading_results": grading_results,
                    "total_score": total_score,
                    "max_score": grader.max_possible_score,
                    "student_name": student_name,
                    "detailed_scores": {
                        "critical_bugs": critical_bugs_score,
                        "logic_issues": logic_bugs_score,
                        "edge_cases": edge_cases_score,
                        "performance": performance_score
                    }
                }
        except Exception as e:
            return {"error": f"Database error during analysis: {str(e)}"}
    
    def update_leaderboard(self, student_name: str, score: int, submission_id: int):
        """Update the leaderboard with new score - allows multiple submissions from same device"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Get device fingerprint for this submission
                cursor.execute('SELECT device_fingerprint FROM submissions WHERE id = ?', (submission_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"❌ Submission {submission_id} not found")
                    return
                
                device_fingerprint = result[0]
                
                # Always insert new entry for this submission
                cursor.execute('''
                    INSERT INTO leaderboard (student_name, device_fingerprint, submission_id, best_score, submission_time)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (student_name, device_fingerprint, submission_id, score))
                
                # Update best score tracking for this user
                self.update_best_scores_for_user(student_name, conn)
                
                # Update positions for all entries
                cursor.execute('''
                    UPDATE leaderboard 
                    SET position = (
                        SELECT COUNT(*) + 1 
                        FROM leaderboard l2 
                        WHERE l2.best_score > leaderboard.best_score
                    )
                ''')
                
                conn.commit()
                print(f"✅ Added leaderboard entry: {student_name} with score {score} (device: {device_fingerprint[:8]}...)")
        except Exception as e:
            print(f"❌ Error updating leaderboard: {str(e)}")
    
    def update_best_scores_for_user(self, student_name: str, conn):
        """Update which entries represent the best score for each user"""
        cursor = conn.cursor()
        
        # Reset all is_best_for_user flags for this user
        cursor.execute('''
            UPDATE leaderboard 
            SET is_best_for_user = FALSE 
            WHERE student_name = ?
        ''', (student_name,))
        
        # Find the best score for this user
        cursor.execute('''
            SELECT MAX(best_score) 
            FROM leaderboard 
            WHERE student_name = ?
        ''', (student_name,))
        
        max_score_result = cursor.fetchone()
        if not max_score_result or max_score_result[0] is None:
            return
        
        max_score = max_score_result[0]
        
        # Mark the most recent submission with the best score as the user's best
        cursor.execute('''
            UPDATE leaderboard 
            SET is_best_for_user = TRUE 
            WHERE student_name = ? AND best_score = ?
            AND id = (
                SELECT id FROM leaderboard 
                WHERE student_name = ? AND best_score = ?
                ORDER BY submission_time DESC 
                LIMIT 1
            )
        ''', (student_name, max_score, student_name, max_score))
        
        print(f"📊 Updated best score tracking for {student_name}: {max_score} points")
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard - shows best score per user with device tracking info"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Get best scores per user (only show users' best submissions)
                cursor.execute('''
                    SELECT l.student_name, l.best_score, l.submission_time, l.position, 
                           l.submission_id, l.device_fingerprint,
                           ud.submission_count, ud.first_seen
                    FROM leaderboard l
                    LEFT JOIN user_devices ud ON l.student_name = ud.student_name AND l.device_fingerprint = ud.device_fingerprint
                    WHERE l.is_best_for_user = TRUE
                    ORDER BY l.best_score DESC, l.submission_time ASC
                    LIMIT 20
                ''')
                
                results = cursor.fetchall()
                
                leaderboard = []
                for i, (name, score, submission_time, position, submission_id, device_fp, sub_count, first_seen) in enumerate(results):
                    leaderboard.append({
                        "position": i + 1,
                        "student_name": name,
                        "best_score": score,
                        "submission_time": submission_time,
                        "submission_id": submission_id,
                        "percentage": round((score / 135) * 100, 1),  # Max score is 135
                        "device_info": {
                            "device_id": device_fp[:8] + "..." if device_fp else "unknown",
                            "total_submissions": sub_count or 0,
                            "first_seen": first_seen
                        }
                    })
                
                return leaderboard
        except Exception as e:
            print(f"❌ Error getting leaderboard: {str(e)}")
            return []
    
    def get_user_submission_history(self, student_name: str) -> List[Dict[str, Any]]:
        """Get submission history for a specific user"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT s.id, s.submission_time, s.total_score, s.device_fingerprint,
                           s.bug_score, s.edge_case_score, s.performance_score, s.security_score
                    FROM submissions s
                    WHERE s.student_name = ? AND s.analysis_complete = TRUE
                    ORDER BY s.submission_time DESC
                ''', (student_name,))
                
                results = cursor.fetchall()
                
                history = []
                for sub_id, sub_time, total_score, device_fp, bug_score, edge_score, perf_score, sec_score in results:
                    history.append({
                        "submission_id": sub_id,
                        "submission_time": sub_time,
                        "total_score": total_score,
                        "device_id": device_fp[:8] + "..." if device_fp else "unknown",
                        "scores": {
                            "bug_fixing": bug_score,
                            "edge_cases": edge_score,
                            "performance": perf_score,
                            "security": sec_score
                        }
                    })
                
                return history
        except Exception as e:
            print(f"❌ Error getting user history: {str(e)}")
            return []
    
    def get_submission_details(self, submission_id: int) -> Dict[str, Any]:
        """Get detailed analysis of a specific submission"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT student_name, code, submission_time, bug_score, edge_case_score,
                           performance_score, security_score, total_score
                    FROM submissions 
                    WHERE id = ?
                ''', (submission_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    return {"error": "Submission not found"}
                
                return {
                    "student_name": result[0],
                    "code": result[1],
                    "submission_time": result[2],
                    "bug_score": result[3],
                    "edge_case_score": result[4],
                    "performance_score": result[5],
                    "security_score": result[6],
                    "total_score": result[7]
                }
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
    
    def rebuild_leaderboard(self):
        """Rebuild the entire leaderboard from scratch"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Clear existing leaderboard
                cursor.execute('DELETE FROM leaderboard')
                
                # Get all completed submissions
                cursor.execute('''
                    SELECT s.id, s.student_name, s.total_score, s.submission_time, s.device_fingerprint
                    FROM submissions s
                    WHERE s.analysis_complete = TRUE
                    ORDER BY s.total_score DESC, s.submission_time ASC
                ''')
                
                results = cursor.fetchall()
                
                # Rebuild leaderboard with all submissions - FIXED: Use same 4-parameter format
                for submission_id, student_name, score, submission_time, device_fingerprint in results:
                    cursor.execute('''
                        INSERT INTO leaderboard (student_name, device_fingerprint, submission_id, best_score)
                        VALUES (?, ?, ?, ?)
                    ''', (student_name, device_fingerprint, submission_id, score))
                
                # Update best score tracking for all users
                all_users = set(row[1] for row in results)  # Get unique student names
                for student_name in all_users:
                    self.update_best_scores_for_user(student_name, conn)
                
                # Update positions
                cursor.execute('''
                    UPDATE leaderboard 
                    SET position = (
                        SELECT COUNT(*) + 1 
                        FROM leaderboard l2 
                        WHERE l2.best_score > leaderboard.best_score
                    )
                ''')
                
                conn.commit()
                print(f"✅ Leaderboard rebuilt with {len(results)} entries")
                return True
        except Exception as e:
            print(f"❌ Error rebuilding leaderboard: {str(e)}")
            return False
    
    def sanitize_code_with_agent(self, code: str, sanitizer_agent: MOAgent) -> str:
        """
        DEPRECATED: Use manual sanitization instead.
        Sanitize code using a specialized agent to remove potentially dangerous content 
        while preserving functionality. No longer used since switching to grader.py.
        """
        try:
            print("⚠️ Warning: sanitize_code_with_agent is deprecated.")
            return sanitize_code_for_ai(code)
        except Exception as e:
            print(f"❌ AI sanitization failed: {str(e)}")
            print("🔄 Falling back to manual sanitization...")
            # Fallback to existing manual sanitization
            return sanitize_code_for_ai(code)

    def validate_analysis_consistency(self, submission_id: int, agents: Dict[str, MOAgent] = None, validation_runs: int = 2) -> Dict[str, Any]:
        """
        DEPRECATED: Grader.py provides deterministic scoring, so consistency validation is not needed.
        The grader always produces the same results for the same code.
        """
        print("⚠️ Warning: validate_analysis_consistency is deprecated. Grader.py provides deterministic scoring.")
        
        # For backward compatibility, just return that it's consistent
        return {
            "consistent": True,
            "consistency_rate": 1.0,
            "validation_runs": 1,
            "inconsistencies": [],
            "results": [{"note": "Using deterministic grader - always consistent"}],
            "recommended_action": "use_result"
        }

def strip_comments_from_code(code: str) -> str:
    """
    Strip comments from user code at submission time for security and fairness.
    This ensures comments can't be used for prompt injection and that users 
    aren't penalized for lack of documentation.
    """
    try:
        lines = code.split('\n')
        cleaned_lines = []
        in_multiline_string = False
        string_delimiter = None
        
        for line in lines:
            cleaned_line = ""
            i = 0
            
            while i < len(line):
                char = line[i]
                
                # Handle string literals (preserve # inside strings)
                if not in_multiline_string and char in ['"', "'"]:
                    # Check for triple quotes
                    if i + 2 < len(line) and line[i:i+3] in ['"""', "'''"]:
                        if string_delimiter is None:
                            in_multiline_string = True
                            string_delimiter = line[i:i+3]
                            cleaned_line += line[i:i+3]
                            i += 3
                            continue
                        elif line[i:i+3] == string_delimiter:
                            in_multiline_string = False
                            string_delimiter = None
                            cleaned_line += line[i:i+3]
                            i += 3
                            continue
                    
                    # Single quotes
                    if not in_multiline_string:
                        # Find the end of the string
                        quote_char = char
                        cleaned_line += char
                        i += 1
                        while i < len(line):
                            if line[i] == quote_char and line[i-1] != '\\':
                                cleaned_line += line[i]
                                i += 1
                                break
                            cleaned_line += line[i]
                            i += 1
                        continue
                
                # Handle multiline strings
                if in_multiline_string:
                    if i + 2 < len(line) and line[i:i+3] == string_delimiter:
                        in_multiline_string = False
                        string_delimiter = None
                        cleaned_line += line[i:i+3]
                        i += 3
                        continue
                    else:
                        cleaned_line += char
                        i += 1
                        continue
                
                # Handle comments (# character outside of strings)
                if char == '#' and not in_multiline_string:
                    # This is a comment, ignore rest of line
                    break
                
                cleaned_line += char
                i += 1
            
            # Remove trailing whitespace and add line if not empty
            cleaned_line = cleaned_line.rstrip()
            if cleaned_line or not cleaned_lines:  # Keep at least one line
                cleaned_lines.append(cleaned_line)
        
        # Remove leading/trailing empty lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        print(f"Warning: Comment stripping failed: {str(e)}")
        # Return original code if stripping fails
        return code

def get_device_fingerprint() -> str:
    """
    Generate a device fingerprint for tracking submissions.
    
    In a real deployment, this would use more sophisticated device fingerprinting
    like browser fingerprinting, IP + User-Agent, or hardware identifiers.
    For this demo, we'll use a simple approach.
    """
    import os
    import platform
    import socket
    
    # Combine multiple device characteristics
    components = [
        platform.system(),           # OS name
        platform.machine(),          # Machine type
        platform.processor(),        # Processor info
        socket.gethostname(),        # Computer name
        os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))  # Username
    ]
    
    # Create a stable hash from device characteristics
    device_string = '|'.join(str(c) for c in components)
    device_hash = hashlib.sha256(device_string.encode()).hexdigest()[:16]
    
    print(f"🔍 Device fingerprint: {device_hash} (based on {len(components)} characteristics)")
    return device_hash

# Test cases for validation
TEST_CASES = [
    {
        "name": "Normal case",
        "users": [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
            {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Empty users",
        "users": [],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Zero days active",
        "users": [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 0}
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Missing keys",
        "users": [
            {"last_login": "2024-01-15", "posts": 10}  # Missing comments, likes, days_active
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
] 