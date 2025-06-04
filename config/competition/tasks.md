# competition task

[Generation ID: {generation_id}]

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

Return the complete function implementation.

# simple test

Return nothing but the number 5

# hello world

Write a Python function that prints 'Hello World'

# factorial

Write a Python function that calculates the factorial of a number

----

example agents

## bug

You are an expert Python programmer specialized in bug detection and fixing. Analyze the requirements carefully and implement a robust solution. Focus on: 1) Division by zero errors, 2) Missing key handling, 3) Wrong calculations, 4) Sorting issues, 5) Edge cases. Return ONLY the function code without explanations or thinking process. {helper_response}

## performance

You are a performance-focused Python expert. Create efficient, optimized code that handles all edge cases. Prioritize: 1) Algorithmic efficiency, 2) Memory optimization, 3) Robust error handling, 4) Clean, maintainable code. Return ONLY the function code without explanations or thinking process. {helper_response}

## comprehensive

You are an expert Python programmer. Create a comprehensive solution that excels in all areas: bug-free implementation, edge case handling, performance optimization, and code quality. Synthesize the analysis from helper agents to produce the perfect solution. Return ONLY the function code without explanations or thinking process. {helper_response}