#!/usr/bin/env python3
"""
Test script to show what the expected outputs should be for each test case
"""

from competitive_programming import TEST_CASES

print("📋 Expected Test Case Outputs")
print("=" * 50)

def calculate_user_metrics_correct(users, start_date, end_date):
    """Correct implementation of the function"""
    # Handle empty input
    if not isinstance(users, list) or not users:
        return {
            'average_engagement': 0.0,
            'top_performers': [],
            'active_count': 0
        }
    
    # Validate date inputs
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        return {
            'average_engagement': 0.0,
            'top_performers': [],
            'active_count': 0
        }
    
    total_score = 0
    active_users = []
    required_keys = ['last_login', 'posts', 'comments', 'likes', 'days_active']
    
    for user in users:
        # Check if user is a dict and has required keys
        if not isinstance(user, dict):
            continue
            
        if not all(key in user for key in required_keys):
            continue
            
        # Use string comparison for dates (YYYY-MM-DD format)
        if user['last_login'] >= start_date and user['last_login'] <= end_date:
            # Handle division by zero
            if user['days_active'] <= 0:
                continue
                
            # Calculate engagement score
            score = user['posts'] * 2 + user['comments'] * 1.5 + user['likes'] * 0.1
            user_copy = user.copy()  # Don't modify original
            user_copy['engagement_score'] = score / user['days_active']
            total_score += score
            active_users.append(user_copy)
    
    # Handle no active users
    if not active_users:
        return {
            'average_engagement': 0.0,
            'top_performers': [],
            'active_count': 0
        }
    
    # Calculate average correctly (use active_users, not all users)
    avg_score = total_score / len(active_users)
    
    # Sort correctly (reverse=True for descending order)
    top_users = sorted(active_users, key=lambda x: x['engagement_score'], reverse=True)[:5]
    
    return {
        'average_engagement': avg_score,
        'top_performers': top_users,
        'active_count': len(active_users)
    }

# Test each case
for i, test_case in enumerate(TEST_CASES):
    print(f"\n{i+1}️⃣ Test Case: {test_case['name']}")
    print(f"📥 Input:")
    print(f"   Users: {len(test_case['users'])} users")
    print(f"   Date Range: {test_case['start_date']} to {test_case['end_date']}")
    
    if test_case['users']:
        print("   Sample user:", test_case['users'][0])
    
    try:
        result = calculate_user_metrics_correct(
            test_case['users'], 
            test_case['start_date'], 
            test_case['end_date']
        )
        print(f"📤 Expected Output:")
        print(f"   ✅ Should succeed: True")
        print(f"   Average Engagement: {result['average_engagement']}")
        print(f"   Active Count: {result['active_count']}")
        print(f"   Top Performers: {len(result['top_performers'])} users")
        
        if result['top_performers']:
            print(f"   Best Score: {result['top_performers'][0]['engagement_score']:.2f}")
            
    except Exception as e:
        print(f"   ❌ Should fail: {str(e)}")

print(f"\n{'='*50}")
print("🎯 Summary of Expected Behavior:")
print("1. Normal case: Should return calculated metrics")
print("2. Empty users: Should return zeros gracefully")  
print("3. Zero days active: Should skip those users")
print("4. Missing keys: Should skip users with missing data")
print("\n💡 Key Requirements:")
print("- Use string date comparison (no datetime.strptime)")
print("- Handle division by zero (days_active = 0)")
print("- Handle missing dictionary keys gracefully")
print("- Calculate average using active_users, not all users")
print("- Sort top performers in descending order") 