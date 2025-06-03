import traceback
from typing import Dict, List, Any, Callable

class FunctionQualityGrader:
    """
    Direct code quality grader that tests Python function implementations
    and scores them based on bug fixes and robustness.
    """

    SCORING = {
        'critical_bugs': {
            'handles_division_by_zero': 20,
            'handles_empty_users_list': 15,
            'handles_missing_keys': 15,
            'correct_average_calculation': 15,
        },
        'logic_bugs': {
            'correct_sorting_direction': 10,
            'handles_no_active_users': 10,
            'doesnt_mutate_input': 8,
        },
        'edge_cases': {
            'handles_less_than_5_users': 5,
            'handles_invalid_dates': 5,
            'robust_error_handling': 7,
        },
        'performance': {
            'efficient_implementation': 10,
        }
    }

    def __init__(self):
        self.max_possible_score = sum(
            sum(category.values()) for category in self.SCORING.values()
        )

    def test_function(self, func: Callable, function_name: str = "Unknown") -> Dict[str, Any]:
        """Test a function implementation and return detailed scoring"""

        results = {
            'function_name': function_name,
            'total_score': 0,
            'max_score': self.max_possible_score,
            'tests_passed': 0,
            'tests_failed': 0,
            'detailed_results': {},
            'execution_logs': []
        }

        # Test 1: Division by zero handling
        score, passed, log = self._test_division_by_zero(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_division_by_zero'] = {
            'score': score, 'max': 20, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 9: Invalid dates handling
        score, passed, log = self._test_invalid_dates(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_invalid_dates'] = {
            'score': score, 'max': 5, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 10: Robust error handling
        score, passed, log = self._test_robust_error_handling(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['robust_error_handling'] = {
            'score': score, 'max': 7, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 11: Efficient implementation
        score, passed, log = self._test_efficient_implementation(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['efficient_implementation'] = {
            'score': score, 'max': 10, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 2: Empty users list
        score, passed, log = self._test_empty_users(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_empty_users_list'] = {
            'score': score, 'max': 15, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 3: Missing keys
        score, passed, log = self._test_missing_keys(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_missing_keys'] = {
            'score': score, 'max': 15, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 4: Correct average calculation
        score, passed, log = self._test_correct_average(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['correct_average_calculation'] = {
            'score': score, 'max': 15, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 5: Sorting direction
        score, passed, log = self._test_sorting_direction(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['correct_sorting_direction'] = {
            'score': score, 'max': 10, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 6: No active users
        score, passed, log = self._test_no_active_users(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_no_active_users'] = {
            'score': score, 'max': 10, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 7: Input mutation
        score, passed, log = self._test_input_mutation(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['doesnt_mutate_input'] = {
            'score': score, 'max': 8, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Test 8: Less than 5 users
        score, passed, log = self._test_less_than_5_users(func)
        results['total_score'] += score
        results['tests_passed'] += passed
        results['tests_failed'] += (1 - passed)
        results['detailed_results']['handles_less_than_5_users'] = {
            'score': score, 'max': 5, 'passed': bool(passed)
        }
        results['execution_logs'].append(log)

        # Calculate percentage
        results['percentage'] = round((results['total_score'] / self.max_possible_score) * 100, 1)

        return results

    def _test_division_by_zero(self, func):
        """Test handling of division by zero when days_active = 0"""
        test_users = [
            {
                'last_login': '2024-01-15',
                'posts': 10,
                'comments': 5,
                'likes': 20,
                'days_active': 0  # This should cause division by zero in buggy version
            }
        ]

        try:
            result = func(test_users, '2024-01-01', '2024-01-31')
            # If it doesn't crash, check if it handled it reasonably
            if isinstance(result, dict) and 'average_engagement' in result:
                return 20, 1, "✅ PASS: Handles division by zero gracefully"
            else:
                return 5, 0, "_test_division_by_zero: ⚠️  PARTIAL: No crash but unexpected result format"
        except ZeroDivisionError:
            return 0, 0, "_test_division_by_zero: ❌ FAIL: ZeroDivisionError - days_active = 0 not handled"
        except Exception as e:
            return 0, 0, f"_test_division_by_zero: ❌ FAIL: Unexpected error - {type(e).__name__}: {e}"

    def _test_empty_users(self, func):
        """Test handling of empty users list"""
        try:
            result = func([], '2024-01-01', '2024-01-31')
            if isinstance(result, dict) and result.get('active_count') == 0:
                return 15, 1, "✅ PASS: Handles empty users list correctly"
            else:
                return 5, 0, "⚠️  PARTIAL: No crash but unexpected result"
        except ZeroDivisionError:
            return 0, 0, "_test_empty_users: ❌ FAIL: ZeroDivisionError on empty list"
        except Exception as e:
            return 0, 0, f"_test_empty_users: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_missing_keys(self, func):
        """Test handling of missing dictionary keys"""
        incomplete_users = [
            {
                'last_login': '2024-01-15',
                'posts': 10
                # Missing: comments, likes, days_active
            }
        ]

        try:
            result = func(incomplete_users, '2024-01-01', '2024-01-31')
            return 15, 1, "✅ PASS: Handles missing keys gracefully"
        except KeyError as e:
            return 0, 0, f"_test_missing_keys: ❌ FAIL: KeyError - missing key {e}"
        except Exception as e:
            return 0, 0, f"_test_missing_keys: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_correct_average(self, func):
        """Test if average is calculated correctly (should use active users, not all users)"""
        test_users = [
            {
                'last_login': '2023-12-15',  # Outside date range
                'posts': 100, 'comments': 50, 'likes': 200, 'days_active': 30
            },
            {
                'last_login': '2024-01-15',  # Inside date range
                'posts': 10, 'comments': 5, 'likes': 20, 'days_active': 10
            }
        ]

        try:
            result = func(test_users, '2024-01-01', '2024-01-31')

            # Only 1 user should be active, so average should be based on 1 user, not 2
            expected_score = 10*2 + 5*1.5 + 20*0.1  # 27.5
            expected_engagement = expected_score / 10  # 2.75

            if abs(result.get('average_engagement', 0) - expected_engagement) < 0.01:
                return 15, 1, "✅ PASS: Calculates average correctly using active users only"
            else:
                # Check if it's using wrong denominator (all users instead of active)
                wrong_avg = expected_score / len(test_users)  # Using all users
                if abs(result.get('average_engagement', 0) - wrong_avg) < 0.01:
                    return 0, 0, "_test_correct_average: ❌ FAIL: Using len(users) instead of len(active_users) for average"
                else:
                    return 5, 0, f"_test_correct_average: ⚠️  PARTIAL: Average calculation unclear - got {result.get('average_engagement')}"
        except Exception as e:
            return 0, 0, f"_test_correct_average: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_sorting_direction(self, func):
        """Test if top performers are actually the highest scoring users"""
        test_users = [
            {'last_login': '2024-01-15', 'posts': 1, 'comments': 1, 'likes': 1, 'days_active': 1},  # Low score
            {'last_login': '2024-01-16', 'posts': 10, 'comments': 10, 'likes': 10, 'days_active': 1},  # High score
            {'last_login': '2024-01-17', 'posts': 5, 'comments': 5, 'likes': 5, 'days_active': 1},  # Medium score
        ]

        try:
            result = func(test_users, '2024-01-01', '2024-01-31')
            top_performers = result.get('top_performers', [])

            if len(top_performers) > 0:
                # Check if the first top performer has highest posts (should be the user with 10 posts)
                if top_performers[0].get('posts') == 10:
                    return 10, 1, "✅ PASS: Top performers correctly sorted (highest first)"
                elif top_performers[-1].get('posts') == 10:
                    return 0, 0, "_test_sorting_direction: ❌ FAIL: Sorting in wrong direction (lowest first instead of highest)"
                else:
                    return 5, 0, "_test_sorting_direction: ⚠️  PARTIAL: Sorting behavior unclear"
            else:
                return 0, 0, "_test_sorting_direction: ❌ FAIL: No top performers returned"
        except Exception as e:
            return 0, 0, f"_test_sorting_direction: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_no_active_users(self, func):
        """Test handling when no users match the date range"""
        test_users = [
            {'last_login': '2023-12-15', 'posts': 10, 'comments': 5, 'likes': 20, 'days_active': 10}
        ]

        try:
            result = func(test_users, '2024-01-01', '2024-01-31')  # Date range with no matches
            if result.get('active_count') == 0 and result.get('average_engagement') == 0:
                return 10, 1, "✅ PASS: Handles no active users correctly"
            else:
                return 5, 0, "_test_no_active_users: ⚠️  PARTIAL: Unexpected behavior with no active users"
        except Exception as e:
            return 0, 0, f"_test_no_active_users: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_input_mutation(self, func):
        """Test if function modifies the original input data"""
        test_users = [
            {'last_login': '2024-01-15', 'posts': 10, 'comments': 5, 'likes': 20, 'days_active': 10}
        ]
        original_user = test_users[0].copy()

        try:
            func(test_users, '2024-01-01', '2024-01-31')
            # Check if original data was modified
            if 'engagement_score' in test_users[0]:
                return 0, 0, "_test_input_mutation: ❌ FAIL: Function mutates input data (adds engagement_score)"
            elif test_users[0] != original_user:
                return 0, 0, "_test_input_mutation: ❌ FAIL: Function modifies input data"
            else:
                return 8, 1, "✅ PASS: Function doesn't mutate input data"
        except Exception as e:
            return 0, 0, f"_test_input_mutation: ❌ FAIL: {type(e).__name__}: {e}"

    def _test_less_than_5_users(self, func):
        """Test handling when there are fewer than 5 active users"""
        test_users = [
            {'last_login': '2024-01-15', 'posts': 10, 'comments': 5, 'likes': 20, 'days_active': 10},
            {'last_login': '2024-01-16', 'posts': 5, 'comments': 3, 'likes': 15, 'days_active': 5}
        ]

        try:
            result = func(test_users, '2024-01-01', '2024-01-31')
            top_performers = result.get('top_performers', [])

            # Should return 2 users, not try to get 5
            if len(top_performers) == 2:
                return 5, 1, "✅ PASS: Correctly handles fewer than 5 users"
            else:
                return 0, 0, f"_test_less_than_5_users: ❌ FAIL: Expected 2 top performers, got {len(top_performers)}"
        except Exception as e:
            return 0, 0, f"_test_less_than_5_users: ❌ FAIL: {type(e).__name__}: {e}"

    def print_results(self, results: Dict[str, Any]):
        """Print detailed test results"""
        print(f"=== FUNCTION QUALITY REPORT: {results['function_name']} ===\n")

        print(f"Overall Score: {results['total_score']}/{results['max_score']} ({results['percentage']}%)")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}\n")

        print("Detailed Results:")
        print("-" * 60)

        for test_name, test_result in results['detailed_results'].items():
            status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
            print(f"{test_name}: {test_result['score']}/{test_result['max']} {status}")

        print("\nExecution Logs:")
        print("-" * 60)
        for log in results['execution_logs']:
            print(log)

        # Grade interpretation
        print(f"\nGrade: {self._get_letter_grade(results['percentage'])}")

    def _test_invalid_dates(self, func):
        """Test handling of invalid or malformed date parameters"""
        test_users = [
            {'last_login': '2024-01-15', 'posts': 10, 'comments': 5, 'likes': 20, 'days_active': 10}
        ]

        try:
            # Test with end_date before start_date
            result = func(test_users, '2024-01-31', '2024-01-01')
            if result.get('active_count') == 0:
                return 5, 1, "✅ PASS: Handles invalid date range (end < start) correctly"
            else:
                return 3, 0, "_test_invalid_dates: ⚠️  PARTIAL: Doesn't validate date range properly"
        except Exception as e:
            # If it crashes on invalid dates, that's not great but not terrible
            return 2, 0, f"_test_invalid_dates: ⚠️  PARTIAL: Crashes on invalid dates - {type(e).__name__}: {e}"

    def _test_robust_error_handling(self, func):
        """Test function's robustness with various problematic inputs"""
        tests_passed = 0
        total_tests = 3

        # Test 1: None values in user data
        try:
            problematic_users = [
                {'last_login': '2024-01-15', 'posts': None, 'comments': 5, 'likes': 20, 'days_active': 10}
            ]
            result = func(problematic_users, '2024-01-01', '2024-01-31')
            tests_passed += 1
        except Exception:
            print("HERE 1")
            pass

        # Test 2: Negative values
        try:
            negative_users = [
                {'last_login': '2024-01-15', 'posts': -5, 'comments': 5, 'likes': 20, 'days_active': 10}
            ]
            result = func(negative_users, '2024-01-01', '2024-01-31')
            tests_passed += 1
        except Exception:
            print("HERE 2")
            pass

        # Test 3: String numbers
        try:
            string_users = [
                {'last_login': '2024-01-15', 'posts': '10', 'comments': 5, 'likes': 20, 'days_active': 10}
            ]
            result = func(string_users, '2024-01-01', '2024-01-31')
            tests_passed += 1
        except Exception:
            print("HERE 3")
            pass

        if tests_passed == total_tests:
            return 7, 1, "✅ PASS: Robust error handling - handles all problematic inputs"
        elif tests_passed >= 2:
            return 4, 0, f"_test_robust_error_handling: ⚠️  PARTIAL: Handles {tests_passed}/{total_tests} problematic inputs"
        elif tests_passed == 1:
            return 2, 0, f"_test_robust_error_handling: ⚠️  PARTIAL: Handles {tests_passed}/{total_tests} problematic inputs"
        else:
            return 0, 0, "_test_robust_error_handling: ❌ FAIL: Poor error handling - crashes on problematic inputs"

    def _test_efficient_implementation(self, func):
        """Test if implementation uses efficient algorithms (e.g., not sorting full list for top 5)"""
        # Create a larger dataset to test efficiency
        large_users = []
        for i in range(100):
            large_users.append({
                'last_login': '2024-01-15',
                'posts': i,
                'comments': i,
                'likes': i,
                'days_active': 10
            })

        try:
            result = func(large_users, '2024-01-01', '2024-01-31')

            # Check if we get exactly 5 top performers (or less if fewer users)
            top_performers = result.get('top_performers', [])

            if len(top_performers) == 5:
                # Check if they are actually the top 5 (highest posts should be 99, 98, 97, 96, 95)
                top_posts = [user.get('posts', 0) for user in top_performers]
                if top_posts == [99, 98, 97, 96, 95]:
                    return 10, 1, "✅ PASS: Efficient implementation - correct top 5 selection"
                elif 99 in top_posts and len(set(top_posts)) == 5:
                    return 8, 1, "✅ PASS: Correct top 5 but possibly inefficient sorting"
                else:
                    return 3, 0, "_test_efficient_implementation: ⚠️  PARTIAL: Top 5 selection has issues"
            elif len(top_performers) < 5 and len(large_users) >= 5:
                return 2, 0, "_test_efficient_implementation: ⚠️  PARTIAL: Not returning enough top performers"
            else:
                return 0, 0, "_test_efficient_implementation: ❌ FAIL: Top performers selection broken"
        except Exception as e:
            return 0, 0, f"_test_efficient_implementation: ❌ FAIL: Function crashes on larger dataset - {type(e).__name__}: {e}"
        if percentage >= 90: return "A (Excellent - Production Ready)"
        elif percentage >= 80: return "B (Good - Minor Issues)"
        elif percentage >= 70: return "C (Fair - Several Bugs)"
        elif percentage >= 60: return "D (Poor - Major Issues)"
        else: return "F (Failing - Critical Bugs)"