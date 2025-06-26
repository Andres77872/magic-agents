#!/usr/bin/env python3
"""
Comprehensive test runner for magic-agents project.
Runs all test suites and provides a summary report.
"""

import asyncio
import time
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test classes
from test_comprehensive_flows import TestComprehensiveFlows, run_all_tests as run_comprehensive
from test_advanced_flows import TestAdvancedFlows, run_advanced_tests
from test_advanced_flows_fixed import TestAdvancedFlowsFixed, run_fixed_advanced_tests
from test_edge_cases import TestEdgeCases, run_edge_case_tests
from test_run1 import test_run_agent, test_run_agent_loop

# Load API keys
var_env = json.load(open('/home/andres/Documents/agents_key.json'))


class TestRunner:
    """Main test runner with reporting capabilities."""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    async def run_test_suite(self, suite_name, test_func):
        """Run a single test suite."""
        print(f"\n{'='*80}")
        print(f"Running {suite_name}")
        print(f"{'='*80}")
        
        try:
            # Handle different types of test functions
            if hasattr(test_func, '__name__') and 'run_fixed_advanced_tests' in test_func.__name__:
                # For our fixed tests that don't use asyncio.run()
                import test.test_advanced_flows_fixed as fixed_tests
                test_suite = fixed_tests.TestAdvancedFlowsFixed()
                test_suite.setup_method()
                
                # Run each test individually
                tests = [
                    test_suite.test_send_message_with_extras(),
                    test_suite.test_deeply_nested_inner_flows_fixed(),
                    test_suite.test_parser_to_sendmessage_flow(),
                    test_suite.test_loop_with_sendmessage_aggregation()
                ]
                
                for i, test in enumerate(tests, 1):
                    print(f"\n{'='*60}")
                    print(f"Running Fixed Advanced Test {i}")
                    print(f"{'='*60}")
                    try:
                        await test
                        print(f"✓ Fixed Advanced Test {i} passed")
                    except Exception as e:
                        print(f"✗ Fixed Advanced Test {i} failed: {e}")
                        raise e
            elif asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            print(f"\n✅ {suite_name} completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ {suite_name} failed with error: {e}")
            self.results['errors'].append({
                'suite': suite_name,
                'error': str(e)
            })
            return False
    
    async def run_individual_tests(self):
        """Run individual async tests from test_run1.py"""
        print(f"\n{'='*80}")
        print("Running Individual Tests from test_run1.py")
        print(f"{'='*80}")
        
        # Test 1: Browsing agent
        try:
            print("\n--- Test 1: Browsing Agent ---")
            await test_run_agent()
            print("\n✅ Browsing agent test passed")
            self.results['passed'] += 1
        except Exception as e:
            print(f"\n❌ Browsing agent test failed: {e}")
            self.results['failed'] += 1
            self.results['errors'].append({
                'test': 'Browsing Agent',
                'error': str(e)
            })
        
        # Test 2: Loop agent
        try:
            print("\n--- Test 2: Loop Agent ---")
            await test_run_agent_loop()
            print("\n✅ Loop agent test passed")
            self.results['passed'] += 1
        except Exception as e:
            print(f"\n❌ Loop agent test failed: {e}")
            self.results['failed'] += 1
            self.results['errors'].append({
                'test': 'Loop Agent',
                'error': str(e)
            })
    
    def generate_report(self):
        """Generate a test report."""
        duration = self.results['end_time'] - self.results['start_time']
        
        print("\n" + "="*80)
        print("TEST SUMMARY REPORT")
        print("="*80)
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Tests Passed: {self.results['passed']}")
        print(f"Tests Failed: {self.results['failed']}")
        print(f"Total Tests: {self.results['passed'] + self.results['failed']}")
        
        if self.results['errors']:
            print("\nERRORS:")
            for error in self.results['errors']:
                if 'suite' in error:
                    print(f"  - {error['suite']}: {error['error']}")
                else:
                    print(f"  - {error['test']}: {error['error']}")
        
        print("\n" + "="*80)
        
        # Save report to file
        report_path = Path(__file__).parent / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to: {report_path}")
    
    async def run_all(self):
        """Run all test suites."""
        self.results['start_time'] = time.time()
        
        # Run test suites
        suites = [
            ("Individual Tests", self.run_individual_tests),
            ("Comprehensive Flow Tests", run_comprehensive),
            ("Advanced Flow Tests (Fixed)", run_fixed_advanced_tests),
            ("Edge Case Tests", run_edge_case_tests)
        ]
        
        for suite_name, test_func in suites:
            success = await self.run_test_suite(suite_name, test_func)
            if success:
                self.results['passed'] += 1
            else:
                self.results['failed'] += 1
        
        self.results['end_time'] = time.time()
        self.generate_report()


async def run_specific_test(test_name):
    """Run a specific test suite by name."""
    runner = TestRunner()
    
    test_map = {
        'comprehensive': run_comprehensive,
        'advanced': run_advanced_tests,
        'advanced_fixed': run_fixed_advanced_tests,
        'edge': run_edge_case_tests,
        'individual': runner.run_individual_tests
    }
    
    if test_name in test_map:
        runner.results['start_time'] = time.time()
        success = await runner.run_test_suite(f"{test_name.title()} Tests", test_map[test_name])
        if success:
            runner.results['passed'] += 1
        else:
            runner.results['failed'] += 1
        runner.results['end_time'] = time.time()
        runner.generate_report()
    else:
        print(f"Unknown test suite: {test_name}")
        print(f"Available suites: {', '.join(test_map.keys())}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run magic-agents tests')
    parser.add_argument(
        '--suite',
        choices=['all', 'comprehensive', 'advanced', 'advanced_fixed', 'edge', 'individual'],
        default='all',
        help='Which test suite to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                  Magic Agents Test Runner                      ║
║                                                               ║
║  Running: {args.suite:<20}                             ║
║  API Keys: Loaded from agents_key.json                       ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    if args.suite == 'all':
        runner = TestRunner()
        asyncio.run(runner.run_all())
    else:
        asyncio.run(run_specific_test(args.suite))


if __name__ == "__main__":
    main() 