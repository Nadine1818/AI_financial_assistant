"""
scripts/test_scenarios.py — Interactive scenario testing tool

Run with:
    python scripts/test_scenarios.py

This script lets you test specific scenarios without running the full app.
"""

import json
from datetime import datetime
from pathlib import Path

from app.generation.response_generator import generate_with_history
from app.validation.verifier import verify
from app.retrieval.retriever import retrieve
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioTester:
    """
    Interactive tool for testing scenarios.
    
    Usage:
        1. Choose a scenario category
        2. Pick specific scenario
        3. Ask a question
        4. Review response, sources, verification
    """
    
    def __init__(self):
        self.chat_history = []
        self.test_results = []
        self.start_time = datetime.now()
    
    def print_header(self, text):
        """Print formatted header."""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    
    def print_section(self, text):
        """Print section divider."""
        print(f"\n{text}")
        print("-" * len(text))
    
    def test_bank_statement_scenario(self):
        """Test with bank statement scenario."""
        self.print_section("💳 Bank Statement Scenario Tests")
        
        scenarios = [
            {
                "name": "Direct Transaction Query",
                "question": "What was the rent payment in January?",
                "expected": "Should cite specific amount from bank_statement.txt",
            },
            {
                "name": "Sum Calculation",
                "question": "How much was spent on rent and utilities combined?",
                "expected": "Should sum rent ($1,200) + utilities ($150) = $1,350",
            },
            {
                "name": "Balance Inquiry",
                "question": "What was the opening balance?",
                "expected": "Should return $10,000 with source citation",
            },
            {
                "name": "Change Calculation",
                "question": "By how much did the balance change in January?",
                "expected": "$13,650 - $10,000 = $3,650 increase",
            },
        ]
        
        return self.run_scenario_batch(scenarios, "bank_statement")
    
    def test_financial_report_scenario(self):
        """Test with financial report scenario."""
        self.print_section("📊 Financial Report Scenario Tests")
        
        scenarios = [
            {
                "name": "Revenue Query",
                "question": "What was the total revenue?",
                "expected": "Should return $250,000 from financial_report_q3_2024.txt",
            },
            {
                "name": "Percentage Calculation",
                "question": "What percentage of revenue came from services?",
                "expected": "$75K / $250K = 30%",
            },
            {
                "name": "Profitability Analysis",
                "question": "Is the company profitable?",
                "expected": "Yes, with $75K net income (30% margin)",
            },
            {
                "name": "Cash Flow Analysis",
                "question": "What was the free cash flow?",
                "expected": "$60,000 (operating cash flow - capex)",
            },
            {
                "name": "Revenue Breakdown",
                "question": "Which revenue source was largest?",
                "expected": "Product sales at $150,000 (60% of total)",
            },
        ]
        
        return self.run_scenario_batch(scenarios, "financial_report")
    
    def test_edge_cases_scenario(self):
        """Test edge cases."""
        self.print_section("⚠️  Edge Case Scenario Tests")
        
        scenarios = [
            {
                "name": "Out of Context",
                "question": "What is the stock price?",
                "expected": "Should refuse: 'I don't have enough information'",
                "pass_if": "refuses to answer",
            },
            {
                "name": "Ambiguous Question",
                "question": "What is the profit margin?",
                "expected": "Should ask which type (gross, operating, net)",
                "pass_if": "asks for clarification OR lists all types",
            },
            {
                "name": "Missing Data",
                "question": "What was the Q4 revenue?",
                "expected": "Should state data not available",
                "pass_if": "explicitly says data is missing",
            },
            {
                "name": "Forecasting Refusal",
                "question": "When will we reach $500K in revenue?",
                "expected": "Should refuse to forecast",
                "pass_if": "refuses to speculate",
            },
        ]
        
        return self.run_scenario_batch(scenarios, "edge_cases")
    
    def test_multi_turn_scenario(self):
        """Test multi-turn conversation."""
        self.print_section("🔄 Multi-Turn Conversation Scenario")
        
        turns = [
            {
                "turn": 1,
                "question": "What was our revenue last quarter?",
                "expected": "Should return specific amount",
            },
            {
                "turn": 2,
                "question": "How much of that came from services?",
                "expected": "Should understand 'that' = revenue, and find services portion",
            },
            {
                "turn": 3,
                "question": "What was the profit margin then?",
                "expected": "Should reference the same quarter",
            },
        ]
        
        print(f"Running {len(turns)} conversation turns...\n")
        
        for turn in turns:
            print(f"Turn {turn['turn']}: {turn['question']}")
            print(f"Expected: {turn['expected']}")
            
            try:
                result = generate_with_history(turn['question'], self.chat_history)
                
                print(f"✓ Answer: {result.answer[:100]}...")
                print(f"  Sources: {', '.join(result.sources) if result.sources else '(none)'}")
                print(f"  Latency: {result.metadata.get('llm_ms', 'N/A')}ms\n")
                
                self.chat_history.append((turn['question'], result.answer))
                
            except Exception as e:
                print(f"✗ Error: {e}\n")
    
    def run_scenario_batch(self, scenarios, category):
        """Run a batch of scenarios."""
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
            print(f"Question: {scenario['question']}")
            print(f"Expected: {scenario['expected']}")
            
            try:
                # Reset chat history for independent scenarios
                self.chat_history = []
                
                # Generate response
                result = generate_with_history(scenario['question'], self.chat_history)
                
                # Verify
                verify_result = verify(result)
                
                # Display
                print(f"✓ Answer: {result.answer[:150]}...")
                print(f"  Sources: {', '.join(result.sources) if result.sources else '(none)'}")
                if verify_result:
                    print(f"  Verdict: {verify_result.verdict}")
                    print(f"  Confidence: {'HIGH' if verify_result.verdict == 'PASS' else 'LOW'}")
                
                # Track result
                results.append({
                    'scenario': scenario['name'],
                    'question': scenario['question'],
                    'answer': result.answer[:200],
                    'sources': result.sources,
                    'verdict': verify_result.verdict if verify_result else 'UNVERIFIED',
                    'passed': verify_result.verdict == 'PASS' if verify_result else False,
                })
                
                # Brief pause
                print()
                
            except Exception as e:
                print(f"✗ Error: {type(e).__name__}: {str(e)[:100]}\n")
                results.append({
                    'scenario': scenario['name'],
                    'question': scenario['question'],
                    'error': str(e),
                    'passed': False,
                })
        
        return results
    
    def display_summary(self):
        """Display summary of all tests."""
        if not self.test_results:
            return
        
        self.print_header("Test Summary")
        
        passed = sum(1 for r in self.test_results if r.get('passed', False))
        total = len(self.test_results)
        
        print(f"Total Scenarios: {total}")
        print(f"Passed: {passed} ({100*passed//total if total else 0}%)")
        print(f"Failed: {total - passed}")
        
        if passed < total:
            print("\n⚠️  Failed scenarios:")
            for r in self.test_results:
                if not r.get('passed', False):
                    print(f"  - {r.get('scenario', 'Unknown')}")
                    if 'error' in r:
                        print(f"    Error: {r['error'][:80]}")
    
    def save_results(self):
        """Save test results to file."""
        if not self.test_results:
            return
        
        results_file = Path(f"test_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json")
        
        summary = {
            'timestamp': self.start_time.isoformat(),
            'total_scenarios': len(self.test_results),
            'passed': sum(1 for r in self.test_results if r.get('passed', False)),
            'results': self.test_results,
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    def interactive_mode(self):
        """Free-form question mode."""
        self.print_header("🤖 Interactive Mode")
        
        print("Ask any question about your financial documents.")
        print("Commands: 'reset' (clear history), 'quit' (exit)\n")
        
        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                break
            
            if question.lower() == 'reset':
                self.chat_history.clear()
                print("✓ History cleared\n")
                continue
            
            print("\nProcessing...")
            
            try:
                result = generate_with_history(question, self.chat_history)
                
                print(f"\nAssistant: {result.answer}\n")
                
                if result.sources:
                    print(f"Sources: {', '.join(result.sources)}")
                
                verify_result = verify(result)
                if verify_result:
                    print(f"Verdict: {verify_result.verdict}")
                
                print()
                
                self.chat_history.append((question, result.answer))
                
            except Exception as e:
                print(f"Error: {e}\n")
    
    def run_all(self):
        """Run all scenario tests."""
        self.print_header("Running All Scenario Tests")
        
        # Bank statement scenarios
        try:
            results = self.test_bank_statement_scenario()
            self.test_results.extend(results)
        except Exception as e:
            print(f"⚠️  Bank statement tests failed: {e}")
        
        # Financial report scenarios
        try:
            results = self.test_financial_report_scenario()
            self.test_results.extend(results)
        except Exception as e:
            print(f"⚠️  Financial report tests failed: {e}")
        
        # Edge cases
        try:
            results = self.test_edge_cases_scenario()
            self.test_results.extend(results)
        except Exception as e:
            print(f"⚠️  Edge case tests failed: {e}")
        
        # Multi-turn
        try:
            self.test_multi_turn_scenario()
        except Exception as e:
            print(f"⚠️  Multi-turn tests failed: {e}")
        
        self.display_summary()
        self.save_results()


def main():
    """Entry point."""
    import sys
    
    tester = ScenarioTester()
    
    print("\n🎯 Financial Assistant Scenario Tester")
    print("=" * 60)
    print("\nOptions:")
    print("  1 - Run all scenarios")
    print("  2 - Test bank statements only")
    print("  3 - Test financial reports only")
    print("  4 - Test edge cases only")
    print("  5 - Multi-turn conversation test")
    print("  6 - Interactive mode (free-form questions)")
    print("  0 - Exit")
    
    choice = input("\nSelect (0-6): ").strip()
    
    if choice == '1':
        tester.run_all()
    elif choice == '2':
        tester.test_results = tester.test_bank_statement_scenario()
        tester.display_summary()
    elif choice == '3':
        tester.test_results = tester.test_financial_report_scenario()
        tester.display_summary()
    elif choice == '4':
        tester.test_results = tester.test_edge_cases_scenario()
        tester.display_summary()
    elif choice == '5':
        tester.test_multi_turn_scenario()
    elif choice == '6':
        tester.interactive_mode()
    else:
        print("Goodbye.")
        return
    
    tester.save_results()


if __name__ == "__main__":
    main()
