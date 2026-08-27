"""
scripts/setup_test_data.py — Create sample financial documents for testing

Run with:
    python scripts/setup_test_data.py

This creates sample documents in data/raw/ so you can test the assistant
without having to provide your own financial data.
"""

from pathlib import Path


def setup_test_data():
    """Create sample financial documents."""
    
    # Create data/raw directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("📄 Creating sample financial documents...\n")
    
    # Sample 1: Bank Statement
    bank_statement = """ACME BANK STATEMENT
Account Number: 1234-5678-9012
Period: January 2024
Statement Date: February 1, 2024

ACCOUNT SUMMARY
Opening Balance (Jan 1):        $10,000.00
Total Deposits:                  $5,000.00
Total Withdrawals:              ($1,350.00)
Closing Balance (Jan 31):       $13,650.00

DETAILED TRANSACTIONS
Date        | Description              | Debit      | Credit    | Balance
------------|--------------------------|------------|-----------|----------
2024-01-01  | Opening Balance          |            |           | $10,000
2024-01-15  | Direct Deposit - Salary  |            | $5,000    | $15,000
2024-01-20  | Check #101 - Rent       | $1,200     |           | $13,800
2024-01-25  | Utility Payment         | $150       |           | $13,650

ANALYSIS
Average Daily Balance: $12,450
Interest Earned: $2.50
Service Fees: $0.00
"""
    
    bank_file = raw_dir / "bank_statement_jan2024.txt"
    bank_file.write_text(bank_statement)
    print(f"✓ Created: {bank_file.name}")
    
    # Sample 2: Quarterly Financial Report
    report = """QUARTERLY FINANCIAL REPORT
Company: TechCorp Inc.
Quarter: Q3 2024
Fiscal Year: 2024
Report Date: October 15, 2024

EXECUTIVE SUMMARY
Strong Q3 performance with 12% revenue growth YoY and 30% net margin.
All operating divisions exceeded targets.

REVENUE
Product Sales:              $150,000 (60.0% of total)
Service Revenue:             $75,000 (30.0% of total)
Licensing Fees:              $25,000 (10.0% of total)
────────────────────────────────────
Total Revenue:              $250,000 (100%)

Year-over-Year Growth: 12% (vs. $223,000 in Q3 2023)

COST BREAKDOWN
Cost of Goods Sold:        $100,000 (40% of revenue)
Operating Expenses:         $60,000 (24% of revenue)
  - Salaries & Benefits:    $40,000
  - Marketing:              $12,000
  - Utilities & Facilities:  $8,000
Administrative:             $15,000 (6% of revenue)
────────────────────────────────────
Total Expenses:            $175,000 (70%)

PROFITABILITY ANALYSIS
Gross Profit:              $150,000 (60% margin)
Operating Income:           $90,000 (36% margin)
Net Income:                 $75,000 (30% margin)

CASH FLOW STATEMENT
Operating Cash Flow:        $80,000
Capital Expenditures:      ($20,000)
Free Cash Flow:             $60,000

BALANCE SHEET SNAPSHOT
Total Assets:              $500,000
Current Assets:            $250,000
  - Cash Reserve:          $150,000
  - Accounts Receivable:   $100,000
Total Liabilities:         $200,000
Shareholders' Equity:      $300,000

KEY METRICS
Current Ratio: 1.25
Debt-to-Equity: 0.67
Return on Equity: 25%
Earnings Per Share: $2.50 (assuming 30M shares)

SEGMENT PERFORMANCE
Product Sales:   +15% YoY
Services:        +8% YoY
Licensing:       +20% YoY

OUTLOOK
Expected Q4 revenue: $265,000 (continued growth trajectory)
Full-year projection: $950,000
"""
    
    report_file = raw_dir / "financial_report_q3_2024.txt"
    report_file.write_text(report)
    print(f"✓ Created: {report_file.name}")
    
    # Sample 3: Tax Document
    tax_document = """TAX DOCUMENTATION
Tax Year: 2023
Taxpayer: Jane Doe, SSN: XXX-XX-1234
Filing Date: April 15, 2024

INCOME SUMMARY
W-2 Wages from Employer A:      $75,000
1099 Freelance Income:           $25,000
Interest Income:                    $500
Dividend Income:                  $1,000
Capital Gains (Long-term):        $3,000
────────────────────────────────
Adjusted Gross Income (AGI):    $104,500

DEDUCTIONS
Standard Deduction:              $13,850
Alternative: Itemized Deductions:
  - State & Local Taxes (SALT):   $8,000
  - Mortgage Interest:            $12,000
  - Charitable Contributions:      $3,000
                                 --------
  Total Itemized:                $23,000
Deduction Taken:                 $23,000 (itemized > standard)

TAXABLE INCOME:                  $81,500

TAX CALCULATION
Federal Income Tax:              $9,450
Self-Employment Tax:              $3,532
State Income Tax:                 $2,445
────────────────────────────────
Total Tax Liability:             $15,427

CREDITS & PAYMENTS
Withheld from W-2:               $11,000
Estimated Payments:               $3,000
────────────────────────────────
Total Payments:                  $14,000

REFUND DUE:                       -$1,427

NOTES
- Qualified Dividend Income: $1,000 (taxed at 15%)
- Capital Loss Carryover: $0
- Dependent: 1 (provides $2,000 child tax credit)
"""
    
    tax_file = raw_dir / "tax_return_2023.txt"
    tax_file.write_text(tax_document)
    print(f"✓ Created: {tax_file.name}")
    
    # Sample 4: Investment Portfolio
    portfolio = """INVESTMENT PORTFOLIO STATEMENT
Account Holder: John Smith
Account Number: INV-2024-001
Statement Period: January 1 - March 31, 2024
Generated: April 10, 2024

HOLDINGS SUMMARY
Stock Portfolio:                $125,000 (50%)
Bond Portfolio:                 $75,000 (30%)
Cash & Money Market:            $50,000 (20%)
────────────────────────────────
Total Portfolio Value:         $250,000

DETAILED HOLDINGS
STOCKS (50% allocation, $125,000)
Apple Inc. (AAPL):
  Shares: 200
  Cost Basis: $22,000
  Current Value: $35,000
  Gain/Loss: +$13,000

Microsoft Corp. (MSFT):
  Shares: 150
  Cost Basis: $28,000
  Current Value: $45,000
  Gain/Loss: +$17,000

Tesla Inc. (TSLA):
  Shares: 50
  Cost Basis: $18,000
  Current Value: $22,000
  Gain/Loss: +$4,000

S&P 500 Index Fund (VOO):
  Shares: 150
  Cost Basis: $52,000
  Current Value: $58,000
  Gain/Loss: +$6,000

Other Holdings:            $65,000
Total Stocks:             $125,000

BONDS (30% allocation, $75,000)
Treasury Securities:       $35,000 (Duration: 5 years)
Corporate Bonds:           $25,000 (Yield: 4.5%)
Municipal Bonds:           $15,000 (Tax-exempt)

CASH (20% allocation, $50,000)
Money Market Account:      $30,000 (Yield: 5.0%)
Cash Reserve:              $20,000

PERFORMANCE SUMMARY
Cost Basis (Total):        $215,000
Current Value:             $250,000
Unrealized Gain:            $35,000
Return: 16.3%

Year-to-Date Performance:
Q1 2024: +$8,500 (+3.5%)
YTD Return: +3.5%

DIVIDENDS & INTEREST (Q1 2024)
Stock Dividends:             $625
Bond Interest:               $875
Money Market Interest:       $375
────────────────────────────
Total Q1 Income:           $1,875

Annual Projections:
Projected Dividend Income: $2,500
Projected Interest Income: $3,500
────────────────────────────
Projected Annual Income:   $6,000
"""
    
    portfolio_file = raw_dir / "investment_portfolio_q1_2024.txt"
    portfolio_file.write_text(portfolio)
    print(f"✓ Created: {portfolio_file.name}")
    
    print(f"\n✅ Setup complete!")
    print(f"📁 Documents created in: {raw_dir.absolute()}")
    print(f"\nYou can now run:")
    print(f"   python main.py")
    print(f"\nAnd ask questions like:")
    print(f"   - 'What was the total revenue in Q3?'")
    print(f"   - 'How much was spent on rent and utilities combined?'")
    print(f"   - 'What is my net income after taxes?'")
    print(f"   - 'What stocks do I own and what are they worth?'")


if __name__ == "__main__":
    setup_test_data()
