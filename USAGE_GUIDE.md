# Financial AI Assistant — Usage & Testing Guide

## 🚀 How to Use the Model

### Quick Start

```bash
# 1. Set up environment
export OPENAI_API_KEY="sk-..."  # Your OpenAI API key

# 2. Add financial documents
# Place .pdf, .csv, .txt, or .json files in:
# → data/raw/

# 3. Run the assistant
python main.py

# 4. Start asking questions
You: What was the total revenue in Q3?
Assistant: According to bank_statement.pdf, the total revenue in Q3 was $450,000...
```

### What Happens Behind the Scenes

```
Your Question
     ↓
[RETRIEVER] → Semantic search in ChromaDB
     ↓
[GENERATOR] → Build RAG prompt with context
     ↓
[LLM] → Generate answer citing sources
     ↓
[VERIFIER] → Check if answer is grounded in context
     ↓
Safe Answer with Sources
```

---

## 📊 Real Data Examples

### Example 1: Bank Statement Analysis

**Input Document** (`data/raw/bank_statement_2024.csv`):
```csv
Date,Description,Amount,Balance
2024-01-15,Opening Balance,0,10000
2024-01-20,Salary Deposit,5000,15000
2024-01-25,Rent Payment,-1200,13800
2024-02-01,Utility Bill,-150,13650
2024-02-15,Salary Deposit,5000,18650
```

#### Test Case 1: Direct Factual Query
```
Question: What was the salary deposit in February?
Expected Answer: According to bank_statement_2024.csv, the salary deposit 
                 in February was $5,000 on 2024-02-15.
Expected Sources: bank_statement_2024.csv
Expected Verdict: PASS (answer is explicitly in context)
```

#### Test Case 2: Calculation Query
```
Question: How much did the user spend on rent and utilities combined?
Expected Answer: According to bank_statement_2024.csv, the rent payment was 
                 $1,200 and the utility bill was $150, for a combined total 
                 of $1,350.
Expected Sources: bank_statement_2024.csv
Expected Verdict: PASS (math is derived from explicit data)
```

#### Test Case 3: Out-of-Context Query
```
Question: What was the stock performance in Q3?
Expected Answer: I don't have enough information in the provided documents 
                 to answer this question.
Expected Sources: (none)
Expected Verdict: PASS (correctly refused to hallucinate)
```

---

### Example 2: Financial Report Analysis

**Input Document** (`data/raw/annual_report_2023.txt`):
```
ACME CORPORATION
Annual Report 2023

FINANCIAL SUMMARY
Revenue: $50M
Operating Expenses: $30M
Net Income: $20M
Gross Margin: 60%

SEGMENT BREAKDOWN
- Product Sales: $35M (70%)
- Services: $12M (24%)
- Licensing: $3M (6%)

CASH FLOW
Operating Cash Flow: $18M
Capital Expenditures: $5M
Free Cash Flow: $13M
```

#### Test Case 4: Multi-Step Reasoning
```
Question: What percentage of revenue came from services?
Expected Answer: According to annual_report_2023.txt, services generated 
                 $12M out of total revenue of $50M, which represents 24%.
Expected Sources: annual_report_2023.txt
Expected Verdict: PASS
```

#### Test Case 5: Comparative Query
```
Question: How much higher was revenue than operating expenses?
Expected Answer: According to annual_report_2023.txt, revenue was $50M and 
                 operating expenses were $30M, making revenue $20M higher 
                 than expenses.
Expected Sources: annual_report_2023.txt
Expected Verdict: PASS
```

#### Test Case 6: Implicit Information
```
Question: Is the company profitable?
Expected Answer: Yes, according to annual_report_2023.txt, the company had 
                 net income of $20M in 2023, indicating profitability.
Expected Sources: annual_report_2023.txt
Expected Verdict: PASS (reasonable inference from explicit data)
```

---

### Example 3: Multi-Document Scenario

**Document 1** (`Q1_report.txt`):
```
Q1 2024 Financial Results
Revenue: $25M
Profit: $8M
```

**Document 2** (`Q2_report.txt`):
```
Q2 2024 Financial Results
Revenue: $28M
Profit: $9M
```

#### Test Case 7: Cross-Document Query
```
Question: How much did revenue grow from Q1 to Q2?
Expected Answer: According to Q1_report.txt and Q2_report.txt, revenue grew 
                 from $25M in Q1 to $28M in Q2, an increase of $3M or 12%.
Expected Sources: Q1_report.txt, Q2_report.txt
Expected Verdict: PASS
```

---

## ⚠️ Edge Cases to Test

### 1. **Ambiguous Numbers Without Units**
```
Document: "The account value increased by 500"
Question: What was the increase?

Expected: I don't have enough information to answer this question. The 
          document doesn't specify if the increase was in dollars, 
          thousands, or millions.

Why: Financial data without currency symbols is ambiguous
```

### 2. **Negative Numbers in Accounting Format**
```
Document: "Loss: (2,500,000) — shown in parentheses per accounting standards"
Question: What was the loss?

Expected: According to the document, there was a loss of $2,500,000.

Why: Accountants use parentheses for negatives; LLM should interpret correctly
```

### 3. **Multiple Sections with Same Metric**
```
Document: 
  "Operating Margin: 15%
   Net Profit Margin: 8%
   Gross Profit Margin: 42%"
   
Question: What is the profit margin?

Expected: The document contains multiple types of profit margins. Could you 
          clarify which one you're interested in? (Operating, Net, or Gross)

Why: Ambiguous question with multiple valid answers
```

### 4. **Time-Sensitive Data**
```
Document: "As of March 31, 2023, revenue was $15M"
Question Asked in 2024: What is the current revenue?

Expected: I don't have current data. According to available documents, 
          revenue was $15M as of March 31, 2023, but I cannot confirm 
          current figures.

Why: Old data shouldn't be presented as current
```

### 5. **Currency Conversion**
```
Document: "Revenue: £50M (GBP)"
Question: What was the revenue in USD?

Expected: I don't have exchange rate information to convert the figure. 
          The document shows revenue of £50M in British Pounds, but I 
          cannot provide the USD equivalent.

Why: Should not hallucinate exchange rates
```

### 6. **Percentage vs Absolute Value Confusion**
```
Document: "Gross Margin improved by 5 percentage points (from 40% to 45%)"
Question: By how much did gross margin improve?

Expected: Gross margin improved by 5 percentage points, from 40% to 45%.

Why: Should distinguish between "5 percentage points" and "5%"
```

### 7. **Rounding and Precision**
```
Document: "Net Income: $4,567,890.34"
Question: What was the net income?

Expected: According to the document, net income was $4,567,890.34 (or 
          approximately $4.57M).

Why: Should preserve precision from source or clearly state when rounding
```

### 8. **Missing Data Indicators**
```
Document: "Q3 Revenue: N/A (data pending audit)"
Question: What was the Q3 revenue?

Expected: According to the document, Q3 revenue data is not available 
          (pending audit completion).

Why: Should not fill in missing values
```

### 9. **Comparative Questions Without Context**
```
Document: "Our revenue of $50M is 15% higher than last year"
Question: Was our revenue higher than competitors?

Expected: I don't have competitor revenue data. The document only indicates 
          our revenue grew 15% year-over-year but doesn't provide competitor 
          comparisons.

Why: Should not invent external knowledge
```

### 10. **Implicit Negative Results**
```
Document: "Expenses: $30M, Revenue: $25M"
Question: Did the company make a profit?

Expected: No, according to the document, the company had a loss. Revenue 
          was $25M but expenses were $30M, resulting in a $5M loss.

Why: Should correctly identify negative results even when not explicitly stated
```

### 11. **Footnotes and Caveats**
```
Document: "Revenue: $100M* (*includes one-time licensing deal for $15M)"
Question: What was recurring revenue?

Expected: According to the document, total revenue was $100M, but this 
          includes a one-time licensing deal of $15M. The recurring revenue 
          would be approximately $85M.

Why: Should acknowledge when figures include non-recurring items
```

### 12. **Inconsistent Data Format**
```
Document: 
  "2024-01-01: Balance = $1,000,000
   Feb 2024: Balance = $1,200,000"
   
Question: What was the balance in January and February?

Expected: According to the document, the balance in January was $1,000,000 
          and in February was $1,200,000.

Why: Should handle different date formats
```

### 13. **Duplicate or Contradictory Data**
```
Document 1: "Q2 Revenue: $50M"
Document 2: "Q2 Revenue: $52M"

Question: What was Q2 revenue?

Expected: According to the available documents, there are conflicting figures 
          for Q2 revenue ($50M in document1.pdf and $52M in document2.pdf). 
          Please verify which figure is accurate.

Why: Should flag contradictions rather than pick one arbitrarily
```

### 14. **Complex Calculations**
```
Document: "Revenue: $1M, Tax Rate: 25%, Operating Expenses: 30% of Revenue"

Question: What was the net income after taxes?

Expected: I can calculate the operating expenses ($300K) from the given data, 
          but determining net income requires additional information about 
          cost of goods sold and other expenses.

Why: Should not attempt calculations beyond what's explicitly supported
```

### 15. **Seasonal Variations**
```
Document: "Q4 Revenue: $50M (includes holiday season spike)"

Question: What is typical monthly revenue?

Expected: I cannot determine typical monthly revenue from this data. Q4 was 
          reported as $50M including a holiday season spike, but without 
          baseline data for comparison, I cannot calculate typical monthly 
          figures.

Why: Should not assume uniform distribution without evidence
```

---

## 🧪 Testing with Real Data

### Step 1: Prepare Test Documents

```bash
mkdir -p data/raw/test_scenarios

# Create test document
cat > data/raw/test_scenarios/sample_financial.txt << 'EOF'
SAMPLE CORPORATION
Quarterly Financial Report Q3 2024

REVENUE
Product Sales:     $150,000
Service Revenue:   $75,000
Other Income:      $25,000
Total Revenue:     $250,000

EXPENSES
Cost of Goods:     $100,000
Operating Costs:   $60,000
Admin & Legal:     $15,000
Total Expenses:    $175,000

NET INCOME:        $75,000
Profit Margin:     30%

CASH POSITION
Operating Cash:    $100,000
Reserves:          $50,000
Total Liquidity:   $150,000
EOF
```

### Step 2: Run Interactive Test Session

```bash
# Start the assistant
python main.py

# You: What was the total revenue?
# Expected: Should cite sample_financial.txt with $250,000

# You: How much more was revenue than expenses?
# Expected: $250,000 - $175,000 = $75,000 (which equals net income)

# You: What percentage of revenue went to Cost of Goods?
# Expected: $100,000 / $250,000 = 40%

# You: What was the operating profit?
# Expected: Should state $75,000 net income shown in document

# You: When will we reach $500,000 in revenue?
# Expected: Should refuse to forecast/hallucinate
```

### Step 3: Programmatic Testing

```python
# tests/test_with_real_data.py
import pytest
from app.generation.response_generator import generate_with_history
from app.validation.verifier import verify
from app.retrieval.retriever import retrieve

def test_with_real_financial_data():
    """Test RAG pipeline with sample financial document."""
    
    # Question that should be answerable from sample_financial.txt
    question = "What was the total revenue in Q3 2024?"
    
    # Generate response
    gen_result = generate_with_history(
        question=question,
        chat_history=[]
    )
    
    # Verify answer
    ver_result = verify(gen_result)
    
    # Assertions
    assert "250000" in gen_result.answer.lower() or "250,000" in gen_result.answer
    assert "sample_financial.txt" in gen_result.sources
    assert ver_result.verdict == "PASS"
```

### Step 4: Check the Logs

```bash
# After running a session, check logs
tail -50 logs/assistant.log

# You'll see:
# 2024-05-30 10:15:23 | INFO | app.retrieval.retriever | Retrieved 2 chunk(s)
# 2024-05-30 10:15:24 | INFO | app.generation.llm | Invoking LLM | model=gpt-4o-mini
# 2024-05-30 10:15:25 | INFO | app.validation.verifier | Verification verdict: PASS
```

---

## ✅ Expected Behaviors Summary

| Scenario | Expected Behavior |
|----------|-------------------|
| **Direct factual match** | Return the exact value + source |
| **Simple arithmetic** | Calculate and show work from context |
| **Out-of-context question** | "I don't have enough information..." |
| **Ambiguous question** | Ask for clarification |
| **Multiple contradictions** | Flag both values and request verification |
| **Time-sensitive data** | Note the date data was from |
| **Complex inferences** | Acknowledge limitations clearly |
| **Missing details** | State what's missing, don't guess |
| **Formatting inconsistencies** | Normalize and cite both formats |
| **Data quality issues** | Explicitly note (N/A, estimates, etc.) |

---

## 🔧 Command Reference

```bash
# Interactive session
python main.py

# In conversation:
You: your question                    # Ask anything
You: reset                            # Clear conversation history
You: reingest                         # Reload documents from data/raw/
You: quit                             # Exit the program
```

---

## 📈 Performance Metrics to Monitor

After each test session:

```python
# From gen_result.metadata:
retrieval_ms      # How fast was semantic search? (should be <1000ms)
condense_ms       # Did history need condensing? (multi-turn only)
llm_ms            # How slow was the LLM? (typically 1000-5000ms)
chunk_count       # How many chunks were retrieved? (usually 3-5)
fallback_used     # Did we fail to retrieve context? (should be False)

# From ver_result:
verification_ms   # How fast was verification? (should be <5000ms)
verdict           # PASS, FAIL, or INCONCLUSIVE
```

---

## 🐛 Debugging Failed Responses

If an answer seems wrong:

```python
# 1. Check what was retrieved
question = "Your question"
docs = retrieve(question, top_k=5)
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...\n")

# 2. Check the raw LLM response (before verification)
from app.generation.response_generator import generate_with_history
result = generate_with_history(question, [])
print(result.answer)
print(result.sources)
print(result.metadata)

# 3. Check verification details
from app.validation.verifier import verify
verdict = verify(result)
print(f"Verdict: {verdict.verdict}")
print(f"Explanation: {verdict.explanation}")
```

