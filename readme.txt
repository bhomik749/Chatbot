Financial Data Chatbot Assignment

1. Overview
This project implements a chatbot using LangGraph. It is designed to answer financial questions based on two provided CSV files (`trades.csv` and `holdings.csv`). 
The bot ensures mathematical accuracy by reading and preprocessing data using pandas library before passing the data to the LLM for natural language formatting.

2. Architecture(The Summary Node)
I utilized a Deterministic Data Pre-processing through pandas followed by node based approach using LangGraph.

3. The Problem with Standard Approaches:
- When passing pandas object directly to langgraph/llm provides bad results, as llm often hallucinates on raw extracted data.
- Code Generation Risks: Letting an LLM to do the preprocessing on its own proved unstable results or exceeded per minute API calls for Gemini.

4. Solution(Two-Node Graph):
 Node 1: Preprocessing
    - Uses Pandas for aggregating data before the query is processed.
    - Calculates `Trade Counts`, `Holdings Counts`, and `Yearly PnL` for every fund.
    - Generates a text-based summary.
    - Benefit: Deterministic.

 Node 2: Analyst 
    - Receives the User Question and the final summary.
    - Acts as a semantic router:
        - If the answer is in the summary -> provides a formatted response.
        - If the answer is missing -> Returns the strict fallback: "Sorry can not find the answer".
    - Benefit: Eliminates hallucinations and strictly adheres to scope constraints.

5. Tech Stack
- Framework: LangGraph (State management), LangChain (messaging and llm interface).
- LLM: Llama-3.3-70b-versatile(via Groq API)
- Data Processing: Pandas.

6. Setup & Execution
-  Put `trades.csv` and `holdings.csv` in root directory.
-  Install dependencies:
    `pip install langgraph langchain-groq pandas python-dotenv`
-  Set your API Key in the `.env` file 
-  Run the script through terminal using `python app.py`.

7. Test Cases 
- Trade Counts: "Total number of trades for HoldCo 1?" -> Returns 43.
- Performance: "Which funds performed better based on PnL?" -> Returns funds ranked by `PL_YTD` sum.
- Holdings Check: "Total number of holdings for Heather?" -> Verifies data retrieval from the holdings file.
- Out of Scope: "Who won the FIFA world cup?" -> Triggers the "Sorry cannot find the answer" .
