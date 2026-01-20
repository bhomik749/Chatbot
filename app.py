import os
import pandas as pd
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

groq_api_key = os.getenv("groq_api_key")
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile", # High accuracy, fast model
    api_key=groq_api_key
)

trades_df = pd.read_csv('trades.csv')
holdings_df = pd.read_csv('holdings.csv')
print("\n data loaded!!")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Conversation history"]
    data_summary: str 

# Instead of letting the LLM do math, pre-calculate the stats in Python.
def preprocessing_node(state: AgentState):

    #converting all protfolio column rows into a dictionary.
    #{'HoldCo 1': 43, 'Fund B': 12}
    trade_counts = trades_df.groupby('PortfolioName').size().to_dict()
    
    #groups by fund name, takes the `PL_YTD` column, sums it up, and rounds to 2 decimal places
    pnl_stats = holdings_df.groupby('PortfolioName')['PL_YTD'].sum().round(2).to_dict()
    
    holding_counts = holdings_df.groupby('PortfolioName').size().to_dict()
    
    summary_lines = ["FINANCIAL DATA REPORT"]
    
    #set union operation to get all unique funds from both files
    all_funds = set(trade_counts.keys()) | set(pnl_stats.keys())
    
    for fund in all_funds:
        t_count = trade_counts.get(fund, 0)
        h_count = holding_counts.get(fund, 0)
        perf = pnl_stats.get(fund, 0.0)
        
        line = f"FUND: {fund} | TRADES: {t_count} | HOLDINGS: {h_count} | YEARLY_PnL: {perf}"
        summary_lines.append(line)
        
    summary_text = "\n".join(summary_lines)
    
    return {"data_summary": summary_text}

#reading the report
def analyst_node(state: AgentState):
    query = state['messages'][-1].content
    summary = state['data_summary']
    
    system_prompt = """You are a strictly grounded Financial Assistant.
    You have been given a DATA REPORT.
    
    RULES:
    1. Answer the user's question using ONLY the DATA REPORT provided below.
    2. If the answer is found, answer concisely.
    3. CRITICAL: If the answer is NOT in the DATA REPORT, you must output EXACTLY: 
       "Sorry can not find the answer"
    """
    
    user_prompt = f"""
    DATA REPORT:
    {summary}
    
    USER QUESTION: 
    {query}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {"messages": [response]}

workflow = StateGraph(AgentState)

workflow.add_node("preprocess", preprocessing_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

print("\n running test cases-->")

questions = [
    "Total number of trades for HoldCo 1?", 
    "Which funds performed better depending on the yearly Profit and Loss?", 
    "Who won the FIFA world cup?", 
    "What is the total number of holdings for Heather?"
]

for q in questions:
    print(f"\n User: {q}")
    result = app.invoke({"messages": [HumanMessage(content=q)]})
    print(f"\n Bot:  {result['messages'][-1].content}")
