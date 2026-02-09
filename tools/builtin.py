from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool(description="Perform basic arithmetic on two numbers")
def calculator(a: float, b: float, op: str):
    """
    Perform a basic arithmetic operation.

    Args:
        a: First number
        b: Second number
        op: Operation to perform. One of: add, sub, mul, div

    Returns:
        Dictionary containing the result of the operation.
    """
    if op == "add":
        return {"result": a + b}
    if op == "sub":
        return {"result": a - b}
    if op == "mul":
        return {"result": a * b}
    if op == "div":
        if b == 0:
            return {"error": "Division by zero"}
        return {"result": a / b}

    return {"error": f"Unsupported operation: {op}"}

@tool
def get_stock_price(symbol: str):
    """
    Fetch the latest stock price for a given stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, TSLA)

    Returns:
        Raw JSON response from the Alpha Vantage API.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    response = requests.get(url, timeout=10)
    return response.json()
