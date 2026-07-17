from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

@tool
def calculate_leave_balance(days_taken: int) -> str:
    """Calculate remaining vacation days given how many the employee has already taken. Company allows 15 days per year."""
    remaining = 15 - days_taken
    if remaining < 0:
        return f"Employee has exceeded their allowance by {-remaining} days."
    return f"Employee has {remaining} vacation days remaining."

model = ChatGroq(model="llama-3.3-70b-versatile")
#agent = create_agent(model=model, tools=[calculate_leave_balance])
agent = create_agent(
    model=model,
    tools=[calculate_leave_balance],
    system_prompt=(
        "You only have access to the tools explicitly provided to you. "
        "Never call a tool that wasn't given to you. "
        "If a question doesn't require your tools, answer directly from your own knowledge."
    ),
)
try:
    #result = agent.invoke({"messages": [{"role": "user", "content": "I've taken 6 vacation days so far this year. How many do I have left?"}]})
    result = agent.invoke({"messages": [{"role": "user", "content": "What's the capital of France?"}]})
    print(result["messages"][-1].content)
except Exception as e:
    print(f"Agent call failed: {e}")
# result = agent.invoke({"messages": [{"role": "user", "content": "What's the capital of France?"}]})
# print(result["messages"][-1].content)