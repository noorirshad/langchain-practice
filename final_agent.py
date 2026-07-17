from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- retriever setup, same as rag.py ---
docs = TextLoader("handbook.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- tool 1: search the handbook ---
@tool
def search_handbook(query: str) -> str:
    """Search the employee handbook for policy information like leave, expenses, or remote work."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the handbook."
    return "\n\n".join(d.page_content for d in results)

# --- tool 2: calculate leave balance ---
@tool
def calculate_leave_balance(days_taken: int) -> str:
    """Calculate remaining vacation days. Company allows 15 days per year."""
    remaining = 15 - days_taken
    return f"Employee has {remaining} vacation days remaining." if remaining >= 0 else f"Exceeded allowance by {-remaining} days."

model = ChatGroq(model="llama-3.3-70b-versatile")
agent = create_agent(
    model=model,
    tools=[search_handbook, calculate_leave_balance],
    system_prompt=(
        "You only have access to the tools explicitly provided. Never invent a tool. "
        "Use search_handbook for policy questions. Use calculate_leave_balance for math on leave days. "
        "If neither tool has the answer and you don't know from general knowledge, say so honestly — never guess."
    ),
)

def ask(question):
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"Q: {question}\nA: {result['messages'][-1].content}\n")
    except Exception as e:
        print(f"Q: {question}\nFAILED: {e}\n")

print("Running Q1...", flush=True)
ask("If I've taken 4 vacation days, how many do I have left?")

print("Running Q2...", flush=True)
ask("What's the remote work policy?")

print("Running Q3...", flush=True)
ask("What's the company's parental leave policy?")
