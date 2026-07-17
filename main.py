from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
#prompt = ChatPromptTemplate.from_template("Explain {topic} like I'm switching careers into it.")

prompt = ChatPromptTemplate.from_templateprompt = ChatPromptTemplate.from_template(
    "You are a teacher. Explain the concept of {topic} clearly, "
    "using an analogy suited for someone switching careers into tech. "
    "Do not give career advice — only explain {topic} itself."
)
model = ChatGroq(model="llama-3.3-70b-versatile")
chain = prompt | model | StrOutputParser()

print(chain.invoke({"topic": "LangChain"}))