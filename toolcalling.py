from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model = "nvidia/nemotron-3-super-120b-a12b:free",
    base_url="https://openrouter.ai/api/v1",
    temperature = 0,
    api_key= key
)

from langchain_core.tools import tool

@tool

def multiply(a, b):

    """
    Multiplies two numbers

    args:
    a: first number
    b: second number
    """
    return a * b

@tool

def add(a,b):
    """
    Adds two numbers

    args:
    a: first number
    b: second number
    """
    return a + b

local_model = ChatOllama(
    model = "qwen3:4b",
    model_provider = "ollama",
    temperature= 0
)

tools = [add, multiply]

llm_tools = local_model.bind_tools(tools)

response = llm_tools.invoke("What is 10 multiplied to result by adding 2 and 4?")
print(response.tool_calls)