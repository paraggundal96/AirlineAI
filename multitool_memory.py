from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage


# ---------Create A Model------------ #

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


# -----------Create Tools------------ #

@tool

def add(a,b):
    ''' Adds two numbers '''
    return int(a) + int(b)

@tool

def multiply(a,b):
    ''' Mulitplies two numbers '''
    return int(a) * int(b)

@tool

def substract(a,b):
    ''' Substracts two numbers '''
    return int(a) - int(b)

tools = [add, multiply, substract]

llm_with_tool = model.bind_tools(tools)

select_tool = {tool.name:tool for tool in tools}

# ------------Creating Memory -----------#

store = {}

def get_session_history(session_id:str):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ----------Multi Tool Calling Chat Bot W Memory-------#

def chat(user_input:str, session_id = "user1"):
    # Get the user history
    history = get_session_history(session_id)
    # Append the current user prompt
    history.add_user_message(user_input)
    # Invoke with whole history
    response = llm_with_tool.invoke(history.messages) # ChatMessageHistory is a wrapper, history.messsage -> List[BaseMessages]

    # Check if tool call is requested from LLM
    if response.tool_calls:

        tool_messages = [] # If Multiple call in same query

        for tool_call in response.tool_calls:
            name = tool_call['name']
            args = tool_call['args']
            id = tool_call['id']

            if name in select_tool:
                result = select_tool[name].invoke(args)
            
            tool_messages.append(
                ToolMessage(
                    content = result,
                    tool_call_id = id
                )
            )
        
        # -----Append AI message(Tool Request) and Tool Messages to history-----------
        history.add_message(response)
        for msg in tool_messages:
            history.add_message(msg)
        
        #-------Send the Tool result to LLM and invoke with history-------------------
        final_response = llm_with_tool.invoke(history.messages)

        #--------Add the response from LLM to history

        history.add_message(final_response)
        return final_response.content
    
    history.add_message(response)

    return response.content


for i in range(3):

    print(chat(input('Enter Your Query')))