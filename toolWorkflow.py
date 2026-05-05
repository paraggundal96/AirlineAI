import select
from langchain_core.tools import tool
from model import select_model
from langchain_core.messages import HumanMessage, ToolMessage

ticket_prices = {
    "london": "$799",
    "paris": "$899",
    "tokyo": "$1400",
    "berlin": "$499"
}

llm = select_model()

@tool
def get_ticket_price(destination_city:str) -> str:
    ''' Get the price of a return ticket to destination city '''
    price = ticket_prices.get(destination_city.lower(), "Unknown Destination City")
    return f"The price of a ticket to {destination_city} is {price}"

llm_with_tool = llm.bind_tools([get_ticket_price])

def chat(user_input):

    messages = [HumanMessage(content = user_input)]
    response = llm_with_tool.invoke(messages)

    if response.tool_calls:

        tool_call = response.tool_calls[0]
        name = tool_call["name"]
        args = tool_call["args"]

        if name == "get_ticket_price":
            result = get_ticket_price.invoke(args)
        
        tool_message = ToolMessage(
            content = result,
            tool_call_id = tool_call["id"]
        )

        messages.append(response)
        messages.append(tool_message)

        final_response = llm_with_tool.invoke(messages)
        return final_response.content

    return response.content


print(chat("What is Ticket to LONDON"))
