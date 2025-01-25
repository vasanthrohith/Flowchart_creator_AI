import os
from dotenv import load_dotenv
from pprint import pprint
import re
from html2image import Html2Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_openai import ChatOpenAI
from typing import Annotated, Dict, TypedDict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END
from langgraph.graph import StateGraph,add_messages, START, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough


from my_agents.flowchart_creator import flowchart_maker_app


# Load environment variables from a .env file
load_dotenv(override=True)
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# # Initialize Groq LLM
# llm = ChatGroq(
#     model_name="llama3-70b-8192",
#     temperature=0.7
# )

def call_openai_40mini():
    try:
        llm_openai = ChatOpenAI(
        model="gpt-4o-mini", temperature=0,
        )

        # print(self.llm_openai.invoke("Hi"))
        # print("Successfully connected to 'gpt-4o-mini'")

        return llm_openai

    except Exception as error:
        print("Error in connecting to openai : ", error)


llm = call_openai_40mini()


class MainAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    additional_args: Dict[str, any]

# Create tools
@tool
def create_project_plan(question)->str:
    """
    Create project plan with the user usecase or problem statement
    example question: "im planning to create a speech to text transcribe AI app with openai model"
    return: flowchart image path
    """
    planner_response = flowchart_maker_app.invoke({"keys": {"usecase_summary": question}})
    print("returning flowchart path >>>" , planner_response["keys"]["flowchart"])
    return planner_response["keys"]["flowchart"]


tools = [create_project_plan]
tool_node = ToolNode(tools)

model_with_tools = llm.bind_tools(tools)





def should_continue(state):
    print("state >>> ",state)
    messages = state["messages"]
    last_message = messages[-1]
    call_tool = last_message.tool_calls
    # user_=input("Enter to continue: ")
    # print(call_tool)
    if call_tool:
        return "call_tools"
    return "end"

def main_agent(state):
    print("state >>> ",state)
    messages = state["messages"]

    system_temp_default = """You are expert creating flowchart for the given user project one line summary or usecase"""
    
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_temp_default),
            ("human", "{query}"),
        ]
    )
    
    # response = model_with_tools.invoke(messages)

    chain = {"query": RunnablePassthrough()} | few_shot_prompt | model_with_tools
    response = chain.invoke(messages)
    # response = model_with_tools.invoke(messages)
    pprint(response.content)

    # print("response >>>", response)
    return {"messages": [response]}

def react_agent(state):
    print("state >>> ",state)
    messages = state["messages"]
    last_message = messages[-1]
    call_tool = last_message.tool_calls
    if call_tool:
        return "call_tools"
    return "end"


workflow_main = StateGraph(MainAgentState)

# Define the two nodes we will cycle between
workflow_main.add_node("main_agent", main_agent)
workflow_main.add_node("tool_node", tool_node)

# Build Graph
workflow_main.add_edge(START, "main_agent")
workflow_main.add_conditional_edges(
    "main_agent", 
    should_continue, 
        {
        "call_tools":"tool_node", 
        "end":END
        })

workflow_main.add_edge("tool_node", "main_agent")

main_app = workflow_main.compile()


# user_query = "Iam planning to create a speech to text transcribe AI app with openai model."
# user_query="Hi"


# response = main_app.invoke({"messages": [("human", user_query)],})
