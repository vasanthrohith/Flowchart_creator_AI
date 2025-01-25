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


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]



def create_summary_on_usecase(state):
    print("--create_summary_on_usecase--")
    state_dict = state["keys"]
    usecase_summary = state_dict["usecase_summary"]

    create_summary_on_usecase_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an professional project plan designer. with the given user project summary/description give a detailed plan in more extended way.
        \nexample Project Summary from the user:
             im planning to create an app to summarize youtube video


        \nexample detailed planfor the above user summary:
            create a workflow to for AI video summarizer project
            descripotion:
            - First need to get a query from user on video topic
            - then pass it to an LLM(Large Language Model) to generate a topic of the video from user query
            - then pass it to youtube video downloader.
            - then pass the downloaded video to transcribe text from video
            - then pass it to an LLM to get the summary of the video.
         
         note: do not mention any other detsils like you created the plan and presenting to user. just give the plan in more detailed way.
                
        """),
        ("user", "{usecase_summary}")
    ])

    create_summary_on_usecase_chain = create_summary_on_usecase_prompt | llm
    project_summary = create_summary_on_usecase_chain.invoke({"usecase_summary": usecase_summary})

    return {"keys": {"project_summary": project_summary.content}}
    


def project_summary_to_markdown_summary(state):
    print("--project_summary_to_markdown_summary--")
    state_dict = state["keys"]
    project_summary = state_dict["project_summary"]

    projectsummary_to_mkdnsummary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an professional project plan designer. with the given user project summary give a plan to design a markdown flow chart.
        \nexample Project Summary from the user:
            create a workflow to for AI video summarizer project
            descripotion:
            - First need to get a query from user on video topic
            - then pass it to an LLM(Large Language Model) to generate a topic of the video from user query
            - then pass it to youtube video downloader.
            - then pass the downloaded video to transcribe text from video
            - then pass it to an LLM to get the summary of the video.
        
        \nmarkdown summary output for the above example project summary:
            Start:
            The workflow begins at node A, labeled "Start."
            The process is initiated when a user provides a query about the video topic.
        
            Step 1: Query Processing:
            At node B, the user's query is passed to a Large Language Model (LLM) to interpret and refine the topic of the video based on the query.
            
            Step 2: Video Download:
            From node B, the refined topic is passed to node C, where a YouTube video downloader fetches the relevant video.
            
            Step 3: Transcription:
            The downloaded video is passed to node D, where the content of the video is transcribed into text.
            
            Step 4: Video Summarization:
            The transcribed text is passed to node E, where an LLM generates a concise and meaningful summary of the video.
            End:

            At node F, the summarized content is provided to the user, marking the end of the workflow.
            
            Flow:
            Start (A): User provides a query.
            Query Processing (B): Query is processed by the LLM.
            Video Download (C): Video is downloaded based on the refined topic.
            Transcription (D): Video is transcribed into text.
            Summarization (E): LLM generates a video summary.
            End (F): Summary is delivered to the user.
         
         note: Keep the summary short and precise, do not create more than 4 end to end nodes. Please Do not overcook the flowchart.
                Create condition conditions if needed don't be always linear.

            
        """),
        ("user", "{project_summary}")
    ])

    projectsummary_to_mkdnsummary_chain = projectsummary_to_mkdnsummary_prompt | llm

    markdown_summary = projectsummary_to_mkdnsummary_chain.invoke({"project_summary": project_summary})

    print(markdown_summary)
    return {"keys": {"markdown_summary": markdown_summary.content}}



def markdown_summary_to_markdown(state):
    print("--markdown_summary_to_markdown--")
    state_dict = state["keys"]
    markdown_summary = state_dict["markdown_summary"]

    mkdnsummary_to_mkdn_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an professional markdown workflow designer with the given description content below create a mardown workflow.
        \n\nexample description content:
            Start:
            The workflow begins at node A labeled "Start."
        
            Decision:
            The process moves to node B, which represents a decision point labeled "Decision?".
            Here, a condition is evaluated to determine the next step.
        
            Yes Path:
            If the decision is "Yes", the workflow proceeds to node C, labeled "Process 1."
        
            No Path:
            If the decision is "No", the workflow proceeds to node D, labeled "Process 2."
        
            End:
            Both paths, whether "Yes" or "No", eventually lead to node E, labeled "End," where the workflow terminates.
        
        \n\nexample markdown output for the above description:
                graph TD
                    A[Start] --> B{{Decision?}}
                    B -->|Yes| C[Process 1]
                    B -->|No| D[Process 2]
                    C --> E[End]
                    D --> E[End]
        
        note: - Give only the markdown file without explicitly any additional information or text or characters
              - Keep the chart short and precise, do not create more than 4 end-end nodes. Please Do not overcook the flowchart.
              - Create condition conditions if needed don't be always linear.
            
        """),
        ("user", "{input}")
    ])

    markdown_gen_chain = mkdnsummary_to_mkdn_prompt | llm

    gen_markdown = markdown_gen_chain.invoke({"input": markdown_summary})
    
    return {"keys": {"gen_markdown": gen_markdown.content}}


def clean_markdown_response(state):
    print("--clean_markdown_response--")
    state_dict = state["keys"]
    gen_markdown = state_dict["gen_markdown"]

    mermaid_content = re.search(r"```markdown\s*(.*?)\s*```", gen_markdown, re.DOTALL)

    if mermaid_content:
        # Clean and format the content for markdown processing
        cleaned_content = f"""\n{mermaid_content.group(1).strip()}\n"""
        # return cleaned_content
        return {"keys": {"cleaned_markdown": cleaned_content}}
    else:
        # return "No Mermaid markdown content found." 
        return {"keys": {"cleaned_markdown": "No Mermaid markdown content found."}}
    
    # ! Post this node need to add decision node - if any error in parsing the markdown content it should be handles
    
    

def create_flowchart(state):
    print("--create_flowchart--")
    state_dict = state["keys"]
    cleaned_markdown = state_dict["cleaned_markdown"]
    html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Mermaid Flowchart</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'dark',  // Use dark theme as the base
                    themeVariables: {{
                        primaryColor: '#ffffff',      // Node border and line color
                        edgeLabelBackground: '#000', // Edge label background color
                        nodeTextColor: '#ffffff',    // Text inside nodes
                        fontFamily: 'Arial',         // Font customization
                    }}
                }});
            </script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    line-height: 1.5;
                    background-color: #1e1e1e; /* Dark background for contrast */
                }}
                .mermaid {{
                    margin: 50px auto;
                }}
            </style>
        </head>
        <body>
            <div class="mermaid">
                {cleaned_markdown.strip()}
            </div>
        </body>
        </html>
        """

    # Step 3: Render the HTML and save it as a PNG
    hti = Html2Image(output_path='output')
    hti.screenshot(html_str=html_template, save_as="flowchart.png")

    print("Flowchart with white lines saved as 'output/flowchart.png'")
    return {"keys": {"flowchart": "Flowchart is successfully created and saved as 'output/flowchart.png'"}}
    

workflow1 = StateGraph(GraphState)


# add nodes
workflow1.add_node("create_summary_on_usecase", create_summary_on_usecase)
workflow1.add_node("project_summary_to_markdown_summary", project_summary_to_markdown_summary)
workflow1.add_node("markdown_summary_to_markdown", markdown_summary_to_markdown)
workflow1.add_node("clean_markdown_response", clean_markdown_response)
workflow1.add_node("create_flowchart", create_flowchart)

# Building graph
workflow1.add_edge(START, "create_summary_on_usecase")
workflow1.add_edge("create_summary_on_usecase", "project_summary_to_markdown_summary")
workflow1.add_edge("project_summary_to_markdown_summary", "markdown_summary_to_markdown")
workflow1.add_edge("markdown_summary_to_markdown", "clean_markdown_response")
workflow1.add_edge("clean_markdown_response", "create_flowchart")
workflow1.add_edge("create_flowchart", END)


flowchart_maker_app = workflow1.compile()


# flowchart_maker_app.invoke({"keys": {"usecase_summary": "im planning to create a speech to text transcribe AI app with openai model"}})

    
