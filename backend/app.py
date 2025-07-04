import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import Tool,AgentExecutor,initialize_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
#from langchain.agents.format_scratchpad import openai_functions
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentType
import openai
from flask import Flask, render_template, request, jsonify
"""from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import uvicorn
import datetime
from fastapi.middleware.cors import CORSMiddleware"""
app = Flask(__name__)

OPENAI_API_KEY = "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"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-3.5-turbo",
temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=300)
embs = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/response", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def setup_allergy_knowledge_base():
    new_db = FAISS.load_local("allergies_faiss_index", embs)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=new_db.as_retriever(search_type="mmr",k="3")
    )
    return knowledge_base

def setup_patients_knowledge_base():
    new_db = FAISS.load_local("patients_faiss_index", embs)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=new_db.as_retriever(search_type="mmr",k="3")
    )
    return knowledge_base


allergy_knowledge_base = setup_allergy_knowledge_base()
patients_knowledge_base = setup_patients_knowledge_base()

allergy_tool = [
    Tool(
      name="AllergySearch",
      func=allergy_knowledge_base.run,
      description="Useful for when you need to answer questions about allergies of patients"
       )]

patient_tool = [Tool(
      name="PatientDetailsSearch",
      func=patients_knowledge_base.run,
      description="Useful for when you need to answer questions about personal information of patients which includes address, phone number, gender, birthdate, etc."
    )
]

tools = allergy_tool + patient_tool
MEMORY_KEY = "chat_history"
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
tool_names = ["allergy_tool - Used to find the allergy of the patient", "patient_tool - Used to find the personal information of the patient which includes address, phone number,gender, birthdate, etc."]
chat_history = []
sys_prompt = f""" You are a helpful assistant who outputs the tool and the question it solves from the aksed question. To use for the given specific question, You can use the following tools: {tool_names}, output only the tool names along with the question they answer, dont answer the question directly. 
If asked about the patient summary give all the tools and the questions they can answer.
Output:
tool_name1: sub_question1
tool_name2: sub_question2
"""
prompt = ChatPromptTemplate.from_messages(
[
    (
            "system",
            sys_prompt,
    ),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
) 
agent = (
        {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],

        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
        )
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_Chat_response(user_input):
    #user_message = "What allergies does Saint Etienne Ottawa have and where does he live?"
    response = agent_executor.invoke({"input":user_input, "chat_history":[],"tool_names":tool_names})
    tools_to_be_used = response["output"].split("\n")
    responses = []
    for tool in tools_to_be_used:
        tool_name = tool.split(":")[0]
        tool_que = tool.split(":")[1]
        if(tool_name == "allergy_tool"):
            responses.append(allergy_knowledge_base.invoke(tool_que))
        if(tool_name== "patient_tool"):
            responses.append(patients_knowledge_base.invoke(tool_que))
        completion_prompt = """ You are a helpful assistant who will be given query and results from the tool results. 
                        You will output one meaningful final response, which answers the whole questions asked by combining the results from the tool results present in {responses}
                    """

        openai.api_key = "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"

        prompt = completion_prompt.format(responses=responses)
        final_response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=300,
                temperature=0
                )
       
        return final_response.choices[0].text.strip()







if __name__ == '__main__':
    app.run(debug=True)
