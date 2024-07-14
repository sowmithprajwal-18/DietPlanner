import streamlit as st
import pandas as pd
import mysql.connector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, load_agent, load_tools,AgentType, Tool
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import mysql.connector
from langchain.schema import HumanMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import time,random
import os

apiKey = "AIzaSyBuNH_6DxghWHDC1FEwf5AwylucoSH4F78"
llm = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key=apiKey,temperature=0,convert_system_message_to_human=True)
memory = ConversationBufferMemory(return_messages=True)
os.environ['SERPAPI_API_KEY'] = '7cba696fb1f563d5f7c805bfff211e138ccc42eadf0db1e7aa62559c56e45707'

def generateOutput(input, diet_chart):
    agent_executor = create_pandas_dataframe_agent(
        llm=llm,
        df=diet_chart,
        extra_tools=load_tools(['llm-math','serpapi'], llm=llm),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_code=True,
        max_execution_time=15
    )
    output = agent_executor.invoke(input)['output']
    template = "System Prompt = You are professional diet planner. If the question is not related to diet,nutrition then give the response saying you dont know. Given this question : {input}. Given the Answer to this question : {output}. If the response generated is valid to question rephrase the response and send it. If the generated response is invalid then return a suitable response to the question. Just give me the valid respose if the response is wrong or if the reponse is right then rephrase it and Nothing else "
    prompt = HumanMessagePromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_messages([prompt]).format_prompt(input=input,output=output).to_messages()
    reply = llm(prompt)
    reply = reply.content
    return reply

def response_generator():
    df = generateOutput(prompt,getDataFrame())
    print(df)
    type(df)
    response = random.choice(
        [
            df
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

print("Starting script")
def getDataFrame():
    mydb = mysql.connector.connect(
    host="newdbarogyaa.mysql.database.azure.com",
    user="arogyaUserr",
    password="HealthIsWealth@123#",
    database='arogya_data'
    )
    
    mycursor = mydb.cursor()

    sql = f"SELECT meal_time,item,calories FROM arogya_data.diet_chart a inner join dummy_table b on a.email = b.email;"
    mycursor.execute(sql)
    results = mycursor.fetchall()
    print(results)
    df = pd.DataFrame(results)
    df.rename(columns={0: 'MealTime', 1: 'item',2:'calories'}, inplace=True)
    print("dg",df)
    return df


st.set_page_config(layout="wide")

col1, col2 = st.columns(2)


with col1:
    st.header("Chat with your Diet Planner")
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        #result = generateOuput()
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())

with col2:
    st.header("Diet Plan")
    st.dataframe(getDataFrame())