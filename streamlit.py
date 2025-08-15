import streamlit as st
from crewai import Task, Agent, Crew, LLM, Process
import os


api_key = "AIza********************OjMk"

os.environ["GEMINI_API_KEY"] = api_key

llm = LLM(model="gemini/gemini-2.5-flash", temperature=0)

st.title("Generating Articles and Producing Concise Summaries")

author_agent = Agent(
    role="searcher",
    goal="trying write everything single topic about chess",
    backstory="you are a geek who know the rare ,strang and interesting things",
    llm=llm,
    verbose=True)

task_1 = Task(
    description="Generate an article include short introduction ,events ,characters and styles ",
    expected_output="minimum fifty lines ",
    agent=author_agent,
    verbose=True
)

summerize_agent = Agent(
    role="profitional writer",
    goal="Summerize the article",
    backstory="you are a profitional writer who can catch the main point",
    llm=llm,
    verbose=True)

task_2 = Task(
    description="""
        Summerize the articles or rewrite it in different way so that 
        you can show the main point ,the important information and can
        ignore what think that everyone knows or not interesting and if 
        you mention at most just two examples 
        """,
    expected_output="maximum tewnty lines ",
    agent=summerize_agent,
    context=[task_1],
    verbose=True)

fact_checker_agent = Agent(
    role="detector writer",
    goal="Ensure that the content of article is matching with the titles",
    backstory="you have great skill with logical thinking and data analysis",
    llm=llm,
    verbose=True)

task_3 = Task(
    description="""
        Extract some lines of the article and ensure that it is 
        matching with the title by first tell how much it matching 
        the title (precentage) and adding some detail which not mentioned 
        in the article or giving your own evidence
            """,
    expected_output="maximum ten lines",
    agent=fact_checker_agent,
    context=[task_1],
    verbose=True)

metadata_agent = Agent(
    role="conculsion writer",
    goal="Extract the main points and associated tags",
    backstory="you have great skill with fast reading who can ignore the details and extract main points",
    llm=llm,
    verbose=True)

task_4 = Task(
    description="Just extract the main point of each paragraph and Tags (concepts that corrlated with the titles or high associated with it)",
    expected_output="Just the main points and list of Tags which associated article , just one line for each part of the article maximum ten lines anyway",
    agent=metadata_agent,
    context=[task_1],
    verbose=True)

article_crew = Crew(
    agents=[author_agent, summerize_agent, fact_checker_agent, metadata_agent],
    tasks=[task_1, task_2, task_3, task_4],
    process=Process.sequential)

result = article_crew.kickoff()

for task in result.tasks_output:

    for line in task.raw.split("\n"):

        st.markdown(line.strip())
    
    st.markdown("*"*26 + " Task End " + "*"*26)


