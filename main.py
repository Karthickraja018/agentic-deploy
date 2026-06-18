import os
os.environ["CREWAI_DISABLE_IPYTHON"] = "true"

import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai.tools import tool
from fastapi import FastAPI, HTTPException

@tool("Generate Blog Post Image")
def generate_image(prompt: str) -> str:
    """Generates a featured image using Pollinations.ai (free, no API key required).
    Pass a descriptive text prompt; returns the image URL and markdown embed code."""
    encoded = requests.utils.quote(prompt[:500])
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1200&height=630&nologo=true"
    try:
        r = requests.head(url, timeout=15)
        if r.status_code == 200:
            return f"Image URL: {url}\nEmbed: ![Featured Image]({url})"
    except Exception:
        pass
    return f"Image URL (rendering may take a few seconds): {url}\nEmbed: ![Featured Image]({url})"

# Load environment variables from .env file
load_dotenv()

GEMINI_MODEL = "gemini/gemini-3.1-flash-lite"
# Initialize Gemini 2.5 Flash model using CrewAI's LLM class
gemini_llm = LLM(
    model=GEMINI_MODEL,
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)
@CrewBase
class AiTrendCrew:
    """AiTrendCrew system orchestration using YAML files"""
    
    # Path paths relative to where this class module sits
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['writing_task']
        )

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluator'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @task
    def evaluator_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluator_task']
        )

    @agent
    def refiner(self) -> Agent:
        return Agent(
            config=self.agents_config['refiner'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @task
    def refiner_task(self) -> Task:
        return Task(
            config=self.tasks_config['refiner_task']
        )

    @agent
    def image_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_generator'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm,
            tools=[generate_image]
        )

    @task
    def image_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_generation_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,       # Automatically collected from @agent decorators
            tasks=self.tasks,         # Automatically collected from @task decorators
            process=Process.sequential,  
            verbose=True
        )  
        
app = FastAPI(title="CrewAI Azure Service")


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "CrewAI Runner"}

@app.post("/run-crew")
def run_crew():
    try:
        result = AiTrendCrew().crew().kickoff()
        return {"status": "success", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))