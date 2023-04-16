from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent

from utils import prompts


class ToDo:
    def __init__(self):
        print(f"Initializing ToDo Chain")
        self.todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. \
                                           Come up with a todo list for this objective: {objective}")     
        # define llm chain for generating todo list
        self.todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt= self.todo_prompt)                   # initialize the google search wrapper     

    # define the prompt helpers
    @prompts(name="ToDo",
             description="useful for when you need to come up with todo lists."
                         "Input: an objective to create a todo list for."
                         "Output: a todo list for that objective. Please be very clear what the objective is!.")
    
    def inference(self, user_query):
        # return the google featched data
        return self.todo_chain.run(user_query)