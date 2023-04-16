"""this module responsible for check backend systems or for web scraping"""

import inspect
from typing import List

import numpy as np
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.conversation.memory import ConversationBufferMemory

from tools.google_search import Search
from tools.todo_list import ToDo

# load the usable class names
modules = {
    'ToDo' : 'active',
    'Search': 'active'

}
# load the enviroment variables
load_dotenv()

# initialize the agent
class ChatAgentExecutor :
    """this class responsible to act as backend query system"""

    llm_model = OpenAI(temperature=0)                               # define the openai llm
    memory = ConversationBufferMemory(memory_key="chat_history")    # define chathistory buffer memeory               

    def __init__(self) -> None:
        pass

    def init_tools(self) -> List[Tool]:
        """this method responsible to initialize serveral tools google api/balance check"""

        self.plugins = {}

        # Load Basic Foundation Models
        for class_name, status in modules.items():
            if(status=='active'):
                self.plugins[class_name] = globals()[class_name]()

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.plugins.values()])
                if template_required_names.issubset(loaded_names):
                    self.plugins[class_name] = globals()[class_name](
                        **{name: self.plugins[name] for name in template_required_names})
        
        # load the agent plugins
        self.tools = []
        for instance in self.plugins.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, 
                                           description=func.description, 
                                           func=func))

        
        return self.tools
    
    def init_prompt(self):
        """this method responsible to initialize the agent prompt"""

        prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}. 
                    You have access to the following tools:"""
        
        suffix = """Question: {task}
                    {agent_scratchpad}"""  # we currently require an agent_scratchpad input variable to put notes on previous actions and observations
        return prefix , suffix
    
    def init_agent(self) -> AgentExecutor:
        """this method responsible to initialize the chat agent for backend system query"""

        # step 1. get tools
        tools = self.init_tools()

        # step 2. init prompts
        prefix, suffix = self.init_prompt()

        # create a prompt template matching with the zeroshot agent
        prompt = ZeroShotAgent.create_prompt(
            tools= tools,
            prefix= prefix,
            suffix= suffix,
            input_variables=["objective", "task", "context","agent_scratchpad"],
        )

        # define the multi lingual llmchain
        llm = LLMChain(llm= self.llm_model,prompt= prompt )

        # define the zeroshot agent
        agi_agent = ZeroShotAgent(
            llm_chain=llm,
            tools=tools,
            #agent="conversational-react-description",
            #memory=self.memory,
        )

        # define the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent= agi_agent,
            tools=tools, 
            verbose=True
        )

        return agent_executor