"""this module for user based google search query fetching"""

from langchain.utilities import GoogleSerperAPIWrapper

from utils import prompts


class Search:
    def __init__(self):
        print(f"Initializing Google Search Engine")
        self.search_api = GoogleSerperAPIWrapper()                           # initialize the google search wrapper     

    # define the prompt helpers
    @prompts(name="Search/Browse Google From User Input Text",
             description="useful when you want to get information about general knowledge and current events of the universe."
                         "like:  provide details related to this topic ? or provide search results for this query."
                         "Input: should be a text input, user query requesting information about current eventsl.")
    
    def inference(self, user_query):
        # return the google featched data
        return self.search_api.run(user_query)