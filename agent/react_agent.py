from langchain.agents import create_agent
from rich.diagnose import report

from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from tools.agent_tools import rag_summarize,get_weather,get_city
from tools.middleware import monitor_tools,log_before_model


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize,get_weather,get_city],
            middleware=[monitor_tools,log_before_model]
        )


    def execute_stream(self,query:str):
       input_dict={
           "messages": [
               {"role": "user", "content": query},
           ]
       }

       for chunk in self.agent.stream(input_dict, stream_mode="values"):
           lastest_message = chunk["messages"][-1]
           if lastest_message.content:
               yield lastest_message.content.strip() + "\n"








if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫拖一体机器人在我所在地区的气温下如何保养"):
        print(chunk,end="",flush=True)