from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory


## Include llm from previous lesson
from llm import llm

agent_prompt = hub.pull("hwchase17/react-chat")

tools = [
    Tool.from_function(
        name="General Chat"
        , description="For general chat not covered by other tools"
        , func=llm.invoke,
        return_direct=True
    )
]

agent = create_react_agent(llm, tools, agent_prompt)



memory = ConversationBufferMemory(
    memory_key = 'chat_history'
    , k=5
    , return_messages=True
    
)

agent_executor = AgentExecutor (
    agent=agent
    , tools=tools
    , memory = memory
    , verbose=True
    
)

def generate_response(prompt):
    """_summary_
    Create a handler that calls the Conversation agent
    and returns a response to be rendered in the UI
    Args:
        prompt (_type_): _description_
    """
    
    response = agent_executor.invoke({"input":prompt})
    
    return response['output']