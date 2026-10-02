from config import get_logger
from memory.topic_pool.topic_manager import TopicManager
from memory.topic_pool.project_pool.project_manager import ProjectManager
from memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager import ConversationPoolManager
from knowledge_acquisition.ksv_manager import KSVManager

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self):
        pass

    # this is the function that is going to get me all the chunks by talking to the data layer 
    # then post the chunks into the conversation 
    def get_chunks(self , topic , query):
        pass
