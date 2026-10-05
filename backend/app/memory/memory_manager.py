from memory_mapping_handler import MemoryMappingHandler

from config import get_logger
from knowledge_sufficiency.ksv_manager import KSVManager
from memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager import (
    ConversationPoolManager,
)
from memory.topic_pool.project_pool.project_manager import ProjectManager
from memory.topic_pool.topic_manager import TopicManager

logger = get_logger(__name__)


class MemoryManager:
    __conversation_manager: ConversationPoolManager
    __topic_manager: TopicManager
    __project_manager: ProjectManager
    __memory_mapping_hanlder: MemoryMappingHandler
    __ksv_manager: KSVManager

    def __init__(self, query: str, conversation_id: str, user_id: str, topic: str):
        pass
