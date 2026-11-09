"""Names for a new conversation and a new project, proposed by the draft model from the first query."""

import json
import re
from typing import NamedTuple

from config import get_logger
from inference.model_port import NAMING, TextModel
from memory.conversation_registry import clean_name
from prompt_registry import load_section

logger = get_logger(__name__)

# No maxLength: as a grammar constraint it made the draft model produce
# fragments ("...WriteAHEAD..."); clean_name bounds the length instead.
NAME_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}

_FILLER = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "describe", "difference",
    "do", "does", "explain", "for", "give", "help", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "please", "should", "tell", "the", "their", "them", "they", "this",
    "to", "use", "what", "when", "where", "which", "who", "why", "with", "would", "you",
    "your", "between", "about", "work", "works",
}
_GENERIC = {"question", "help", "query", "conversation", "chat", "new conversation", "untitled"}
_WORD = re.compile(r"[A-Za-z0-9][\w+#.\-]*")


class NamingPrompts(NamedTuple):
    conversation: str
    project: str


def fallback_name(query: str, words: int = 5) -> str:
    """A title from the query's own content words, used when the model gives nothing usable."""
    picked = []
    for word in _WORD.findall(query or ""):
        word = word.rstrip(".-")
        if word.lower() in _FILLER or not word:
            continue
        picked.append(word if any(c.isupper() for c in word[1:]) or word.isupper() else word.capitalize())
        if len(picked) == words:
            break
    return clean_name(" ".join(picked)) or "Untitled Conversation"


class Namer:
    def __init__(self, model: TextModel, prompts: NamingPrompts | None = None) -> None:
        self.model = model
        self.prompts = NamingPrompts(**load_section("naming", NamingPrompts._fields)) if prompts is None else prompts

    def conversation_name(self, first_query: str) -> str:
        return self.__name(self.prompts.conversation, first_query)

    def project_name(self, first_query: str) -> str:
        return self.__name(self.prompts.project, first_query)

    def __name(self, system: str, query: str) -> str:
        try:
            reply = self.model.infer(system, query[:4000], schema=NAME_SCHEMA, max_tokens=32,
                                     purpose=NAMING)
            name = clean_name(json.loads(reply).get("name"))
        except Exception as error:
            # Naming is never worth failing a query over.
            logger.warning("Naming failed (%s); using a title from the query", error)
            return fallback_name(query)
        if not name or name.lower() in _GENERIC:
            return fallback_name(query)
        return name
