"""What an acquisition is trying to learn about."""

from typing import Callable, NamedTuple, Sequence, Tuple

from numpy import ndarray


class AcquisitionTarget(NamedTuple):
    """The subject, its specialisations, and the question being answered.

    The topic decides what may be collected at all; the query decides whether
    what was collected answers anything. They are separate because material can
    be squarely on-topic and still not answer the question — that is worth
    keeping — while material that answers a differently-worded question about
    another subject is not.
    """

    topic: str
    query: str
    topic_vector: ndarray
    query_vector: ndarray
    subtopics: Tuple[str, ...] = ()
    subtopic_vectors: Tuple[ndarray, ...] = ()

    @property
    def terms(self) -> str:
        """What to type into a source's own search box."""
        return " ".join(part for part in (self.topic, *self.subtopics, self.query)
                        if part and part.strip())


def build_target(
    topic: str,
    query: str,
    embed_one: Callable[[str], ndarray],
    subtopics: Sequence[str] = (),
) -> AcquisitionTarget:
    """Embed the parts of a target. The caller supplies the embedder."""
    if not topic or not topic.strip():
        raise ValueError("a target needs a topic; it is what bounds collection")
    if not query or not query.strip():
        raise ValueError("a target needs a query")
    subtopics = tuple(s for s in subtopics if s and s.strip())
    return AcquisitionTarget(
        topic=topic,
        query=query,
        topic_vector=embed_one(topic),
        query_vector=embed_one(query),
        subtopics=subtopics,
        subtopic_vectors=tuple(embed_one(s) for s in subtopics),
    )
