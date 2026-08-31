from dataclasses import dataclass
from typing import Tuple

from numpy import ndarray

from data_layer.ingestion.metadata.metadata import (
    ChunkMetaData,
    EmbeddedChunkMetaData,
    NormalizedTextMetaData,
)


@dataclass(frozen=True)
class SectionSpan:
    """A heading and the body that follows it, located in normalized content.

    The normalizer emits these because it is the only stage that still sees the
    document's line structure; every offset is absolute into
    NormalizedContent.content so a chunk can be traced back to its source.
    """

    name: str
    heading_start: int
    heading_end: int
    content_start: int
    content_end: int


@dataclass(frozen=True)
class Document:
    documentId: str
    documentName: str
    normalizedText: str


@dataclass(frozen=True)
class Section:
    sectionId: str
    documentId: str

    sectionName: str
    content: str
    contentLength: int

    startOffSet: int
    endOffSet: int


@dataclass(frozen=True)
class Context:
    contextId: str
    sectionId: str

    context: str
    contextLen: int

    startOffSet: int
    endOffSet: int


@dataclass(frozen=True)
class HChunk:
    chunk_id: str
    context_id: str

    chunk: str

    start_off_set: int
    end_off_set: int
    meta_data: ChunkMetaData


@dataclass(frozen=True)
class RChunk:
    chunk: str
    meta_data: ChunkMetaData
    chunk_id: str

    start_off_set: int = 0
    end_off_set: int = 0


@dataclass(frozen=True)
class NormalizedContent:
    content: str
    has_section: bool
    meta_data: NormalizedTextMetaData
    sections: Tuple[SectionSpan, ...] = ()


@dataclass(frozen=True)
class EmbeddedChunk:
    vector: ndarray
    vector_id: int
    meta_data: EmbeddedChunkMetaData
