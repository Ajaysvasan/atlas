from dataclasses import dataclass

from numpy import ndarray

from data_layer.ingestion.metadata.metadata import (
    ChunkMetaData,
    EmbeddedChunkMetaData,
    NormalizedTextMetaData,
)


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


@dataclass(frozen=True)
class NormalizedContent:
    content: str
    has_section: bool
    meta_data: NormalizedTextMetaData


@dataclass(frozen=True)
class EmbeddedChunk:
    vector: ndarray
    vector_id: int
    meta_data: EmbeddedChunkMetaData
