from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedTextMetaData:
    document_id: str
    source_file_path: str | None
    file_name: str
    file_type: str
    ingestion_time: str
    normalizer_version: str
    content_hash: str
    source_path: str | None = None


@dataclass(frozen=True)
class ChunkMetaData:
    document_name: str
    document_id: str
    chunking_algorithm_used: str
    section_name: str | None = None


@dataclass(frozen=True)
class EmbeddedChunkMetaData:
    chunk_id: str
    chunk: str
    modelUsedForChunking: str
