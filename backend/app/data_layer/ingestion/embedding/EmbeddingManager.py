import hashlib
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    InvalidEmbeddingArgument,
)
from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
from data_layer.ingestion.nodes.nodes import EmbeddedChunk, HChunk, RChunk


# Re-exported so callers of this module can reason about the id range without
# reaching for Config. See Config.VECTOR_ID_MASK for the rationale.
VECTOR_ID_MASK = Config.VECTOR_ID_MASK


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        embeddding_dimension: int = Config.EMBEDDING_DIMENSIONS,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = embeddding_dimension

    def __generate_vector_id(self, chunk: str) -> int:
        hash_bytes = hashlib.md5(chunk.encode("utf-8")).digest()
        uint64_id = int.from_bytes(hash_bytes[:8], byteorder="little", signed=False)
        # Clearing the top bit keeps the id non-negative AND inside the signed
        # 64-bit range both storage backends accept. Without this, ~49% of ids
        # raise "Python int too large to convert to SQLite INTEGER" on insert.
        return uint64_id & VECTOR_ID_MASK

    def __create_meta_data(self, chunk_id: str, chunk: str) -> EmbeddedChunkMetaData:
        return EmbeddedChunkMetaData(chunk_id, chunk, self.model_name)

    def __embed_chunk(self, chunkObj: HChunk | RChunk) -> EmbeddedChunk:
        chunk = chunkObj.chunk
        chunk_id = chunkObj.chunk_id
        embedded_chunk = self.model.encode(chunk, truncate_dim=self.embedding_dimension)
        return EmbeddedChunk(
            embedded_chunk.astype(np.float32),
            self.__generate_vector_id(chunk),
            self.__create_meta_data(chunk_id, chunk),
        )

    def __embed_chunks(self, chunks: List[HChunk | RChunk]) -> List[EmbeddedChunk]:
        embedded_chunks: List[EmbeddedChunk] = []
        texts = []
        for chunkObj in chunks:
            if not isinstance(chunkObj, HChunk | RChunk):
                raise InvalidEmbeddingArgument(
                    f"The argument passed is of type {type(chunkObj).__name__}. The valid types are Chunk and list"
                )
            texts.append(chunkObj.chunk)

        vectors = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_vectors = self.model.encode(
                texts[i : i + batch_size], truncate_dim=self.embedding_dimension
            )
            vectors.extend(batch_vectors)
        for chunkObj, vector in zip(chunks, vectors):
            chunk = chunkObj.chunk
            chunk_id = chunkObj.chunk_id
            embedded_chunks.append(
                EmbeddedChunk(
                    vector.astype(np.float32),
                    self.__generate_vector_id(chunk),
                    self.__create_meta_data(chunk_id, chunk),
                )
            )
        return embedded_chunks

    def __embed_text(self, text: str, chunk_id: str | None) -> EmbeddedChunk:
        if not isinstance(text, str) or not text.strip():
            raise InvalidEmbeddingArgument(
                f"embed_text expects a non-empty str, got {type(text).__name__}"
            )
        if chunk_id is None:
            chunk_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vector = self.model.encode(text, truncate_dim=self.embedding_dimension)
        return EmbeddedChunk(
            vector.astype(np.float32),
            self.__generate_vector_id(text),
            self.__create_meta_data(chunk_id, text),
        )

    def embed_text(self, text: str, chunk_id: str | None = None) -> EmbeddedChunk:
        """Embed a raw string that is not a pipeline chunk.

        The memory layer needs this for conversation summaries and turns, which
        never pass through the Chunker and so are not HChunk/RChunk. When
        chunk_id is omitted it is derived from the content.
        """
        return self.__embed_text(text, chunk_id)

    def embed_texts(
        self, texts: List[str], chunk_ids: List[str] | None = None
    ) -> List[EmbeddedChunk]:
        """Batch form of embed_text. chunk_ids, if given, must align with texts."""
        if chunk_ids is not None and len(chunk_ids) != len(texts):
            raise InvalidEmbeddingArgument(
                f"Got {len(chunk_ids)} chunk_ids for {len(texts)} texts. They must align."
            )
        if chunk_ids is None:
            chunk_ids = [None] * len(texts)
        return [
            self.__embed_text(text, chunk_id)
            for text, chunk_id in zip(texts, chunk_ids)
        ]

    def embed(
        self, arg: HChunk | RChunk | List[HChunk | RChunk]
    ) -> EmbeddedChunk | List[EmbeddedChunk]:
        if isinstance(arg, HChunk | RChunk):
            return self.__embed_chunk(arg)
        if isinstance(arg, list):
            return self.__embed_chunks(arg)
        raise InvalidEmbeddingArgument(
            f"The argument passed is of type {type(arg).__name__}. The valid types are Chunk and List[Chunk]"
        )
