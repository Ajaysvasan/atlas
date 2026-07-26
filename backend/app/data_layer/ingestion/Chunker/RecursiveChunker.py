import hashlib
from typing import List

from data_layer.ingestion.metadata.metadata import ChunkMetaData
from data_layer.ingestion.nodes.nodes import NormalizedContent, RChunk


class RecursiveChunker:
    def __init__(
        self,
        normalized_documents_contents: List[NormalizedContent],
        chunk_size,
        overlap=20,
        separator=None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.normalized_documents_contents = normalized_documents_contents
        self.default_separators = (
            separator if separator is not None else ["\n\n", "\n", ".", " ", ""]
        )

    def __make_chunk_meta_data(self, document_name, document_id):
        return ChunkMetaData(document_name, document_id, "recursive")

    def __make_chunk_id(self, *args):

        value = "".join(args)
        hash_object = hashlib.sha256(value.encode("utf-8"))
        hex_digest = hash_object.hexdigest()
        return str(hex_digest)

    def __make_chunk_obj(self, chunk, document_name, document_id):
        return RChunk(
            chunk,
            self.__make_chunk_meta_data(document_name, document_id),
            self.__make_chunk_id(chunk, document_name, document_id),
        )

    def __r_chunker(self, normalized_content, separators):
        if len(normalized_content) <= self.chunk_size:
            return [normalized_content] if normalized_content.strip() else []

        separator = separators[-1] if separators else ""
        remaining_separator = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in normalized_content:
                separator = sep
                remaining_separator = separators[i + 1 :]
                break
        if separator:
            splits = normalized_content.split(separator)
        else:
            splits = list(normalized_content)
        chunks = []
        current_chunk = ""

        for part in splits:
            piece = part + separator if separator else part
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(piece) > self.chunk_size:
                    chunks.extend(self.__r_chunker(piece, remaining_separator))
                    current_chunk = ""
                else:
                    current_chunk = piece
        if current_chunk:
            chunks.append(current_chunk)

        if self.overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.overlap :]
                space_left = self.chunk_size - len(chunks[i])
                if space_left > 0:
                    tail_to_use = prev_tail[-space_left:]
                    overlapped.append(tail_to_use + chunks[i])
                else:
                    overlapped.append(chunks[i])
            chunks = overlapped
        return chunks

    def recursive_chunker(self) -> List[RChunk]:
        chunk_values: List[RChunk] = []
        for normalized_document_content in self.normalized_documents_contents:
            normalizedContent = normalized_document_content.content
            documentName = normalized_document_content.meta_data.file_name
            documentId = normalized_document_content.meta_data.document_id
            chunks = self.__r_chunker(normalizedContent, self.default_separators)
            for chunk in chunks:
                chunk_values.append(
                    self.__make_chunk_obj(chunk, documentName, documentId)
                )
        return chunk_values
