import hashlib
from typing import List, Sequence, Tuple

from data_layer.ingestion.metadata.metadata import ChunkMetaData
from data_layer.ingestion.nodes.nodes import Document, NormalizedContent, RChunk

from .windowing import sliding_windows

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class RecursiveChunker:
    def __init__(
        self,
        normalized_documents_contents: List[NormalizedContent],
        chunk_size,
        overlap=20,
        separator=None,
        db_path: str | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= overlap < chunk_size:
            raise ValueError(
                f"overlap must be in [0, {chunk_size}), got {overlap}"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.normalized_documents_contents = normalized_documents_contents
        self.db_path = db_path
        self.default_separators = (
            separator if separator is not None else list(DEFAULT_SEPARATORS)
        )

    def __make_chunk_meta_data(self, document_name, document_id):
        return ChunkMetaData(document_name, document_id, "recursive")

    def __make_chunk_id(self, *args):
        value = "".join(str(arg) for arg in args)
        hash_object = hashlib.sha256(value.encode("utf-8"))
        hex_digest = hash_object.hexdigest()
        return str(hex_digest)

    def __make_chunk_obj(self, chunk, document_name, document_id, ordinal, start, end):
        return RChunk(
            chunk,
            self.__make_chunk_meta_data(document_name, document_id),
            # The ordinal is what keeps the id unique: a document that repeats a
            # boilerplate paragraph would otherwise produce two chunks with the
            # same id, and the embedder derives its vector id from the same text.
            self.__make_chunk_id(document_id, ordinal, chunk),
            start,
            end,
        )

    def __pick_separator(
        self, text: str, separators: Sequence[str]
    ) -> Tuple[str | None, Sequence[str]]:
        for index, separator in enumerate(separators):
            if separator and separator in text:
                return separator, separators[index + 1 :]
        return None, []

    def __separator_pieces(self, text: str, separator: str) -> List[Tuple[int, int]]:
        """Offsets of text split on separator, each piece keeping its own separator.

        The pieces tile the text exactly. The previous implementation rebuilt
        pieces as `part + separator`, which appended a separator the document
        never had to whichever piece came last.
        """
        pieces: List[Tuple[int, int]] = []
        start = 0
        while True:
            found = text.find(separator, start)
            if found == -1:
                break
            end = found + len(separator)
            pieces.append((start, end))
            start = end
        if start < len(text):
            pieces.append((start, len(text)))
        return pieces

    def __split(
        self, text: str, base: int, separators: Sequence[str]
    ) -> List[Tuple[int, int]]:
        if len(text) <= self.chunk_size:
            return [(base, base + len(text))] if text.strip() else []

        separator, remaining = self.__pick_separator(text, separators)
        if separator is None:
            return [
                (base + start, base + end)
                for start, end in sliding_windows(text, self.chunk_size, 0)
            ]

        spans: List[Tuple[int, int]] = []
        open_start: int | None = None
        open_end = 0

        def flush() -> None:
            if open_start is not None and text[open_start:open_end].strip():
                spans.append((base + open_start, base + open_end))

        for piece_start, piece_end in self.__separator_pieces(text, separator):
            if open_start is not None and piece_end - open_start <= self.chunk_size:
                open_end = piece_end
                continue

            flush()
            open_start, open_end = piece_start, piece_end
            if open_end - open_start > self.chunk_size:
                spans.extend(
                    self.__split(text[open_start:open_end], base + open_start, remaining)
                )
                open_start = None

        flush()
        return spans

    def __apply_overlap(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Extend each chunk backwards into the one before it.

        Applied once over the finished spans rather than inside __split, where
        it used to run again at every level of the recursion and duplicate the
        same text into a chunk several times over.
        """
        if self.overlap <= 0 or len(spans) < 2:
            return spans

        overlapped = [spans[0]]
        for previous, (start, end) in zip(spans, spans[1:]):
            room = self.chunk_size - (end - start)
            back = min(self.overlap, room, start - previous[0])
            overlapped.append((start - back, end) if back > 0 else (start, end))
        return overlapped

    def recursive_chunker(self) -> List[RChunk]:
        chunk_values: List[RChunk] = []
        for normalized_document_content in self.normalized_documents_contents:
            content = normalized_document_content.content
            document_name = normalized_document_content.meta_data.file_name
            document_id = normalized_document_content.meta_data.document_id

            spans = self.__apply_overlap(
                self.__split(content, 0, self.default_separators)
            )
            for ordinal, (start, end) in enumerate(spans):
                chunk_values.append(
                    self.__make_chunk_obj(
                        content[start:end], document_name, document_id, ordinal, start, end
                    )
                )

        self.__persist(chunk_values)
        return chunk_values

    def __persist(self, chunks: List[RChunk]) -> None:
        if not self.db_path or not chunks:
            return
        from .DB_Manager import Manager

        documents = [
            Document(
                content.meta_data.document_id,
                content.meta_data.file_name,
                content.content,
            )
            for content in self.normalized_documents_contents
        ]
        manager = Manager(self.db_path, is_chunker_type_hierarchical=False)
        try:
            manager.insert_documents(documents)
            manager.insert_recursive_chunks(chunks)
        finally:
            manager.close()
