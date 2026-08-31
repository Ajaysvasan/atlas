import hashlib
import re
from typing import List

from data_layer.ingestion.metadata.metadata import ChunkMetaData
from data_layer.ingestion.nodes.nodes import (
    Context,
    Document,
    HChunk,
    NormalizedContent,
    Section,
    SectionSpan,
)

from .DB_Manager import Manager
from .windowing import sliding_windows

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class HierarchicalChunker:
    documentName: str
    documentId: str
    normalizedText: str
    chunkOverlap: int
    chunkSize: int

    def __init__(
        self,
        chunkOverlap: int,
        chunkSize: int,
        db_path: str,
        normalizedDocumentsContents: List[NormalizedContent],
    ) -> None:
        if chunkSize <= 0:
            raise ValueError(f"chunkSize must be positive, got {chunkSize}")
        if not 0 <= chunkOverlap < chunkSize:
            raise ValueError(
                f"chunkOverlap must be in [0, {chunkSize}), got {chunkOverlap}"
            )

        self.chunkOverlap = chunkOverlap
        self.chunkSize = chunkSize
        self.normalizedDocumentsContents = normalizedDocumentsContents

        self.db_path = db_path

    def __generate_id(self, *args):
        value = "".join(str(arg) for arg in args)
        hash_object = hashlib.sha256(value.encode("utf-8"))
        hex_digest = hash_object.hexdigest()
        return str(hex_digest)

    def __make_document_objs(self):
        docObjs: List[Document] = []
        for normalizedDocuments in self.normalizedDocumentsContents:
            normalizedContent = normalizedDocuments.content
            documentName = normalizedDocuments.meta_data.file_name
            documentId = normalizedDocuments.meta_data.document_id
            docObj = Document(documentId, documentName, normalizedContent)
            docObjs.append(docObj)
        return docObjs

    def __make_section(
        self, doc: Document, ordinal: int, name: str, start: int, end: int
    ) -> Section | None:
        raw = doc.normalizedText[start:end]
        content = raw.strip()
        if not content:
            return None
        # The ordinal is part of the id because section names repeat: two
        # "NOTES" headings in one document hashed to a single sectionId, and the
        # second insert died on the primary key.
        sectionId = self.__generate_id(doc.documentId, ordinal, name)
        offset = start + (len(raw) - len(raw.lstrip()))
        return Section(
            sectionId,
            doc.documentId,
            name,
            content,
            len(content),
            offset,
            offset + len(content),
        )

    def __find_sections(
        self, doc: Document, spans: tuple[SectionSpan, ...]
    ) -> List[Section]:
        """Turn the normalizer's section spans into Section rows.

        The spans are used as given rather than re-derived here: this stage sees
        text whose line structure the normalizer has already reshaped, so any
        heading regex run at this point disagrees with the one that decided the
        document was hierarchical in the first place.
        """
        sections: List[Section] = []

        if not spans:
            section = self.__make_section(
                doc, 0, "MAIN", 0, len(doc.normalizedText)
            )
            return [section] if section else []

        preamble = self.__make_section(
            doc, 0, "PREAMBLE", 0, spans[0].heading_start
        )
        if preamble:
            sections.append(preamble)

        for ordinal, span in enumerate(spans, start=1):
            section = self.__make_section(
                doc, ordinal, span.name, span.content_start, span.content_end
            )
            if section:
                sections.append(section)

        return sections

    def __create_chunk_metadata(self, document: Document, section_name: str | None):
        document_name = document.documentName
        document_id = document.documentId
        chunk_type = "hierarchical"
        return ChunkMetaData(document_name, document_id, chunk_type, section_name)

    def __find_contexts(self, sections: List[Section]) -> List[Context]:
        contexts: List[Context] = []
        for section in sections:
            content = section.content
            boundaries = [
                (match.start(), match.end())
                for match in PARAGRAPH_BREAK.finditer(content)
            ]
            boundaries.append((len(content), len(content)))

            start_idx = 0
            for ordinal, (end_idx, next_start) in enumerate(boundaries):
                raw = content[start_idx:end_idx]
                context_text = raw.strip()
                if context_text:
                    lead = len(raw) - len(raw.lstrip())
                    # Offsets are absolute into the document, not into the
                    # section, so a chunk can still be located in the source
                    # after the hierarchy is flattened for retrieval.
                    absolute = section.startOffSet + start_idx + lead
                    contexts.append(
                        Context(
                            contextId=self.__generate_id(
                                section.sectionId, ordinal, context_text
                            ),
                            sectionId=section.sectionId,
                            context=context_text,
                            contextLen=len(context_text),
                            startOffSet=absolute,
                            endOffSet=absolute + len(context_text),
                        )
                    )
                start_idx = next_start
        return contexts

    def __get_chunks(
        self, contexts: List[Context], document: Document, section_names: dict
    ) -> List[HChunk]:
        chunks: List[HChunk] = []

        for context in contexts:
            windows = sliding_windows(
                context.context, self.chunkSize, self.chunkOverlap
            )
            for ordinal, (start, end) in enumerate(windows):
                chunk = context.context[start:end]
                chunkId = self.__generate_id(context.contextId, ordinal, chunk)
                chunks.append(
                    HChunk(
                        chunkId,
                        context.contextId,
                        chunk,
                        context.startOffSet + start,
                        context.startOffSet + end,
                        self.__create_chunk_metadata(
                            document, section_names.get(context.sectionId)
                        ),
                    )
                )
        return chunks

    def __chunk_text(self, doc: Document, spans, h_manager):
        sections = self.__find_sections(doc, spans)
        h_manager.insert_sections(sections)
        contexts = self.__find_contexts(sections)
        h_manager.insert_contexts(contexts)
        section_names = {section.sectionId: section.sectionName for section in sections}
        chunks = self.__get_chunks(contexts, doc, section_names)
        h_manager.insert_chunks(chunks)
        return chunks

    def process_doc(self) -> List[HChunk]:
        if not self.normalizedDocumentsContents:
            return []

        docObjs = self.__make_document_objs()
        h_manager = Manager(self.db_path, is_chunker_type_hierarchical=True)
        chunks: List[HChunk] = []
        try:
            h_manager.insert_documents(docObjs)
            for docObj, normalized in zip(docObjs, self.normalizedDocumentsContents):
                chunks.extend(
                    self.__chunk_text(docObj, normalized.sections, h_manager)
                )
        finally:
            h_manager.close()
        return chunks
