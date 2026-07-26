import hashlib
import re
from typing import List

from data_layer.ingestion.metadata.metadata import ChunkMetaData
from data_layer.ingestion.nodes.nodes import Context, Document, HChunk, NormalizedContent, Section

from .DB_Manager import Manager


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
        self.chunkOverlap = chunkOverlap
        self.chunkSize = chunkSize
        self.normalizedDocumentsContents = normalizedDocumentsContents

        self.db_path = db_path

    def __generate_id(self, *args):
        value = "".join(args)
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

    def __find_sections(self, doc: Document):

        sections: List[Section] = []
        pattern = re.compile(r"^(?=.*[A-Z])[A-Z\s]+$", re.MULTILINE)
        matches = list(pattern.finditer(doc.normalizedText))
        
        if not matches:
            sectionName = "MAIN"
            sectionId = self.__generate_id(sectionName, doc.documentId)
            content = doc.normalizedText.strip()
            if content:
                sections.append(Section(sectionId, doc.documentId, sectionName, content, len(content), 0, len(doc.normalizedText)))
            return sections

        for i, match in enumerate(matches):
            sectionName = match.group().strip()
            sectionId = self.__generate_id(sectionName, doc.documentId)
            
            if i == 0 and match.start() > 0:
                preamble_content = doc.normalizedText[0:match.start()].strip()
                if preamble_content:
                    preamble_name = "PREAMBLE"
                    preamble_id = self.__generate_id(preamble_name, doc.documentId)
                    sections.append(Section(preamble_id, doc.documentId, preamble_name, preamble_content, len(preamble_content), 0, match.start()))

            headerStart = match.start()
            headerEnd = match.end()
            contentStart = headerEnd

            if i + 1 < len(matches):
                contentEnd = matches[i + 1].start()
            else:
                contentEnd = len(doc.normalizedText)

            content = doc.normalizedText[contentStart:contentEnd].strip()
            sectionObj = Section(
                sectionId,
                doc.documentId,
                sectionName,
                content,
                len(content),
                contentStart,
                contentEnd,
            )
            sections.append(sectionObj)
        return sections

    def __create_chunk_metadata(self, document: Document):
        document_name = document.documentName
        document_id = document.documentId
        chunk_type = "hierarchical"
        return ChunkMetaData(document_name, document_id, chunk_type)

    def __find_contexts(self, sections: List[Section]) -> List[Context]:
        contexts: List[Context] = []
        for section in sections:
            content = section.content
            pattern = re.compile(r"\n\s*\n")
            matches = list(pattern.finditer(content))

            start_idx = 0  # Start tracking from the beginning of the section

            for match in matches:
                # The paragraph ends where the double newline starts
                end_idx = match.start()
                context_text = content[start_idx:end_idx].strip()

                if context_text:  # Only add if it's not an empty string
                    contextId = self.__generate_id(context_text, section.sectionId)
                    contextObj = Context(
                        contextId=contextId,
                        sectionId=section.sectionId,
                        context=context_text,
                        contextLen=len(context_text),
                        startOffSet=start_idx,
                        endOffSet=end_idx,
                    )
                    contexts.append(contextObj)

                # Update the start index for the NEXT paragraph to be after the current double newline
                start_idx = match.end()

            # Capture the final paragraph after the last double newline
            if start_idx < len(content):
                context_text = content[start_idx:].strip()
                if context_text:
                    contextId = self.__generate_id(context_text, section.sectionId)
                    contextObj = Context(
                        contextId=contextId,
                        sectionId=section.sectionId,
                        context=context_text,
                        contextLen=len(context_text),
                        startOffSet=start_idx,
                        endOffSet=len(content),
                    )
                    contexts.append(contextObj)
        return contexts

    def __get_chunks(self, contexts: List[Context], document: Document) -> List[HChunk]:
        if self.chunkOverlap >= self.chunkSize:
            raise ValueError(
                "Invalid chunk overlap and chunk size. chunkSize must be greater than chunkOverlap"
            )

        chunks: List[HChunk] = []

        for context in contexts:
            start = 0
            while start < len(context.context):
                end = min(start + self.chunkSize, len(context.context))
                chunk = context.context[start:end]
                chunkId = self.__generate_id(chunk, context.contextId)
                chunkObj = HChunk(
                    chunkId,
                    context.contextId,
                    chunk,
                    start,
                    end,
                    self.__create_chunk_metadata(document),
                )
                chunks.append(chunkObj)
                if end == len(context.context):
                    break
                start += self.chunkSize - self.chunkOverlap
        return chunks

    def __chunk_text(self, doc: Document, h_manager):

        sections = self.__find_sections(doc)
        h_manager.insert_sections(sections)
        contexts = self.__find_contexts(sections)
        h_manager.insert_contexts(contexts)
        chunks = self.__get_chunks(contexts, doc)
        h_manager.insert_chunks(chunks)
        return chunks

    def process_doc(self) -> List[HChunk]:
        docObjs = self.__make_document_objs()
        h_manager = Manager(self.db_path, is_chunker_type_hierarchical=True)
        h_manager.insert_documents(docObjs)
        chunks = []
        for docObj in docObjs:
            chunks.extend(self.__chunk_text(docObj, h_manager))

        h_manager.close()
        return chunks
