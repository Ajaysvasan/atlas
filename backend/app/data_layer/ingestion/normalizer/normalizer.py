import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from data_layer.ingestion.metadata.metadata import NormalizedTextMetaData
from data_layer.ingestion.nodes.nodes import NormalizedContent

NORMALIZATION_VERSION = "rag_v1"


class TextNormalizer:
    def __init__(
        self,
        lowercase: bool = True,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_newlines: bool = False,
        strip_whitespace: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_newlines = remove_newlines
        self.strip_whitespace = strip_whitespace

    def __generate_content_id(self, normalized_text: str) -> str:
        hash_object = hashlib.sha256(normalized_text.encode("utf-8"))
        hex_digest = hash_object.hexdigest()
        return str(hex_digest)

    def __generate_document_id(self, *args) -> str:
        args = [str(arg) for arg in args]
        value = "".join(args)
        hash_object = hashlib.sha256(value.encode("utf-8"))
        hex_digest = hash_object.hexdigest()
        return str(hex_digest)

    def _replace_urls(self, text: str, placeholder: str = "[URL]") -> str:
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, placeholder, text)
        www_pattern = r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(www_pattern, placeholder, text)
        return text

    def _replace_emails(self, text: str, placeholder: str = "[EMAIL]") -> str:
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return re.sub(email_pattern, placeholder, text)

    def _remove_numbers(self, text: str) -> str:
        return re.sub(r"\d+", "", text)

    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def _remove_special_chars(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s.,!?;:\-\']", "", text)

    def _remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def _normalize_newlines(self, text: str) -> str:
        """Normalize newlines while preserving paragraph structure"""
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text

    def __find_section_names(self, text):
        sections = []
        pattern = re.compile(r"^(?=.*[A-Z])[A-Z\s]+$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        for i, match in enumerate(matches):
            sectionName = match.group().strip()
            sections.append(sectionName)
        return sections

    def __create_normalized_content_meta_data(
        self,
        document_id,
        source_file,
        file_type,
        ingestion_time,
        normalization_version,
        content_hash,
    ):
        file_name = Path(source_file).name
        return NormalizedTextMetaData(
            document_id,
            source_file,
            file_name,
            file_type,
            ingestion_time,
            normalization_version,
            content_hash,
        )

    def _has_section(self, normalizedText):
        return len(self.__find_section_names(normalizedText)) != 0

    def __process_text(self, text: str) -> str:
        if not text:
            return ""

        if self.remove_urls:
            text = self._replace_urls(text)

        if self.remove_emails:
            text = self._replace_emails(text)

        if self.remove_newlines:
            text = self._normalize_newlines(text)

        if self.lowercase:
            text = text.lower()
        if self.remove_numbers:
            text = self._remove_numbers(text)

        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        if self.remove_special_chars:
            text = self._remove_special_chars(text)

        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)

        if self.strip_whitespace:
            text = text.strip()

        return text

    def normalize_text(self, file_path, text) -> NormalizedContent:
        file_name = Path(file_path).name
        file_type = Path(file_path).suffix.lower()
        ingestion_time = datetime.now(timezone.utc).isoformat()
        has_section = self._has_section(text)
        normalized_text = self.__process_text(text)
        document_id = self.__generate_document_id(file_name, file_path, normalized_text)
        content_id = self.__generate_content_id(normalized_text)
        return NormalizedContent(
            content=normalized_text,
            has_section=has_section,
            meta_data=self.__create_normalized_content_meta_data(
                document_id,
                file_path,
                file_type,
                ingestion_time,
                NORMALIZATION_VERSION,
                content_id,
            ),
        )

    def normalize_all(self, extracted_texts: Dict[str, str]) -> List[NormalizedContent]:
        normalized_documents_contents = []
        for file_path, text in extracted_texts.items():
            file_name = Path(file_path).name
            file_type = Path(file_path).suffix.lower()
            ingestion_time = datetime.now(timezone.utc).isoformat()
            has_section = self._has_section(text)
            normalized_text = self.__process_text(text)
            document_id = self.__generate_document_id(
                file_name, file_path, normalized_text
            )
            content_id = self.__generate_content_id(normalized_text)
            normalized_documents_contents.append(
                NormalizedContent(
                    normalized_text,
                    has_section,
                    self.__create_normalized_content_meta_data(
                        document_id,
                        file_path,
                        file_type,
                        ingestion_time,
                        NORMALIZATION_VERSION,
                        content_id,
                    ),
                )
            )

        return normalized_documents_contents


class NormalizationProfiles:

    @staticmethod
    def rag_ingestion():
        return TextNormalizer(
            lowercase=False,
            remove_extra_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            remove_newlines=True,
            remove_special_chars=False,
            remove_numbers=False,
            remove_punctuation=False,
            strip_whitespace=True,
        )

    @staticmethod
    def minimal():
        return TextNormalizer(
            lowercase=False,
            remove_extra_whitespace=True,
            strip_whitespace=True,
            remove_urls=False,
            remove_emails=False,
        )
