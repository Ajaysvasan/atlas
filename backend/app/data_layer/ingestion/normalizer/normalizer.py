import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from data_layer.ingestion.metadata.metadata import NormalizedTextMetaData
from data_layer.ingestion.nodes.nodes import NormalizedContent, SectionSpan

NORMALIZATION_VERSION = "rag_v2"

HEADING_MAX_LENGTH = 80
HEADING_MAX_WORDS = 12
HEADING_MIN_LETTERS = 3
HEADING_MIN_LETTER_RATIO = 0.5

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*$")
_SETEXT_UNDERLINE = re.compile(r"^(?:=|-){3,}\s*$")
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)+|\d+[.)])\s+[A-Z]\S*")
_UPPERCASE_HEADING = re.compile(r"^(?=.*[A-Z])[^a-z]+$")
_CODE_FENCE = re.compile(r"^\s*(?:```|~~~)")
_SENTENCE_TAIL = (".", ",", ";", ":", "!", "?")


def _is_mostly_letters(line: str) -> bool:
    """Guards the ALL CAPS rule, which otherwise fires on anything without a
    lowercase letter — a spreadsheet row like "Q1 | 1.2M" included."""
    letters = sum(1 for char in line if char.isalpha())
    visible = sum(1 for char in line if not char.isspace())
    return (
        letters >= HEADING_MIN_LETTERS
        and letters >= visible * HEADING_MIN_LETTER_RATIO
    )


def _heading_name(line: str) -> str | None:
    """The heading this line announces, or None if it is body text.

    Three shapes are recognised because real documents use all of them:
    markdown `# Title`, numbered `1.2 Title`, and ALL CAPS. Setext underlines
    need the following line, so they are handled by the caller.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > HEADING_MAX_LENGTH:
        return None

    atx = _ATX_HEADING.match(stripped)
    if atx:
        return atx.group(2).strip()

    if _NUMBERED_HEADING.match(stripped) and not stripped.endswith(_SENTENCE_TAIL):
        return stripped

    if (
        _UPPERCASE_HEADING.match(stripped)
        and len(stripped.split()) <= HEADING_MAX_WORDS
        and _is_mostly_letters(stripped)
    ):
        return stripped

    return None


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
        """Collapse runs of spaces and tabs, leaving line breaks alone.

        A plain \\s+ collapse here is what used to flatten every document to a
        single line, after which no heading regex, paragraph split or "\\n\\n"
        chunk separator downstream could ever match.
        """
        return re.sub(r"[^\S\n]+", " ", text)

    def __process_line(self, line: str) -> str:
        if self.remove_urls:
            line = self._replace_urls(line)

        if self.remove_emails:
            line = self._replace_emails(line)

        if self.lowercase:
            line = line.lower()

        if self.remove_numbers:
            line = self._remove_numbers(line)

        if self.remove_punctuation:
            line = self._remove_punctuation(line)

        if self.remove_special_chars:
            line = self._remove_special_chars(line)

        if self.remove_extra_whitespace:
            line = self._remove_extra_whitespace(line)

        if self.strip_whitespace:
            line = line.strip()

        return line

    def __group_lines(self, text: str) -> List[Tuple[bool, List[str]]]:
        """Split raw text into (is_heading, lines) blocks.

        Runs before any lexical transform, so ALL CAPS headings are still
        recognisable even when the profile lowercases, and `#` is still there
        when the profile strips punctuation.
        """
        lines = re.sub(r"\r\n?", "\n", text).split("\n")
        blocks: List[Tuple[bool, List[str]]] = []
        body: List[str] = []
        in_code_fence = False
        index = 0

        while index < len(lines):
            line = lines[index]
            if _CODE_FENCE.match(line):
                in_code_fence = not in_code_fence

            name = None
            if not in_code_fence:
                name = _heading_name(line)
                if (
                    name is None
                    and line.strip()
                    and len(line.strip()) <= HEADING_MAX_LENGTH
                    and index + 1 < len(lines)
                    and _SETEXT_UNDERLINE.match(lines[index + 1])
                ):
                    name = line.strip()
                    index += 1

            if name is not None:
                if body:
                    blocks.append((False, body))
                    body = []
                blocks.append((True, [name]))
            elif line.strip():
                body.append(line)
            elif body:
                blocks.append((False, body))
                body = []

            index += 1

        if body:
            blocks.append((False, body))
        return blocks

    def __build_blocks(self, text: str) -> Tuple[str, Tuple[SectionSpan, ...]]:
        """Normalize text and report where each section landed in the result."""
        if not text:
            return "", ()

        joiner = " " if self.remove_newlines else "\n"
        rendered: List[Tuple[bool, str, str]] = []
        for is_heading, lines in self.__group_lines(text):
            processed = [line for line in map(self.__process_line, lines) if line]
            if not processed:
                continue
            rendered.append((is_heading, lines[0].strip(), joiner.join(processed)))

        content = "\n\n".join(block for _, _, block in rendered)

        headings: List[Tuple[str, int, int]] = []
        cursor = 0
        for is_heading, name, block in rendered:
            if is_heading:
                headings.append((name, cursor, cursor + len(block)))
            cursor += len(block) + 2

        spans: List[SectionSpan] = []
        for position, (name, start, end) in enumerate(headings):
            content_start = min(end + 2, len(content))
            if position + 1 < len(headings):
                content_end = max(content_start, headings[position + 1][1] - 2)
            else:
                content_end = len(content)
            spans.append(SectionSpan(name, start, end, content_start, content_end))

        return content, tuple(spans)

    def __normalize(self, file_path, text: str) -> NormalizedContent:
        source_path = str(file_path)
        file_name = Path(source_path).name
        file_type = Path(source_path).suffix.lower()
        ingestion_time = datetime.now(timezone.utc).isoformat()

        normalized_text, sections = self.__build_blocks(text)
        document_id = self.__generate_document_id(file_name, source_path, normalized_text)
        content_id = self.__generate_content_id(normalized_text)

        return NormalizedContent(
            content=normalized_text,
            has_section=len(sections) > 0,
            meta_data=NormalizedTextMetaData(
                document_id,
                source_path,
                file_name,
                file_type,
                ingestion_time,
                NORMALIZATION_VERSION,
                content_id,
            ),
            sections=sections,
        )

    def normalize_text(self, file_path, text) -> NormalizedContent:
        return self.__normalize(file_path, text)

    def normalize_all(self, extracted_texts: Dict[str, str]) -> List[NormalizedContent]:
        return [
            self.__normalize(file_path, text)
            for file_path, text in extracted_texts.items()
        ]


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
