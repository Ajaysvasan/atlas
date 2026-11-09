"""Untrusted files: their names, and whether their content is the kind of file their extension claims.

Nothing here executes or extracts anything to disk. Office files are zip
archives, so they are read entry by entry with every byte counted, against a
total, an entry count and a compression ratio, before any parser sees them.
"""

import json
import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple

from grpc_server.settings import GrpcSettings

MAX_NAME_LENGTH = 200
TEXT_PROBE_BYTES = 64 * 1024
_UNSAFE = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]")
_OFFICE = {"docx": "word/", "pptx": "ppt/", "xlsx": "xl/"}
_TEXT = {"txt", "md", "html", "htm", "json", "csv"}


class Rejection(NamedTuple):
    code: str
    message: str


def safe_filename(name: str | None) -> str:
    """A display name with no directory part, control characters or path syntax; never empty."""
    name = unicodedata.normalize("NFC", name or "")
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = _UNSAFE.sub("_", name).strip(" .")
    if len(name) > MAX_NAME_LENGTH:
        stem, dot, suffix = name.rpartition(".")
        name = (stem[: MAX_NAME_LENGTH - len(suffix) - 1] + dot + suffix) if dot and len(suffix) <= 10 \
            else name[:MAX_NAME_LENGTH]
    return name or "upload"


def extension(name: str) -> str:
    suffix = PurePosixPath(name).suffix
    return suffix[1:].lower() if suffix else ""


def validate(path: Path, name: str, settings: GrpcSettings) -> Rejection | None:
    """None if the file may be ingested; otherwise why not."""
    kind = extension(name)
    if kind not in settings.allowed_extensions:
        return Rejection("UNSUPPORTED_TYPE",
                         f"'.{kind}' files are not accepted" if kind else "files need an extension")
    size = path.stat().st_size
    if size == 0:
        return Rejection("EMPTY", "the file is empty")
    if size > settings.max_file_bytes:
        return Rejection("TOO_LARGE", f"the file is larger than {settings.max_file_bytes} bytes")
    if kind == "pdf":
        return _pdf(path)
    if kind in _OFFICE:
        return _office(path, _OFFICE[kind], settings)
    return _text(path, kind)


def _pdf(path: Path) -> Rejection | None:
    with open(path, "rb") as handle:
        head = handle.read(1024)
    if b"%PDF-" not in head:
        return Rejection("MALFORMED", "the file is not a PDF")
    return None


def _office(path: Path, folder: str, settings: GrpcSettings) -> Rejection | None:
    if not zipfile.is_zipfile(path):
        return Rejection("MALFORMED", "the file is not an Office document")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > settings.max_archive_entries:
                return Rejection("MALFORMED", "the document has too many parts")
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or not any(n.startswith(folder) for n in names):
                return Rejection("MALFORMED", "the file is not the Office document its extension claims")
            total = 0
            for entry in entries:
                if entry.flag_bits & 0x1:
                    return Rejection("MALFORMED", "encrypted documents are not accepted")
                if entry.filename.startswith("/") or ".." in PurePosixPath(entry.filename).parts:
                    return Rejection("MALFORMED", "the document contains an unsafe part name")
                # Declared sizes can lie; count what actually decompresses.
                actual = 0
                with archive.open(entry) as part:
                    while chunk := part.read(1024 * 1024):
                        actual += len(chunk)
                        total += len(chunk)
                        if total > settings.max_uncompressed_bytes:
                            return Rejection("TOO_LARGE", "the document expands beyond the allowed size")
                if entry.compress_size and actual / entry.compress_size > settings.max_compression_ratio \
                        and actual > 1024 * 1024:
                    return Rejection("MALFORMED", "the document is compressed suspiciously far")
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, NotImplementedError, RuntimeError):
        return Rejection("MALFORMED", "the document could not be read")
    return None


def _text(path: Path, kind: str) -> Rejection | None:
    with open(path, "rb") as handle:
        probe = handle.read(TEXT_PROBE_BYTES)
    if b"\x00" in probe:
        return Rejection("MALFORMED", "the file is binary, not text")
    controls = sum(1 for b in probe if b < 32 and b not in (9, 10, 12, 13))
    if probe and controls / len(probe) > 0.01:
        return Rejection("MALFORMED", "the file is binary, not text")
    if kind == "json":
        try:
            with open(path, "rb") as handle:
                json.loads(handle.read().decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            return Rejection("MALFORMED", "the file is not valid JSON")
    return None
