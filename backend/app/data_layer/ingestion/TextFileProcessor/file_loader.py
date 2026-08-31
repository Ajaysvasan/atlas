import os
from pathlib import Path
from typing import Dict, Iterable, List, Set

from config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_FILE_SIZE = 64 * 1024 * 1024

NON_DOCUMENT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".ico", ".webp",
    ".heic", ".psd", ".ai", ".eps", ".raw",
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".iso",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".class",
    ".pyc", ".pyo", ".pyd", ".wasm",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".db", ".sqlite", ".sqlite3", ".mdb", ".dat", ".idx", ".pack",
    ".npy", ".npz", ".pt", ".pth", ".onnx", ".gguf", ".safetensors", ".h5",
    ".pkl", ".pickle", ".joblib", ".parquet", ".feather", ".arrow",
    ".lock", ".swp", ".ds_store",
}

IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".bzr",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".tox", ".eggs", "site-packages",
    ".idea", ".vscode", "dist", "build", ".next", ".cache",
}


class FileLoader:
    """Collects candidate documents from a directory tree.

    The policy is open: anything that is not a known binary format is offered to
    the extractor, which decides whether it can turn the bytes into text. An
    allowlist here would have to be extended for every new format the extractor
    learns, and silently dropped the rest without saying so.
    """

    def __init__(
        self,
        allowed_extensions: Iterable[str] | None = None,
        excluded_extensions: Iterable[str] | None = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ):
        self.allowed_extensions = (
            {ext.lower() for ext in allowed_extensions}
            if allowed_extensions is not None
            else None
        )
        self.excluded_extensions = (
            {ext.lower() for ext in excluded_extensions}
            if excluded_extensions is not None
            else set(NON_DOCUMENT_EXTENSIONS)
        )
        self.max_file_size = max_file_size

    def __is_directory(self, path: str) -> bool:
        return os.path.isdir(path)

    def __get_file_category(self, file_name: str) -> str:
        extension = Path(file_name).suffix.lower()
        if self.allowed_extensions is not None:
            return extension[1:] if extension in self.allowed_extensions else "unknown"
        if extension in self.excluded_extensions:
            return "unknown"
        return extension[1:] if extension else "noext"

    def __should_load(self, item_path: str) -> bool:
        if os.path.basename(item_path).startswith("."):
            return False
        try:
            size = os.path.getsize(item_path)
        except OSError as e:
            logger.warning(f"Could not stat '{item_path}': {e}")
            return False
        if size == 0:
            logger.debug(f"Skipped empty file: '{item_path}'")
            return False
        if size > self.max_file_size:
            logger.warning(
                f"Skipped '{item_path}': {size} bytes exceeds the {self.max_file_size} byte limit."
            )
            return False
        return True

    def __scan_directory(
        self, path: str, loaded_files: Dict[str, List[Path]], visited: Set[str]
    ) -> None:
        # Symlinked directories are resolved and remembered: a link pointing at
        # an ancestor otherwise recurses until Python's stack limit, and the
        # RecursionError was caught below as if it were an unreadable folder,
        # so the scan returned a partial tree and reported success.
        real_path = os.path.realpath(path)
        if real_path in visited:
            logger.debug(f"Skipped already visited directory: '{path}'")
            return
        visited.add(real_path)

        try:
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)

                if self.__is_directory(item_path):
                    if item in IGNORED_DIRECTORIES or item.startswith("."):
                        continue
                    self.__scan_directory(item_path, loaded_files, visited)
                    continue

                category = self.__get_file_category(item_path)
                if category == "unknown":
                    logger.debug(f"Ignored non-document file during scan: '{item_path}'")
                    continue
                if not self.__should_load(item_path):
                    continue
                loaded_files.setdefault(category, []).append(Path(item_path))

        except PermissionError:
            logger.error(f"Permission denied while scanning directory: '{path}'")
        except OSError as e:
            logger.error(f"Error scanning directory '{path}': {e}", exc_info=True)

    def load_files(self, folder_path) -> Dict[str, List[Path]]:
        if not os.path.exists(folder_path):
            logger.error(f"load_files failed: path '{folder_path}' does not exist.")
            raise ValueError(f"The provided path '{folder_path}' does not exist.")
        if not self.__is_directory(folder_path):
            logger.error(f"load_files failed: path '{folder_path}' is not a directory.")
            raise ValueError(f"The provided path '{folder_path}' is not a directory.")

        logger.info(f"Scanning directory '{folder_path}' for documents...")
        loaded_files: Dict[str, List[Path]] = {}

        self.__scan_directory(str(folder_path), loaded_files, set())
        total_loaded = sum(len(files) for files in loaded_files.values())
        logger.info(
            f"Loaded {total_loaded} file(s) across {len(loaded_files)} file categories from '{folder_path}'."
        )
        return loaded_files
