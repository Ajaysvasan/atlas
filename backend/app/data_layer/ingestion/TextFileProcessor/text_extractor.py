import os
from pathlib import Path
from typing import Dict, Tuple

from data_layer.datalayer_exceptions.datalayer_exceptions import InvalidFileType
from config import get_logger

logger = get_logger(__name__)


class TextExtractor:
    def __init__(self):
        pass


    def _extract_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file '{file_path}' with latin-1 encoding: {e}", exc_info=True)
                return ""
        except Exception as e:
            logger.error(f"Error reading TXT file '{file_path}': {e}", exc_info=True)
            return ""

    def _extract_from_docx(self, file_path: str) -> str:
        try:
            from docx import Document

            doc = Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            return "\n".join(full_text)
        except ImportError:
            logger.warning("Error: python-docx library not installed. Run: pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"Error reading DOCX file '{file_path}': {e}", exc_info=True)
            return ""

    def _extract_from_doc(self, file_path: str) -> str:
        try:
            import textract

            text = textract.process(file_path).decode("utf-8")
            return text
        except ImportError:
            logger.warning("Error: textract library not installed. Run: pip install textract (Note: textract may require additional system dependencies)")
            return ""
        except Exception as e:
            logger.error(f"Error reading DOC file '{file_path}': {e}", exc_info=True)
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2

            text = []
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return "\n".join(text)
        except ImportError:
            logger.warning("Error: PyPDF2 library not installed. Run: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Error reading PDF file '{file_path}': {e}", exc_info=True)
            return ""

    def __extract_text_from_md(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise Exception(f"The Path {file_path} does not exists")
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text

    def extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
        if not os.path.exists(file_path):
            logger.error(f"extract_text_from_file failed: File does not exist: '{file_path}'")
            raise FileNotFoundError

        extension = Path(file_path).suffix.lower()
        logger.debug(f"Extracting text from '{file_path}' (extension: {extension})")

        if extension in [".txt", ".csv", ".html", ".xml"]:
            extracted_text = self._extract_from_txt(file_path)
        elif extension == ".docx":
            extracted_text = self._extract_from_docx(file_path)
        elif extension == ".doc":
            extracted_text = self._extract_from_doc(file_path)
        elif extension == ".pdf":
            extracted_text = self._extract_from_pdf(file_path)
        elif extension == ".md":
            extracted_text = self.__extract_text_from_md(file_path)
        else:
            logger.error(f"Invalid file extension '{extension}' for file '{file_path}'")
            raise InvalidFileType(file_extention=extension)

        return file_path, extracted_text

    def extract_all(self, loaded_files: Dict[str, list]) -> Dict[str, str]:
        extracted_texts = {}
        total_files = sum(len(paths) for paths in loaded_files.values())
        logger.info(f"Starting batch text extraction across {total_files} file(s) in {len(loaded_files)} categories...")

        for category, file_paths in loaded_files.items():
            if file_paths:
                logger.info(f"Extracting category: '{category}' ({len(file_paths)} files)")
            for file_path in file_paths:
                logger.debug(f"Processing extraction for: '{file_path}'")
                text = self.extract_text_from_file(file_path)[1]
                extracted_texts[file_path] = text

        logger.info(f"Successfully extracted text from {len(extracted_texts)} file(s).")
        return extracted_texts
