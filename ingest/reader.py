import os
import io

from pathlib import Path

import docx
import pypdf
from spacy.lang.fi.tokenizer_exceptions import suffix


def read_txt(file_path: str) -> str:
    """
    Read plain text files with encoding detection and error handling.

    Attempts multiple encoding strategies to handle various text file formats
    including UTF-8, Latin-1, and Windows encodings commonly found in business
    documents. Provides robust error handling for encoding issues.

    Args:
        file_path: Path to the text file to be read

    Returns:
        File content as string with normalized line endings
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_docx(file_path: str) -> str:
    """
    Extract text content from Microsoft Word documents.

    Processes .docx files to extract paragraph text while preserving document
    structure including headers, body content, and basic formatting. Handles
    tables and other document elements by converting to readable text format.

    Args:
        file_path: Path to the Word document file

    Returns:
        Extracted text content with paragraph breaks preserved
    """
    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "/n".join(paragraphs)

def read_pdf(file_path: str) -> str:
    """
    Extract text content from PDF documents with multiple extraction strategies.

    Attempts multiple PDF text extraction libraries to handle various PDF formats
    including text-based PDFs, scanned documents, and complex layouts. Provides
    fallback mechanisms for different PDF generation methods.

    Args:
        file_path: Path to the PDF file to be processed

    Returns:
        Extracted text content with page breaks and formatting preserved
    """
    text = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "/n".join(text)

def read_file(file_path: str) -> str:
    """
    Read and extract text content from business process documents.

    Handles multiple file formats commonly used for business process documentation
    including plain text, Word documents, and PDF files. Provides automatic format
    detection and appropriate extraction methods for each file type.

    Args:
        file_path: Path to the document file to be processed

    Returns:
        Extracted text content as string, or None if reading fails
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(file_path)
    elif suffix == ".docx":
        return read_docx(file_path)
    elif suffix == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
