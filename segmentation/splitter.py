import re
from typing import List, Dict
import spacy
import tiktoken

nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer", "textcat"])


def is_header(line: str) -> bool:
    """
    Determine if a text line represents a document header or section title.

    Analyzes line patterns to identify headers based on common formatting conventions
    including numbered sections (1.0, 2.1.3), titled sections ending with colons,
    and standard document structure markers.

    Args:
        line: Single line of text to analyze for header characteristics

    Returns:
        True if the line matches header patterns, False otherwise
    """
    line = line.strip()
    if not line:
        return False

    # Pattern for numbered sections: "1.0", "2.1", "3.1.2.", etc.
    if re.match(r"^\d+(\.\d+)*\s*[:.]?\s+", line):
        return True

    # Pattern for titled sections: "Phase 1:", "Hardware Engineering:", etc.
    if line.endswith(":") and len(line.split()) < 7:
        return True

    return False


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse document text into structured sections with headers and body content.

    Processes raw document text by identifying headers and grouping subsequent content
    into logical sections. Each section contains a header (title) and body (content).
    Automatically handles documents that start without an explicit header by using
    a default "Introduction" header.

    Args:
        text: Raw document text to be segmented into sections

    Returns:
        List of dictionaries, each containing:
        - 'header': Section title or heading text
        - 'body': Section content text (excluding header)

    Note:
        - Empty sections (headers without content) are automatically filtered out
        - Preserves original text formatting within section bodies
        - Uses intelligent header detection for various document formats
    """
    sections = []
    current_body = []
    current_header = "Introduction"  # Default header for the first block of text

    for line in text.splitlines():
        if is_header(line):
            # If a new header is found, save the previous section
            if current_body:
                sections.append({
                    "header": current_header.strip(),
                    "body": "\n".join(current_body).strip()
                })
            # Start the new section
            current_header = line
            current_body = []
        else:
            current_body.append(line)

    # Append the last section
    if current_body:
        sections.append({
            "header": current_header.strip(),
            "body": "\n".join(current_body).strip()
        })

    return [s for s in sections if s.get("body")]