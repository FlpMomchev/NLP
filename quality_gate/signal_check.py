import re
from typing import Dict, Any
import spacy

_PROCESS_SIGNAL_PATTERN = re.compile(
    r"\b(if|then|else|when|while|for each|in case|otherwise|unless|provided that|once|upon|after|before|until|as soon as)\b",
    flags=re.IGNORECASE,
)

def check_process_signals(doc: spacy.tokens.Doc, threshold: int = 0) -> Dict[str, Any]:
    """
    Analyze document content for business process signal indicators.

    Scans the processed spaCy document to identify linguistic patterns that indicate
    business process content, including temporal markers, conditional statements,
    decision points, and workflow-specific vocabulary. Returns assessment results
    for content classification and quality evaluation.

    Args:
        doc: Processed spaCy document object containing tokenized and analyzed text

    Returns:
        Dictionary containing:
        - 'pass': Boolean indicating if sufficient process signals were detected
        - 'signal_count': Integer count of identified process indicators
        - 'signals_found': List of detected signal types and examples
        - 'message': String with detailed analysis or warning message
    """
    matches = _PROCESS_SIGNAL_PATTERN.findall(doc.text)
    count = len(matches)

    return {
        "pass": count >= threshold,
        "count": count,
        "message": "" if count >= threshold else f"Low count of process signal words ({count}) suggests non-process text."
    }