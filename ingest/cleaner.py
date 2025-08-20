import re
from typing import Dict, List, Tuple


def clean_text(raw: str) -> str:
    """
    Enhanced text cleaning specifically designed for business process documents.

    Removes document metadata, boilerplate content, and formatting artifacts
    while preserving essential business process structure including numbered
    sections, actor roles, sequence indicators, and decision points. Optimizes
    text for downstream workflow extraction and analysis.

    Args:
        raw: Raw document text content from file readers

    Returns:
        Cleaned text with preserved process structure and normalized formatting
    """
    # Normalize line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Enhanced header/footer pattern removal
    text = remove_document_metadata(text)

    # Preserve important business process structure
    text = preserve_process_structure(text)

    # Clean up formatting while preserving meaning
    text = normalize_formatting(text)

    # Final cleanup
    text = final_cleanup(text)

    return text


def remove_document_metadata(text: str) -> str:
    """
    Remove document metadata and boilerplate content that doesn't contribute to process understanding.

    Identifies and removes common document artifacts including page numbers,
    version information, copyright notices, headers, footers, and other
    administrative content that interferes with workflow extraction.

    Args:
        text: Document text with potential metadata and boilerplate content

    Returns:
        Text with metadata patterns removed using multiline regex matching
    """

    # Enhanced patterns for business documents
    metadata_patterns = [
        # Page numbers and document info
        r"Page\s*\d+\s*of\s*\d+",
        r"^\s*Page\s*\d+\s*$",
        r"Document\s*(Version|Rev|Revision).*?(\d+\.?\d*|\d{4}-\d{2}-\d{2}).*?$",

        # Headers and footers
        r"^\s*(Header|Footer):.*?$",
        r"Copyright.*?(\d{4}|\d{4}-\d{4}).*?$",
        r"Confidential.*?$",
        r"Proprietary.*?$",
        r"Internal\s*Use\s*Only.*?$",

        # Author and date metadata
        r"^(Author|Created|Last\s*Updated|Modified|Reviewed):.*?$",
        r"^(Date|Time).*?:.*?$",

        # Table of contents and navigation
        r"Table\s*of\s*Contents.*?$",
        r"^\s*\.{3,}.*?\d+\s*$",  # TOC dots and page numbers
        r"^(Previous|Next|Back\s*to):.*?$",

        # Version control and approval blocks
        r"Approval\s*Matrix.*?$",
        r"Revision\s*History.*?$",
        r"Change\s*Log.*?$",

        # Legal and compliance boilerplate
        r"This\s*document.*?confidential.*?$",
        r"Distribution.*?restricted.*?$",

        # File paths and technical metadata
        r"[A-Z]:\\.*?\.(?:doc|pdf|txt)",
        r"^File\s*Path:.*?$",
        r"^Location:.*?$"
    ]

    for pattern in metadata_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text


def preserve_process_structure(text: str) -> str:
    """
    Preserve and normalize important business process structural elements.

    Standardizes numbering formats, phase headers, step indicators, and
    bullet points while maintaining their semantic meaning for workflow
    extraction. Ensures consistent formatting of actor roles and process
    sequence markers.

    Args:
        text: Document text with mixed formatting of structural elements

    Returns:
        Text with normalized and preserved process structure elements
    """

    # Preserve numbered processes and phases
    # Convert various numbering formats to consistent format
    text = re.sub(r"^(\s*)(\d+)[\.\)]\s*([A-Z])", r"\1\2. \3", text, flags=re.MULTILINE)

    # Preserve phase headers
    text = re.sub(r"^(\s*)(Phase\s*\d+)", r"\1\2:", text, flags=re.MULTILINE | re.IGNORECASE)

    # Preserve step indicators
    text = re.sub(r"^(\s*)(Step\s*\d+)", r"\1\2:", text, flags=re.MULTILINE | re.IGNORECASE)

    # Convert bullet points to consistent format but preserve them
    # Keep bullets for lists but standardize format
    text = re.sub(r"^\s*[\u2022\u2023\u25E6\u2043\u2219]\s*", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\-\*]\s*", "• ", text, flags=re.MULTILINE)

    # Preserve actor indicators (roles in caps or quotes)
    # This helps with later actor extraction
    text = re.sub(r'\b([A-Z][A-Z\s]{2,})\b', r'\1', text)  # Preserve CAPS roles

    return text


def normalize_formatting(text: str) -> str:
    """
    Normalize text formatting while preserving business process semantics.

    Standardizes whitespace, removes excessive line breaks, normalizes
    Unicode characters, and fixes common document conversion artifacts.
    Maintains paragraph structure and process flow indicators.

    Args:
        text: Document text with inconsistent formatting

    Returns:
        Text with normalized formatting and preserved semantic structure
    """

    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]{2,}", " ", text)  # Multiple spaces to single space

    # Preserve intentional line breaks in process descriptions
    # But remove excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple blank lines to double

    # Normalize Unicode quotation marks and remove unnecessary quotes around process names
    text = text.replace('\u201c', '')  # Remove left double quotation mark
    text = text.replace('\u201d', '')  # Remove right double quotation mark
    text = text.replace('\u2018', '')  # Remove left single quotation mark
    text = text.replace('\u2019', '')  # Remove right single quotation mark
    text = text.replace('\u201a', '')  # Remove single low-9 quotation mark
    text = text.replace('\u201e', '')  # Remove double low-9 quotation mark
    text = text.replace('\u2039', '')  # Remove single left-pointing angle quotation mark
    text = text.replace('\u203a', '')  # Remove single right-pointing angle quotation mark
    text = text.replace('\u00ab', '')  # Remove left-pointing double angle quotation mark
    text = text.replace('\u00bb', '')  # Remove right-pointing double angle quotation mark

    # Also clean up any remaining quoted process names that might cause JSON escaping
    text = re.sub(r'"([^"]*(?:rule|process|task|event|gateway)[^"]*)"', r'\1', text, flags=re.IGNORECASE)

    # Normalize dashes (important for process flows)
    text = re.sub(r"[–—]", "-", text)

    # Fix common OCR/conversion errors
    text = fix_common_errors(text)

    return text


def fix_common_errors(text: str) -> str:
    """
    Correct common OCR and document conversion errors that affect process text.

    Identifies and fixes character substitution errors, broken words across
    lines, and spacing issues that commonly occur in digitized business
    documents. Focuses on errors that impact workflow extraction accuracy.

    Args:
        text: Document text with potential OCR or conversion errors

    Returns:
        Text with common character and formatting errors corrected
    """

    # Common OCR substitutions
    ocr_fixes = {
        r"\bl\b": "I",  # lowercase L often confused with I
        r"\b0(?=[A-Za-z])": "O",  # zero confused with O at word boundaries
        r"(?<=[A-Za-z])0\b": "O",
        r"\brn\b": "m",  # rn often confused with m
        r"\bvv": "w",  # double v confused with w
    }

    for pattern, replacement in ocr_fixes.items():
        text = re.sub(pattern, replacement, text)

    # Fix broken words across lines (common in PDF conversion)
    text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])\s*([A-Z])", r"\1 \2", text)

    return text


def final_cleanup(text: str) -> str:
    """
    Perform final text validation and cleanup operations.

    Removes empty lines, normalizes paragraph breaks, and ensures
    consistent text structure for downstream processing. Validates
    that essential content is preserved while removing formatting artifacts.

    Args:
        text: Document text after primary cleaning operations

    Returns:
        Final cleaned text ready for workflow extraction processing
    """

    # Remove empty lines at start and end of paragraphs
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line:  # Keep non-empty lines
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1]:  # Keep single empty line as paragraph break
            cleaned_lines.append("")

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    text = '\n'.join(cleaned_lines)

    # Final strip
    return text.strip()


def analyze_cleaning_impact(original: str, cleaned: str) -> Dict[str, any]:
    """
    Analyze the impact of cleaning operations for quality assessment and debugging.

    Compares original and cleaned text to calculate preservation ratios,
    identify removed content types, and flag potential over-cleaning issues.
    Provides metrics for evaluating cleaning effectiveness and content quality.

    Args:
        original: Raw document text before cleaning operations
        cleaned: Processed text after all cleaning operations

    Returns:
        Dictionary containing cleaning impact analysis:
        - removal_stats: Statistics on removed content and preservation ratios
        - quality_flags: Warnings about potential over-cleaning issues
        - cleaning_effective: Boolean indicating optimal cleaning balance
    """

    original_lines = original.count('\n') + 1
    cleaned_lines = cleaned.count('\n') + 1

    original_words = len(original.split())
    cleaned_words = len(cleaned.split())

    # Calculate preservation ratios
    line_preservation = cleaned_lines / max(original_lines, 1)
    word_preservation = cleaned_words / max(original_words, 1)

    # Identify what was removed
    removed_content = {
        "metadata_removed": original_lines - cleaned_lines,
        "words_removed": original_words - cleaned_words,
        "line_preservation_ratio": line_preservation,
        "word_preservation_ratio": word_preservation
    }

    # Quality checks
    quality_flags = []
    if word_preservation < 0.7:
        quality_flags.append("High word removal - check for over-cleaning")
    if line_preservation < 0.5:
        quality_flags.append("High line removal - verify structure preservation")

    return {
        "removal_stats": removed_content,
        "quality_flags": quality_flags,
        "cleaning_effective": 0.8 <= word_preservation <= 0.95
    }


def preprocess_for_business_process_extraction(text: str) -> str:
    """
    Apply specialized preprocessing to optimize text for business process extraction.

    Enhances actor visibility, sequence indicators, and decision patterns
    to improve downstream workflow extraction accuracy. Called after basic
    cleaning to prepare text for semantic analysis and task identification.

    Args:
        text: Cleaned document text ready for process-specific enhancement

    Returns:
        Text optimized for business process extraction with enhanced patterns
    """

    # Enhance actor visibility
    text = enhance_actor_patterns(text)

    # Enhance sequence indicators
    text = enhance_sequence_patterns(text)

    # Enhance decision points
    text = enhance_decision_patterns(text)

    return text


def enhance_actor_patterns(text: str) -> str:
    """
    Standardize and enhance actor role patterns for improved extraction.

    Expands common role abbreviations to full names and normalizes
    actor references to improve consistency in workflow extraction.
    Helps identify process participants and lane assignments.

    Args:
        text: Document text with various actor role formats

    Returns:
        Text with standardized actor patterns and expanded abbreviations
    """

    # Standardize common role patterns
    role_patterns = [
        (r"\bPM\b", "Project Manager"),
        (r"\bQA\b", "Quality Assurance"),
        (r"\bBA\b", "Business Analyst"),
        (r"\bSME\b", "Subject Matter Expert"),
        (r"\bCVL\b", "Concept Validation Lead"),
        (r"\bPSC\b", "Product Steering Committee")
    ]

    for pattern, replacement in role_patterns:
        text = re.sub(pattern, replacement, text)

    return text


def enhance_sequence_patterns(text: str) -> str:
    """
    Standardize sequence and temporal indicators for workflow extraction.

    Normalizes various forms of sequence markers to consistent patterns
    that are easier for extraction algorithms to identify. Improves
    accuracy of workflow step ordering and dependency detection.

    Args:
        text: Document text with various sequence indicator formats

    Returns:
        Text with standardized sequence patterns and temporal markers
    """

    # Standardize sequence indicators
    sequence_patterns = [
        (r"\bAfter\s+that,?\s*", "Subsequently, "),
        (r"\bFollowing\s+this,?\s*", "Next, "),
        (r"\bUpon\s+completion,?\s*", "After completion, ")
    ]

    for pattern, replacement in sequence_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def enhance_decision_patterns(text: str) -> str:
    """
    Standardize decision and conditional patterns for gateway extraction.

    Normalizes various forms of conditional statements and decision
    indicators to consistent patterns that improve gateway detection
    and branching logic identification in workflow extraction.

    Args:
        text: Document text with various decision indicator formats

    Returns:
        Text with standardized decision patterns and conditional markers
    """

    # Standardize decision indicators
    decision_patterns = [
        (r"\bIn\s+case\s+of\b", "If"),
        (r"\bShould\s+(\w+)\s+(\w+)", r"If \1 \2"),
        (r"\bProvided\s+that\b", "If")
    ]

    for pattern, replacement in decision_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text