import spacy
from typing import Dict, Any

# Lädt SpaCy einmalig
nlp = spacy.load("en_core_web_sm")

# Pronomen, die normalerweise eine Rolle/einen Vorgänger brauchen
PRONOUNS = {
    "it", "they", "them", "this", "that", "these", "those",
    "he", "she", "we", "i"
}

def check_pronoun_ambiguity(text: str) -> Dict[str, Any]:
    """
    Flag pronouns that are likely ambiguous without clear antecedents.

    Counts pronouns in a fixed set and returns a pass flag and ratio.

    Args:
        text: Text to analyze.

    Returns:
        Dict with pass, flagged pronouns, and ratio.
    """
    doc = nlp(text)
    # Alle Pronomen sammeln
    pronouns = [tok for tok in doc if tok.pos_ == "PRON"]
    flagged = [tok.text for tok in pronouns if tok.text.lower() in PRONOUNS]
    total = len(pronouns)
    ratio = len(flagged) / total if total else 0.0
    return {
        "pass": len(flagged) == 0,
        "flagged": flagged,
        "ratio": ratio
    }