import spacy
import coreferee
from typing import Dict, Any
from ambiguity.role import PRONOUNS

# Lädt SpaCy und Coreferee einmalig
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coreferee")

def check_coref_gaps(text: str) -> Dict[str, Any]:
    """
    Identify unresolved pronoun references in business process text.

    Performs coreference resolution analysis to detect pronouns that lack
    clear antecedents within the document context. Unresolved pronouns
    create ambiguity in actor identification and responsibility assignment,
    reducing workflow extraction accuracy and BPMN modeling quality.

    Args:
        text: Business process description text to analyze for coreference gaps

    Returns:
        Dictionary containing coreference analysis results:
        - pass: Boolean indicating if all pronouns have resolved references
        - flagged: List of unresolved pronoun tokens found in text
        - ratio: Proportion of unresolved pronouns to total pronouns
    """
    doc = nlp(text)
    linked_indices = set()

    # Collect all token indices that participate in coreference chains
    for chain in doc._.coref_chains:
        for mention_idx in chain:  # Each mention_idx is an integer token index
            linked_indices.add(mention_idx)

    flagged = []
    pronouns = [tok for tok in doc if tok.pos_ == "PRON" and tok.text.lower() in PRONOUNS]
    for tok in pronouns:
        if tok.i not in linked_indices:
            flagged.append(tok.text)
    total = len(pronouns)
    ratio = len(flagged) / total if total else 0.0
    return {"pass": ratio == 0.0, "flagged": flagged, "ratio": ratio}
