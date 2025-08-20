import spacy
from typing import Dict, List, Any
from .signal_check import check_process_signals

nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat", "lemmatizer"])


def _sentence_has_actor_action(sent: spacy.tokens.Span) -> bool:
    """
    Evaluate sentence for proper actor-action structure using dependency parsing.

    Analyzes sentence syntax to identify well-formed actor-action patterns
    essential for business process descriptions. Checks for presence of root
    verbs and clear subjects to ensure sentences describe actionable activities
    rather than passive descriptions or incomplete statements.

    Args:
        sent: spaCy sentence span to analyze for actor-action patterns

    Returns:
        True if sentence contains clear actor-action structure, False otherwise
    """
    has_root_verb = any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in sent)
    has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
    return has_root_verb and has_subject


def _check_actor_action_ratio(doc: spacy.tokens.Doc, threshold=0.90) -> Dict[str, Any]:
    """
    Assess the proportion of process-oriented sentences in document content.

    Calculates the ratio of sentences with proper actor-action structure to
    total sentences, comparing against configurable threshold to determine
    if content has sufficient procedural characteristics for reliable workflow
    extraction and BPMN generation.

    Args:
        doc: Processed spaCy document containing sentence boundaries and syntax
        threshold: Minimum ratio (0.0-1.0) required for content approval

    Returns:
        Dictionary containing:
        - 'pass': Boolean indicating if ratio meets threshold requirement
        - 'ratio': Calculated actor-action ratio as float value
        - 'message': Empty string if passed, warning message if failed
    """
    sents = list(doc.sents)
    if not sents:
        return {"pass": False, "ratio": 0, "message": "No sentences found."}

    good_sentences = sum(1 for s in sents if _sentence_has_actor_action(s))
    ratio = good_sentences / len(sents)

    return {
        "pass": ratio >= threshold, "ratio": ratio,
        "message": "" if ratio >= threshold else f"Low actor/action ratio ({ratio:.2f}) may indicate non-process text."
    }


def run_quality_check(text: str) -> Dict[str, Any]:
    """
    Execute comprehensive semantic quality assessment for business process text.

    Performs multi-dimensional analysis of input text to determine suitability
    for workflow extraction and BPMN generation. Combines actor-action pattern
    analysis with process signal detection to provide overall quality assessment
    and actionable feedback for content improvement.

    Args:
        text: Raw input text to be evaluated for process extraction quality

    Returns:
        Dictionary containing comprehensive quality assessment:
        - 'actor_action': Results from actor-action pattern analysis
        - 'signals': Results from process signal detection
        - 'overall_pass': Boolean indicating if text meets quality thresholds
        - 'feedback': List of specific improvement recommendations
    """
    doc = nlp(text)
    feedback: List[str] = []

    actor_action_results = _check_actor_action_ratio(doc)
    if not actor_action_results["pass"]:
        feedback.append(actor_action_results["message"])

    signal_results = check_process_signals(doc)
    if not signal_results["pass"]:
        feedback.append(signal_results["message"])

    overall_pass = all([actor_action_results["pass"], signal_results["pass"]])

    return {
        "actor_action": actor_action_results,
        "signals": signal_results,
        "overall_pass": overall_pass,
        "feedback": feedback,
    }