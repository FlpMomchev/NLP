import spacy
import re
from typing import Dict, Any, List, Set

nlp = spacy.load("en_core_web_sm")

# Business process specific patterns
BUSINESS_ROLES = {
    "manager", "lead", "team", "committee", "department", "engineer", "analyst",
    "specialist", "coordinator", "director", "head", "representative", "reviewer",
    "stakeholder", "owner", "member", "group", "board", "council", "unit",
    "administrator", "supervisor", "executive", "officer", "consultant"
}

STRONG_ACTION_VERBS = {
    "approve", "reject", "submit", "review", "analyze", "create", "develop",
    "implement", "execute", "validate", "test", "authorize", "assign",
    "delegate", "escalate", "notify", "document", "record", "track",
    "monitor", "evaluate", "assess", "decide", "determine", "investigate"
}

WEAK_ACTION_INDICATORS = {
    "be", "have", "do", "make", "get", "go", "come", "see", "know", "think",
    "want", "need", "should", "could", "might", "may", "will", "would"
}

SEQUENCE_INDICATORS = {
    "first", "second", "third", "next", "then", "after", "before", "upon",
    "once", "when", "subsequently", "following", "prior", "finally"
}

DECISION_INDICATORS = {
    "if", "when", "unless", "provided", "should", "approve", "reject",
    "decide", "determine", "evaluate", "assess", "review", "choose"
}

GENERIC_ACTORS = {
    "it", "they", "this", "that", "system", "process", "document",
    "the process", "the system", "the document", "someone", "anybody"
}


def check_business_process_clarity(text: str) -> Dict[str, Any]:
    """
    Comprehensive assessment of business process modeling readiness.

    Evaluates text quality across multiple dimensions critical for successful
    BPMN generation including actor identification, action specificity, sequence
    clarity, and decision point definition. Provides detailed feedback for
    content improvement and modeling feasibility assessment.

    Args:
        text: Business process description text to be analyzed

    Returns:
        Dictionary containing comprehensive analysis results:
        - pass: Boolean indicating if text meets business process modeling standards
        - flagged: List of specific issues identified during analysis
        - business_process_score: Overall score combining all clarity dimensions
        - actor_clarity: Score for actor role definition quality
        - action_clarity: Score for action verb specificity
        - sequence_clarity: Score for process flow indicators
        - decision_clarity: Score for decision point definition
    """
    doc = nlp(text)

    issues = []
    metrics = {}

    # Actor clarity analysis
    actor_analysis = analyze_actor_clarity(doc)
    issues.extend(actor_analysis["issues"])
    metrics["actor_clarity"] = actor_analysis["clarity_score"]

    # Action clarity analysis
    action_analysis = analyze_action_clarity(doc)
    issues.extend(action_analysis["issues"])
    metrics["action_clarity"] = action_analysis["clarity_score"]

    # Sequence clarity analysis
    sequence_analysis = analyze_sequence_clarity(doc)
    issues.extend(sequence_analysis["issues"])
    metrics["sequence_clarity"] = sequence_analysis["clarity_score"]

    # Decision point clarity
    decision_analysis = analyze_decision_clarity(doc)
    issues.extend(decision_analysis["issues"])
    metrics["decision_clarity"] = decision_analysis["clarity_score"]

    # Overall business process score
    overall_score = (
            metrics["actor_clarity"] * 0.3 +
            metrics["action_clarity"] * 0.3 +
            metrics["sequence_clarity"] * 0.2 +
            metrics["decision_clarity"] * 0.2
    )

    return {
        "pass": len(issues) == 0 and overall_score >= 0.7,
        "flagged": issues,
        "business_process_score": overall_score,
        **metrics
    }


def analyze_actor_clarity(doc) -> Dict[str, Any]:
    """
    Analyze clarity and specificity of actor role definitions in process text.

    Examines subject-verb relationships to identify process participants and
    evaluates whether actors are clearly defined business roles versus generic
    references. Critical for proper BPMN lane assignment and responsibility
    modeling in workflow diagrams.

    Args:
        doc: Processed spaCy document with dependency parsing and POS tagging

    Returns:
        Dictionary containing actor analysis results:
        - issues: List of specific actor clarity problems identified
        - clarity_score: Ratio of clear actors to total actors found
        - clear_actors: List of well-defined actor roles detected
        - generic_actors: List of vague or generic actor references
    """
    issues = []

    # Find all potential actors (subjects of action verbs)
    actors_found = set()
    generic_actors_found = set()
    unclear_actors = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # Find subject
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                for subj in subjects:
                    actor_text = subj.text.lower().strip()

                    if actor_text in GENERIC_ACTORS:
                        generic_actors_found.add(actor_text)
                        unclear_actors.append(f"Generic actor '{subj.text}' in: {sent.text[:50]}...")
                    elif any(role in actor_text for role in BUSINESS_ROLES):
                        actors_found.add(subj.text)
                    elif len(actor_text) > 2:  # Not just pronouns
                        actors_found.add(subj.text)

    # Calculate clarity score
    total_actors = len(actors_found) + len(generic_actors_found)
    clear_actors = len(actors_found)
    clarity_score = clear_actors / max(total_actors, 1)

    # Add issues for unclear actors
    issues.extend(unclear_actors)

    if clarity_score < 0.5:
        issues.append(f"Low actor clarity: only {clear_actors}/{total_actors} actors are clearly defined")

    return {
        "issues": issues,
        "clarity_score": clarity_score,
        "clear_actors": list(actors_found),
        "generic_actors": list(generic_actors_found)
    }


def analyze_action_clarity(doc) -> Dict[str, Any]:
    """
    Evaluate specificity and clarity of action verbs in business process descriptions.

    Analyzes root verbs to determine whether actions are specific business activities
    versus vague or generic verbs. Identifies passive voice constructions that
    obscure responsibility assignment. Essential for accurate task identification
    and BPMN activity modeling.

    Args:
        doc: Processed spaCy document with linguistic analysis

    Returns:
        Dictionary containing action analysis results:
        - issues: List of weak actions and passive voice problems
        - clarity_score: Ratio of strong business actions to total actions
        - strong_actions: Count of specific business activity verbs
        - weak_actions: Count of generic or vague action indicators
    """
    issues = []

    strong_actions = 0
    weak_actions = 0
    action_issues = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb_lemma = token.lemma_.lower()

                if verb_lemma in STRONG_ACTION_VERBS:
                    strong_actions += 1
                elif verb_lemma in WEAK_ACTION_INDICATORS:
                    weak_actions += 1
                    action_issues.append(f"Weak action verb '{token.text}' in: {sent.text[:50]}...")

                # Check for passive voice (often unclear in business processes)
                if any(child.dep_ == "nsubjpass" for child in token.children):
                    action_issues.append(f"Passive voice in: {sent.text[:50]}...")

    total_actions = strong_actions + weak_actions
    clarity_score = strong_actions / max(total_actions, 1)

    issues.extend(action_issues)

    if clarity_score < 0.4:
        issues.append(f"Low action clarity: {weak_actions} weak actions vs {strong_actions} clear actions")

    return {
        "issues": issues,
        "clarity_score": clarity_score,
        "strong_actions": strong_actions,
        "weak_actions": weak_actions
    }


def analyze_sequence_clarity(doc) -> Dict[str, Any]:
    """
    Assess clarity of activity sequencing and process flow indicators.

    Evaluates presence of temporal markers, sequence indicators, and structural
    elements that indicate proper workflow ordering. Critical for accurate
    sequence flow generation and process step connectivity in BPMN models.

    Args:
        doc: Processed spaCy document containing process description

    Returns:
        Dictionary containing sequence analysis results:
        - issues: List of sequence clarity problems identified
        - clarity_score: Combined score for sequence indicators and structure
        - sequence_indicators: Count of temporal markers found
        - structured_items: Count of numbered or bulleted list items
    """
    issues = []

    sequence_indicators_found = 0
    total_sentences = len(list(doc.sents))

    # Look for sequence indicators
    for token in doc:
        if token.text.lower() in SEQUENCE_INDICATORS:
            sequence_indicators_found += 1

    # Look for numbered or bulleted lists
    numbered_items = len(re.findall(r'^\s*\d+\.', doc.text, re.MULTILINE))
    bulleted_items = len(re.findall(r'^\s*[-•*]', doc.text, re.MULTILINE))

    structured_items = numbered_items + bulleted_items

    # Calculate clarity based on sequence indicators and structure
    sequence_density = sequence_indicators_found / max(total_sentences, 1)
    structure_score = min(1.0, structured_items / max(total_sentences * 0.3, 1))

    clarity_score = (sequence_density + structure_score) / 2

    if sequence_indicators_found == 0 and structured_items == 0:
        issues.append("No clear sequence indicators found - process flow may be unclear")

    if clarity_score < 0.3:
        issues.append(f"Low sequence clarity: limited flow indicators found")

    return {
        "issues": issues,
        "clarity_score": clarity_score,
        "sequence_indicators": sequence_indicators_found,
        "structured_items": structured_items
    }


def analyze_decision_clarity(doc) -> Dict[str, Any]:
    """
    Evaluate clarity of decision points and conditional logic in process descriptions.

    Identifies conditional statements and assesses whether decision outcomes
    are clearly specified. Essential for proper gateway modeling and branching
    logic implementation in BPMN workflows.

    Args:
        doc: Processed spaCy document with sentence segmentation

    Returns:
        Dictionary containing decision analysis results:
        - issues: List of unclear decision points and missing outcomes
        - clarity_score: Ratio of clear decisions to total decisions found
        - clear_decisions: Count of well-defined decision points
        - unclear_decisions: Count of decision points with ambiguous outcomes
    """
    issues = []

    decision_points = []
    unclear_decisions = []

    # Look for conditional statements
    for sent in doc.sents:
        sent_text = sent.text.lower()

        # Check for decision keywords
        has_decision_keyword = any(keyword in sent_text for keyword in DECISION_INDICATORS)

        if has_decision_keyword:
            # Check if the decision has clear outcomes
            has_clear_outcomes = (
                    "approve" in sent_text or "reject" in sent_text or
                    "yes" in sent_text or "no" in sent_text or
                    "accept" in sent_text or "decline" in sent_text
            )

            if has_clear_outcomes:
                decision_points.append(sent.text)
            else:
                unclear_decisions.append(f"Unclear decision outcome: {sent.text[:60]}...")

    total_decisions = len(decision_points) + len(unclear_decisions)
    clear_decisions = len(decision_points)

    clarity_score = clear_decisions / max(total_decisions, 1) if total_decisions > 0 else 1.0

    issues.extend(unclear_decisions)

    if total_decisions > 0 and clarity_score < 0.6:
        issues.append(f"Decision clarity issues: {len(unclear_decisions)} unclear decision points")

    return {
        "issues": issues,
        "clarity_score": clarity_score,
        "clear_decisions": len(decision_points),
        "unclear_decisions": len(unclear_decisions)
    }
