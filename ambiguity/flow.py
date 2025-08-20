import spacy
import re
from typing import Dict, Any, List, Tuple
from quality_gate.sqg import _sentence_has_actor_action

nlp = spacy.load("en_core_web_sm")

# Enhanced conditional keywords with business process focus
CONDITIONAL_KEYWORDS = {"if", "when", "once", "unless", "provided", "should", "in case"}

# Business process flow patterns
SEQUENCE_PATTERNS = {
    "start": ["start", "begin", "initiate", "commence", "launch"],
    "continuation": ["then", "next", "after", "following", "subsequently", "upon"],
    "parallel": ["simultaneously", "concurrently", "at the same time", "in parallel", "meanwhile"],
    "end": ["complete", "finish", "conclude", "end", "terminate", "close"],
    "handoff": ["transfer", "handover", "pass", "assign", "delegate", "forward"],
    "decision": ["decide", "determine", "evaluate", "assess", "review", "approve", "reject"]
}

# Gateway patterns for BPMN
GATEWAY_PATTERNS = {
    "exclusive": ["either", "or", "alternatively", "otherwise"],
    "inclusive": ["and", "also", "additionally", "furthermore"],
    "parallel": ["simultaneously", "at the same time", "in parallel"]
}


def check_flow_ambiguity(text: str) -> Dict[str, Any]:
    """
    Evaluate flow clarity for business process text.

    Finds conditional issues, missing connections, and unclear gateways, then
    computes a flow clarity score and BPMN flow readiness assessment.

    Args:
        text: Text to analyze.

    Returns:
        Dict with pass flag, flagged issues, ratios, detailed analyses, and readiness.
    """
    doc = nlp(text)

    issues = []
    flow_analysis = analyze_process_flow_patterns(doc)

    # Check conditional sentences for actor-action clarity
    conditional_issues = check_conditional_clarity(doc)
    issues.extend(conditional_issues["issues"])

    # Check for missing flow connections
    connection_issues = check_flow_connections(doc)
    issues.extend(connection_issues["issues"])

    # Check for gateway clarity
    gateway_issues = check_gateway_clarity(doc)
    issues.extend(gateway_issues["issues"])

    # Calculate overall flow clarity score
    total_checks = len(conditional_issues["conditional_sentences"]) + len(flow_analysis["sequence_breaks"]) + len(
        gateway_issues["decision_points"])
    total_issues = len(issues)

    clarity_score = max(0, 1 - (total_issues / max(total_checks, 1)))

    return {
        "pass": len(issues) == 0 and clarity_score >= 0.7,
        "flagged": issues,
        "ratio": total_issues / max(total_checks, 1),
        "flow_clarity_score": clarity_score,
        "flow_analysis": flow_analysis,
        "conditional_analysis": conditional_issues,
        "gateway_analysis": gateway_issues,
        "bpmn_flow_readiness": assess_bpmn_flow_readiness(flow_analysis, conditional_issues, gateway_issues)
    }


def analyze_process_flow_patterns(doc) -> Dict[str, Any]:
    """
    Scan sentences for flow indicators and sequence breaks.

    Counts sequence, parallel, decision, and handoff indicators and flags sentences
    that lack clear connection to the previous context.

    Args:
        doc: SpaCy Doc.

    Returns:
        Dict with indicator lists, sequence_breaks, and flow_density.
    """

    sequence_indicators = []
    parallel_indicators = []
    decision_indicators = []
    handoff_indicators = []
    sequence_breaks = []

    sentences = list(doc.sents)

    for i, sent in enumerate(sentences):
        sent_text = sent.text.lower()

        # Check for sequence patterns
        for pattern_type, keywords in SEQUENCE_PATTERNS.items():
            for keyword in keywords:
                if keyword in sent_text:
                    if pattern_type == "continuation":
                        sequence_indicators.append((i, keyword, sent.text))
                    elif pattern_type == "parallel":
                        parallel_indicators.append((i, keyword, sent.text))
                    elif pattern_type == "decision":
                        decision_indicators.append((i, keyword, sent.text))
                    elif pattern_type == "handoff":
                        handoff_indicators.append((i, keyword, sent.text))

        # Check for sequence breaks (sentences without clear connection to previous)
        if i > 0:
            has_connection = any(conn in sent_text for conn in
                                 SEQUENCE_PATTERNS["continuation"] + SEQUENCE_PATTERNS["parallel"])

            # Check if sentence starts with a clear actor or reference
            has_clear_start = (
                    any(sent_text.startswith(start) for start in ["the", "a", "an"]) or
                    re.match(r'^[A-Z][a-z]+\s+(team|manager|department)', sent_text) or
                    any(sent.text.startswith(pron) for pron in ["This", "That", "It", "They"])
            )

            if not has_connection and not has_clear_start and len(sent.text.strip()) > 20:
                sequence_breaks.append(f"Unclear connection: {sent.text[:60]}...")

    return {
        "sequence_indicators": sequence_indicators,
        "parallel_indicators": parallel_indicators,
        "decision_indicators": decision_indicators,
        "handoff_indicators": handoff_indicators,
        "sequence_breaks": sequence_breaks,
        "flow_density": len(sequence_indicators) / max(len(sentences), 1)
    }


def check_conditional_clarity(doc) -> Dict[str, Any]:
    """
    Evaluate the clarity of conditional statements.

    Identifies conditionals and checks for actor-action structure and clear outcomes.

    Args:
        doc: SpaCy Doc.

    Returns:
        Dict with conditional_sentences, issues, and a clarity_ratio.
    """
    conditional_sentences = []
    unclear_conditionals = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Check if sentence contains conditional keywords
        has_conditional = any(
            re.search(rf"\b{keyword}\b", sent_text, re.IGNORECASE)
            for keyword in CONDITIONAL_KEYWORDS
        )

        if has_conditional:
            conditional_sentences.append(sent_text)

            # Check if conditional has clear actor-action structure
            if not _sentence_has_actor_action(sent):
                unclear_conditionals.append(f"Unclear conditional: {sent_text[:60]}...")

            # Check if conditional has clear outcomes
            has_clear_outcomes = any(outcome in sent_text.lower() for outcome in
                                     ["then", "approve", "reject", "accept", "deny", "proceed", "stop"])

            if not has_clear_outcomes:
                unclear_conditionals.append(f"Missing outcomes: {sent_text[:60]}...")

    return {
        "conditional_sentences": conditional_sentences,
        "issues": unclear_conditionals,
        "clarity_ratio": len(unclear_conditionals) / max(len(conditional_sentences), 1)
    }


def check_flow_connections(doc) -> Dict[str, Any]:
    """
    Check whether consecutive sentences are connected.

    Flags likely missing connections using explicit connectors or implicit continuity.

    Args:
        doc: SpaCy Doc.

    Returns:
        Dict with issues and a connection_ratio.
    """
    sentences = list(doc.sents)
    connection_issues = []

    for i in range(1, len(sentences)):
        current_sent = sentences[i].text.lower()
        prev_sent = sentences[i - 1].text.lower()

        # Check if current sentence has clear connection to previous
        has_explicit_connection = any(conn in current_sent for conn in
                                      ["then", "next", "after", "upon", "once", "when", "subsequently"])

        has_implicit_connection = (
            # Starts with same actor/subject
                any(current_sent.startswith(ref) for ref in ["this", "that", "it", "they"]) or
                # Continues same topic (basic check)
                len(set(current_sent.split()[:3]) & set(prev_sent.split()[-5:])) > 0
        )

        if not has_explicit_connection and not has_implicit_connection:
            if len(sentences[i].text.strip()) > 30:  # Ignore very short sentences
                connection_issues.append(f"Missing connection between steps: {sentences[i].text[:50]}...")

    return {
        "issues": connection_issues,
        "connection_ratio": len(connection_issues) / max(len(sentences) - 1, 1)
    }


def check_gateway_clarity(doc) -> Dict[str, Any]:
    """
    Assess decision point clarity for BPMN gateway creation.

    Finds sentences indicating decisions and evaluates gateway type and branching.

    Args:
        doc: SpaCy Doc.

    Returns:
        Dict with decision_points, issues, and gateway_clarity.
    """

    decision_points = []
    unclear_gateways = []

    for sent in doc.sents:
        sent_text = sent.text.lower()

        # Look for decision indicators
        has_decision = any(decision in sent_text for decision in
                           SEQUENCE_PATTERNS["decision"] + ["if", "when", "choose", "select"])

        if has_decision:
            decision_points.append(sent.text)

            # Check gateway type clarity
            gateway_type = determine_gateway_type(sent.text)

            if gateway_type == "unclear":
                unclear_gateways.append(f"Unclear gateway type: {sent.text[:60]}...")

            # Check for clear branching outcomes
            has_branches = any(branch in sent_text for branch in
                               ["or", "either", "alternative", "otherwise", "else"])

            if not has_branches and "if" in sent_text:
                unclear_gateways.append(f"Missing else branch: {sent.text[:60]}...")

    return {
        "decision_points": decision_points,
        "issues": unclear_gateways,
        "gateway_clarity": len(unclear_gateways) / max(len(decision_points), 1)
    }


def determine_gateway_type(sentence: str) -> str:
    """
    Infer BPMN gateway type from sentence text.

    Args:
        sentence: Sentence text.

    Returns:
        String gateway type: 'exclusive', 'inclusive', 'parallel', or 'unclear'.
    """
    sentence_lower = sentence.lower()

    # Exclusive gateway patterns
    if any(pattern in sentence_lower for pattern in GATEWAY_PATTERNS["exclusive"]):
        return "exclusive"

    # Inclusive gateway patterns
    if any(pattern in sentence_lower for pattern in GATEWAY_PATTERNS["inclusive"]):
        return "inclusive"

    # Parallel gateway patterns
    if any(pattern in sentence_lower for pattern in GATEWAY_PATTERNS["parallel"]):
        return "parallel"

    # Check for simple if-then patterns
    if "if" in sentence_lower and (
            "then" in sentence_lower or "approve" in sentence_lower or "reject" in sentence_lower):
        return "exclusive"

    return "unclear"


def assess_bpmn_flow_readiness(flow_analysis: Dict, conditional_analysis: Dict, gateway_analysis: Dict) -> Dict[
    str, Any]:
    """
    Summarize BPMN flow readiness from analyses.

    Combines sequence density, gateway clarity, connection ratio, and conditional
    clarity into a readiness score and recommendations.

    Args:
        flow_analysis: Dict from analyze_process_flow_patterns.
        conditional_analysis: Dict from check_conditional_clarity.
        gateway_analysis: Dict from check_gateway_clarity.

    Returns:
        Dict with readiness_score, readiness_level, issues, recommendations, and
        suggested BPMN elements.
    """

    readiness_score = 0
    issues = []
    recommendations = []

    # Sequence flow readiness
    if flow_analysis["flow_density"] > 0.3:
        readiness_score += 0.3
    else:
        issues.append("Low sequence flow density - may result in disconnected BPMN elements")
        recommendations.append("Add more sequence indicators (then, next, after) to clarify process flow")

    # Decision point readiness
    if gateway_analysis["gateway_clarity"] < 0.3:
        readiness_score += 0.3
    else:
        issues.append("Unclear decision points - may result in ambiguous gateways")
        recommendations.append("Clarify decision outcomes and branching logic")

    # Connection readiness
    connection_ratio = len(flow_analysis["sequence_breaks"]) / 10  # Normalize
    if connection_ratio < 0.2:
        readiness_score += 0.2
    else:
        issues.append("Many disconnected sentences - may result in fragmented BPMN")
        recommendations.append("Add connecting words to link process steps")

    # Conditional clarity
    if conditional_analysis["clarity_ratio"] < 0.3:
        readiness_score += 0.2
    else:
        issues.append("Unclear conditional statements")
        recommendations.append("Clarify if-then logic with specific actors and outcomes")

    readiness_level = "high" if readiness_score >= 0.8 else "medium" if readiness_score >= 0.5 else "low"

    return {
        "readiness_score": readiness_score,
        "readiness_level": readiness_level,
        "issues": issues,
        "recommendations": recommendations,
        "bpmn_elements_suggested": suggest_bpmn_elements(flow_analysis, gateway_analysis)
    }


def suggest_bpmn_elements(flow_analysis: Dict, gateway_analysis: Dict) -> Dict[str, int]:
    """
    Suggest counts of BPMN elements based on detected patterns.

    Args:
        flow_analysis: Dict with flow indicators.
        gateway_analysis: Dict with decision points.

    Returns:
        Dict mapping BPMN element names to integer suggestions.
    """

    suggestions = {
        "sequence_flows": len(flow_analysis["sequence_indicators"]),
        "parallel_gateways": len(flow_analysis["parallel_indicators"]) * 2,  # Open + close
        "exclusive_gateways": len(gateway_analysis["decision_points"]),
        "user_tasks": max(3, len(flow_analysis["sequence_indicators"])),  # Estimate
        "service_tasks": len(flow_analysis["handoff_indicators"])
    }

    return suggestions