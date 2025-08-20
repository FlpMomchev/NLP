import spacy
import nltk
from nltk.corpus import wordnet
from typing import Dict, Any, Set, List

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

# Download wordnet if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Business process specific vague terms (more focused than general vague terms)
BUSINESS_PROCESS_VAGUE_TERMS = {
    # Quantifiers
    "some", "several", "many", "few", "various", "multiple", "numerous",

    # Time indicators
    "soon", "later", "eventually", "sometime", "recently", "quickly",

    # Uncertainty markers
    "might", "could", "should", "may", "possibly", "potentially", "likely",

    # Size/scope indicators
    "large", "small", "big", "little", "significant", "minor", "major",

    # Quality indicators
    "good", "bad", "appropriate", "suitable", "relevant", "important",

    # Process indicators
    "properly", "correctly", "effectively", "efficiently", "appropriately"
}

# Less critical vague terms that are sometimes acceptable in business context
ACCEPTABLE_VAGUE_TERMS = {
    "about", "approximately", "around", "roughly", "generally", "typically",
    "usually", "normally", "often", "sometimes", "occasionally"
}

# Terms that are particularly problematic for BPMN generation
CRITICAL_VAGUE_TERMS = {
    "several", "some", "various", "might", "could", "should", "appropriate",
    "relevant", "suitable", "properly", "effectively"
}


def check_lexical_ambiguity(text: str) -> Dict[str, Any]:
    """
    Identify vague language and missing specification details.

    Classifies vague terms by severity and detects missing quantities, timeframes,
    and criteria. Computes a severity-weighted score and a pass flag based on
    critical issues.

    Args:
        text: Text to analyze.

    Returns:
        Dict with pass, flagged terms, ratios, business_process_issues, and impact.
    """
    doc = nlp(text)

    # Collect different types of vague terms
    critical_vague = []
    moderate_vague = []
    minor_vague = []

    # Business-specific issues
    missing_quantities = []
    missing_timeframes = []
    missing_criteria = []

    for token in doc:
        token_lower = token.text.lower()

        # Categorize vague terms by severity
        if token_lower in CRITICAL_VAGUE_TERMS:
            critical_vague.append(token.text)
        elif token_lower in BUSINESS_PROCESS_VAGUE_TERMS:
            moderate_vague.append(token.text)
        elif token_lower in ACCEPTABLE_VAGUE_TERMS:
            minor_vague.append(token.text)

    # Look for missing quantification patterns
    missing_quantities.extend(find_missing_quantities(doc))
    missing_timeframes.extend(find_missing_timeframes(doc))
    missing_criteria.extend(find_missing_criteria(doc))

    # Calculate severity-weighted score
    total_tokens = len([t for t in doc if t.is_alpha])
    critical_weight = len(critical_vague) * 3
    moderate_weight = len(moderate_vague) * 2
    minor_weight = len(minor_vague) * 1

    severity_score = (critical_weight + moderate_weight + minor_weight) / max(total_tokens, 1)

    # Determine pass/fail based on critical issues
    has_critical_issues = len(critical_vague) > 0 or len(missing_quantities) > 0

    # Combine all flagged terms
    all_flagged = critical_vague + moderate_vague + minor_vague

    return {
        "pass": not has_critical_issues and severity_score < 0.1,
        "flagged": all_flagged,
        "ratio": severity_score,
        "severity_breakdown": {
            "critical": critical_vague,
            "moderate": moderate_vague,
            "minor": minor_vague
        },
        "business_process_issues": {
            "missing_quantities": missing_quantities,
            "missing_timeframes": missing_timeframes,
            "missing_criteria": missing_criteria
        },
        "bpmn_impact": assess_bpmn_impact(critical_vague, moderate_vague, missing_quantities, missing_timeframes)
    }


def find_missing_quantities(doc) -> List[str]:
    """
    Find cases where quantities should be specified but are not.

    Args:
        doc: SpaCy Doc.

    Returns:
        List of issue descriptions.
    """
    issues = []

    quantity_patterns = [
        (r'\bsome\s+(?:number|amount|quantity)\b', "Unspecified quantity"),
        (r'\bseveral\s+(?:items|documents|steps)\b', "Unspecified count"),
        (r'\bmany\s+(?:people|users|cases)\b', "Unspecified number"),
        (r'\ba\s+few\s+(?:days|hours|minutes)\b', "Unspecified timeframe"),
    ]

    text = doc.text.lower()
    for pattern, description in quantity_patterns:
        import re
        if re.search(pattern, text):
            issues.append(description)

    return issues


def find_missing_timeframes(doc) -> List[str]:
    """
    Find vague timeframe expressions.

    Args:
        doc: SpaCy Doc.

    Returns:
        List of issue descriptions.
    """
    issues = []

    timeframe_patterns = [
        (r'\bsoon\b', "Vague timeframe: 'soon'"),
        (r'\blater\b', "Vague timeframe: 'later'"),
        (r'\beventually\b', "Vague timeframe: 'eventually'"),
        (r'\bquickly\b', "Vague timeframe: 'quickly'"),
        (r'\bin\s+a\s+timely\s+manner\b', "Vague timeframe: 'in a timely manner'"),
    ]

    text = doc.text.lower()
    for pattern, description in timeframe_patterns:
        import re
        if re.search(pattern, text):
            issues.append(description)

    return issues


def find_missing_criteria(doc) -> List[str]:
    """
    Find vague criteria or conditions in decisions.

    Args:
        doc: SpaCy Doc.

    Returns:
        List of issue descriptions.
    """
    issues = []

    criteria_patterns = [
        (r'\bappropriate\b', "Unspecified criteria: 'appropriate'"),
        (r'\bsuitable\b', "Unspecified criteria: 'suitable'"),
        (r'\brelevant\b', "Unspecified criteria: 'relevant'"),
        (r'\bif\s+necessary\b', "Unspecified condition: 'if necessary'"),
        (r'\bas\s+needed\b', "Unspecified condition: 'as needed'"),
    ]

    text = doc.text.lower()
    for pattern, description in criteria_patterns:
        import re
        if re.search(pattern, text):
            issues.append(description)

    return issues


def assess_bpmn_impact(critical_vague: List[str], moderate_vague: List[str],
                       missing_quantities: List[str], missing_timeframes: List[str]) -> Dict[str, Any]:
    """
    Estimate the impact of vague language on BPMN modeling.

    Combines counts of critical terms and missing details into an impact score
    and level, with specific issues and recommendations.

    Args:
        critical_vague: List of critical vague terms.
        moderate_vague: List of moderate vague terms.
        missing_quantities: List of missing quantity issues.
        missing_timeframes: List of missing timeframe issues.

    Returns:
        Dict with impact_score, impact_level, specific_issues, recommendations.
    """

    impact_score = 0
    issues = []

    # Critical vague terms have high impact
    if critical_vague:
        impact_score += len(critical_vague) * 0.3
        issues.append(f"Critical vague terms will make BPMN elements unclear: {critical_vague[:3]}")

    # Missing quantities affect gateway conditions
    if missing_quantities:
        impact_score += len(missing_quantities) * 0.25
        issues.append("Missing quantities will make gateway conditions unclear")

    # Missing timeframes affect timer events and deadlines
    if missing_timeframes:
        impact_score += len(missing_timeframes) * 0.2
        issues.append("Missing timeframes will make process timing unclear")

    # Moderate vague terms have medium impact
    if len(moderate_vague) > 3:
        impact_score += 0.15
        issues.append(f"Multiple vague terms may reduce BPMN clarity: {moderate_vague[:3]}")

    impact_level = "low"
    if impact_score > 0.7:
        impact_level = "high"
    elif impact_score > 0.3:
        impact_level = "medium"

    return {
        "impact_score": min(1.0, impact_score),
        "impact_level": impact_level,
        "specific_issues": issues,
        "recommendations": generate_lexical_recommendations(critical_vague, missing_quantities, missing_timeframes)
    }


def generate_lexical_recommendations(critical_vague: List[str], missing_quantities: List[str],
                                     missing_timeframes: List[str]) -> List[str]:
    """
    Create recommendations that reduce lexical ambiguity.

    Args:
        critical_vague: List of critical vague terms.
        missing_quantities: List of missing quantity issues.
        missing_timeframes: List of missing timeframe issues.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if critical_vague:
        recommendations.append(
            f"Replace vague terms ({', '.join(set(critical_vague[:3]))}) with specific values or criteria")

    if missing_quantities:
        recommendations.append("Specify exact quantities or ranges instead of 'some', 'several', 'many'")

    if missing_timeframes:
        recommendations.append("Specify exact timeframes (e.g., '2 business days') instead of 'soon', 'quickly'")

    if not recommendations:
        recommendations.append("Lexical clarity is good for BPMN generation")

    return recommendations