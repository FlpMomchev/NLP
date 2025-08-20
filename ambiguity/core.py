from ambiguity.lexical import check_lexical_ambiguity
from ambiguity.role import check_pronoun_ambiguity
from ambiguity.coref import check_coref_gaps
from ambiguity.flow import check_flow_ambiguity
from ambiguity.business_process import check_business_process_clarity
from segmentation.splitter import split_into_sections


def run_ambiguity_checks_on_chunk(chunk: str) -> dict:
    """
    Run all ambiguity checks on a single text chunk.

    Analyzes lexical vagueness, pronoun usage, coreference gaps, and flow clarity,
    plus business process clarity indicators. Returns flagged issues and metrics,
    including a weighted ambiguity score and BPMN readiness value.

    Args:
        chunk: Text segment to analyze.

    Returns:
        Dict with keys:
        - issues: category-to-list of flagged items.
        - metrics: computed booleans, ratios, and composite scores.
    """
    issues = {}
    metrics = {}

    # Lexical (vague words)
    lex = check_lexical_ambiguity(chunk)
    issues["lexical"] = lex["flagged"]
    metrics["lexical_ok"] = lex["pass"]
    metrics["lexical_ratio"] = lex.get("ratio", 0.0)

    # Pronoun
    pro = check_pronoun_ambiguity(chunk)
    issues["pronoun"] = pro["flagged"]
    metrics["pronoun_ok"] = pro["pass"]

    # Coref
    cof = check_coref_gaps(chunk)
    issues["coref"] = cof["flagged"]
    metrics["coref_ok"] = cof["pass"]

    # Flow
    flo = check_flow_ambiguity(chunk)
    issues["flow"] = flo["flagged"]
    metrics["flow_ok"] = flo["pass"]

    # Business Process Clarity
    bp = check_business_process_clarity(chunk)
    issues["business_process"] = bp["flagged"]
    metrics["business_process_ok"] = bp["pass"]
    metrics["actor_clarity"] = bp.get("actor_clarity", 0.0)
    metrics["action_clarity"] = bp.get("action_clarity", 0.0)
    metrics["sequence_clarity"] = bp.get("sequence_clarity", 0.0)

    # Enhanced scoring that weights business process issues more heavily
    critical_flags = len(issues["business_process"]) * 2  # Weight BP issues 2x
    other_flags = sum(len(lst) for key, lst in issues.items() if key != "business_process")
    total_flags = critical_flags + other_flags
    total_checks = len(issues) + 1  # +1 for weighted BP check

    metrics["ambiguity_score"] = total_flags / (total_checks + 1e-6)

    # BPMN readiness score
    metrics["bpmn_readiness"] = calculate_bpmn_readiness_score(metrics)

    return {
        "issues": issues,
        "metrics": metrics
    }


def calculate_bpmn_readiness_score(metrics: dict) -> float:
    """
    Compute an overall BPMN readiness score from metrics.

    Weights business process clarity highest, then actor/action clarity and flow,
    with smaller weights for coreference, pronouns, and lexical clarity.

    Args:
        metrics: Dictionary of metric values produced by checks.

    Returns:
        A float in [0.0, 1.0] representing readiness.
    """
    weights = {
        "business_process_ok": 0.4,  # Most important
        "actor_clarity": 0.2,
        "action_clarity": 0.2,
        "flow_ok": 0.1,
        "coref_ok": 0.05,
        "pronoun_ok": 0.03,
        "lexical_ok": 0.02
    }

    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            if isinstance(metrics[metric], bool):
                score += weight if metrics[metric] else 0
            else:
                score += weight * metrics[metric]

    return min(1.0, max(0.0, score))


def run_enhanced_ambiguity_analysis(text: str) -> dict:
    """
    Analyze a full document and summarize BPMN readiness by section.

    Splits text into sections, runs ambiguity checks per section, assigns readiness
    levels, and aggregates overall metrics and recommendations.

    Args:
        text: Full business process description.

    Returns:
        Dict with section_analyses, overall_metrics, and summary_recommendations.
    """

    sections = split_into_sections(text)
    section_analyses = []

    overall_metrics = {
        "total_sections": len(sections),
        "bpmn_ready_sections": 0,
        "needs_improvement_sections": 0,
        "problematic_sections": 0
    }

    for i, section in enumerate(sections):
        analysis = run_ambiguity_checks_on_chunk(section["body"])

        bpmn_readiness = analysis["metrics"]["bpmn_readiness"]

        if bpmn_readiness >= 0.7:
            overall_metrics["bpmn_ready_sections"] += 1
            readiness_level = "ready"
        elif bpmn_readiness >= 0.4:
            overall_metrics["needs_improvement_sections"] += 1
            readiness_level = "needs_improvement"
        else:
            overall_metrics["problematic_sections"] += 1
            readiness_level = "problematic"

        section_analysis = {
            "section_id": i + 1,
            "section_header": section["header"],
            "bpmn_readiness": bpmn_readiness,
            "readiness_level": readiness_level,
            "analysis": analysis,
            "recommendations": generate_recommendations(analysis)
        }

        section_analyses.append(section_analysis)

    overall_metrics["overall_bpmn_readiness"] = sum(
        s["bpmn_readiness"] for s in section_analyses
    ) / len(section_analyses)

    return {
        "section_analyses": section_analyses,
        "overall_metrics": overall_metrics,
        "summary_recommendations": generate_overall_recommendations(section_analyses)
    }


def generate_recommendations(analysis: dict) -> list:
    """
    Create targeted recommendations from a single chunk analysis.

    Uses issues and metrics to suggest specific improvements for actors, actions,
    pronouns, and vague terms.

    Args:
        analysis: Output from run_ambiguity_checks_on_chunk.

    Returns:
        List of recommendation objects with priority, category, issue, recommendation.
    """
    recommendations = []
    issues = analysis["issues"]
    metrics = analysis["metrics"]

    if not metrics.get("business_process_ok", True):
        recommendations.append({
            "priority": "high",
            "category": "business_process",
            "issue": "Unclear business process structure",
            "recommendation": "Clearly identify actors, actions, and decision points. Use active voice and specific role names."
        })

    if metrics.get("actor_clarity", 1.0) < 0.6:
        recommendations.append({
            "priority": "high",
            "category": "actors",
            "issue": "Unclear or missing actors",
            "recommendation": "Specify who performs each action using concrete role names (e.g., 'Product Manager' instead of 'they')."
        })

    if metrics.get("action_clarity", 1.0) < 0.6:
        recommendations.append({
            "priority": "medium",
            "category": "actions",
            "issue": "Vague action descriptions",
            "recommendation": "Use specific action verbs and describe what exactly is being done."
        })

    if len(issues.get("pronoun", [])) > 3:
        recommendations.append({
            "priority": "medium",
            "category": "pronouns",
            "issue": f"Too many ambiguous pronouns: {issues['pronoun'][:3]}",
            "recommendation": "Replace pronouns with specific nouns or role names for clarity."
        })

    if len(issues.get("lexical", [])) > 2:
        recommendations.append({
            "priority": "low",
            "category": "lexical",
            "issue": f"Vague terms found: {issues['lexical'][:3]}",
            "recommendation": "Replace vague terms with specific quantities, timeframes, or conditions."
        })

    return recommendations


def generate_overall_recommendations(section_analyses: list) -> list:
    """
    Create high-level recommendations across sections.

    Highlights widespread ambiguity categories and suggests restructuring when
    too few sections are ready.

    Args:
        section_analyses: List of section-level analyses.

    Returns:
        List of recommendation objects.
    """
    recommendations = []

    ready_count = sum(1 for s in section_analyses if s["readiness_level"] == "ready")
    total_count = len(section_analyses)

    if ready_count / total_count < 0.5:
        recommendations.append({
            "priority": "critical",
            "issue": f"Only {ready_count}/{total_count} sections are BPMN-ready",
            "recommendation": "Consider restructuring the document with clearer process flows, specific actors, and explicit decision points before attempting BPMN generation."
        })

    # Aggregate issues across sections
    all_issues = {}
    for section in section_analyses:
        for category, issue_list in section["analysis"]["issues"].items():
            if category not in all_issues:
                all_issues[category] = []
            all_issues[category].extend(issue_list)

    for category, issues in all_issues.items():
        if len(issues) > total_count * 0.3:  # If more than 30% of sections have this issue
            recommendations.append({
                "priority": "high",
                "issue": f"Widespread {category} ambiguity across document",
                "recommendation": f"Perform document-wide review to address {category} clarity issues."
            })

    return recommendations