import json
import re

# Import specialized modules
from ingest.reader import read_file
from ingest.cleaner import clean_text, analyze_cleaning_impact
from quality_gate.sqg import run_quality_check
from segmentation.sentence_workflow_extractor import SentenceWorkflowExtractor
from segmentation.splitter import split_into_sections
from postprocess.actor_consolidator import apply_actor_consolidation
from ambiguity.core import run_enhanced_ambiguity_analysis
from json_gen.prompt_builder import build_enhanced_llm_prompt


def process_bpmn_document(text, process_name="Business Process"):
    """
    Main function to process a document and extract BPMN workflow information.

    Orchestrates the complete pipeline from text cleaning through final output generation.
    Processes the document through seven distinct phases: cleaning, quality checking,
    workflow extraction, actor consolidation, sequential task organization, ambiguity
    analysis, and output generation.

    Args:
        text (str): The raw document text to be processed
        process_name (str, optional): Name identifier for the business process.
                                    Defaults to "Business Process"

    Returns:
        dict: Processing results containing:
            - success (bool): Whether processing completed successfully
            - process_name (str): The process identifier used
            - llm_prompt (str): Clean prompt ready for LLM generation
            - analysis (dict): Comprehensive analysis data including metrics,
                             workflow steps, quality assessment, and debugging information

    Raises:
        Exception: Any processing errors are caught and returned in the analysis
                  data with success=False
    """
    try:
        # Step 1: Clean the text
        cleaned_text = clean_text(text)
        cleaning_stats = analyze_cleaning_impact(text, cleaned_text)

        # Step 2: Check quality
        quality_result = run_quality_check(cleaned_text)
        quality_warnings = [] if quality_result["overall_pass"] else quality_result["feedback"]

        # Step 3: Extract workflows
        all_tasks = extract_all_workflows(cleaned_text)

        # Step 4: Fix actor names
        all_tasks, actor_info = fix_actor_names(all_tasks)

        # Step 5: Make tasks sequential
        sequential_tasks = make_tasks_sequential(all_tasks)

        # Step 6: Check for problems
        ambiguity_result = run_enhanced_ambiguity_analysis(cleaned_text)

        # Step 7: Create final outputs
        success = True
        llm_prompt = create_llm_prompt(sequential_tasks, actor_info)
        analysis = create_analysis_data(sequential_tasks, actor_info, process_name,
                                        quality_warnings, ambiguity_result, cleaning_stats)

        return {
            "success": success,
            "process_name": process_name,
            "llm_prompt": llm_prompt,
            "analysis": analysis
        }

    except Exception as e:
        return {
            "success": False,
            "process_name": process_name,
            "llm_prompt": "",
            "analysis": {"error": str(e), "workflow_summary": {}, "quality_metrics": {}}
        }


def extract_all_workflows(text):
    """
    Extract workflow tasks from all sections of the document.

    Splits the document into logical sections and extracts workflow elements
    from each section that contains sufficient content. Uses the sentence-level
    workflow extractor to identify tasks, gateways, events, and actors from
    natural language text.

    Args:
        text (str): Cleaned document text to extract workflows from

    Returns:
        list: List of task dictionaries, each containing:
            - Basic task information (name, actor, type)
            - Section metadata (section_id, section_name)
            - Workflow-specific attributes based on task type
    """
    extractor = SentenceWorkflowExtractor()
    sections = split_into_sections(text)
    all_tasks = []

    for i, section in enumerate(sections):
        if len(section["body"].split()) < 30:
            continue

        section_tasks = extractor.extract_sentence_workflows(
            section["body"], section["header"]
        )

        # Add section info to each task
        for task in section_tasks:
            task['section_id'] = i + 1
            task['section_name'] = section["header"]

        all_tasks.extend(section_tasks)

    return all_tasks


def fix_actor_names(tasks):
    """
    Consolidate and standardize actor names across all workflow tasks.

    Analyzes actor name variations and consolidates similar names into canonical
    forms. This reduces redundancy and improves the quality of the final BPMN
    model by ensuring consistent actor representation throughout the workflow.

    Args:
        tasks (list): List of task dictionaries containing actor information

    Returns:
        tuple: A tuple containing:
            - tasks (list): Updated tasks with consolidated actor names
            - consolidation_info (dict): Detailed information about the consolidation
                                       process including mappings and statistics
    """
    # Get all actor names
    all_actors = [task['actor'] for task in tasks]

    # Use the consolidation function
    dummy_blocks = [{"primary_actors": all_actors, "steps": []}]
    consolidated_blocks, consolidation_info = apply_actor_consolidation(dummy_blocks)

    # Get the mapping of old names to new names
    mapping = consolidation_info.get("actor_resolution", {}).get("variation_to_canonical", {})

    # Update all task actors
    for task in tasks:
        task['actor'] = mapping.get(task['actor'], task['actor'])

    return tasks, consolidation_info


def make_tasks_sequential(tasks):
    """
    Assign sequential IDs to tasks and establish proper workflow connections.

    Converts the extracted task list into a properly sequenced workflow with
    unique identifiers and explicit connections between tasks. Handles different
    task types (tasks, gateways, events) with appropriate ID naming conventions
    and establishes the flow sequence for BPMN generation.

    Args:
        tasks (list): List of workflow tasks with basic information

    Returns:
        list: List of sequential tasks with:
            - Unique IDs following BPMN conventions (T1, G1, TM1, etc.)
            - Proper 'next' references establishing workflow sequence
            - Type-specific attributes preserved and enhanced
            - Confidence and connection metadata
    """
    sequential_tasks = []
    task_counter = 1
    gateway_counter = 1
    timer_counter = 1
    message_counter = 1

    # Give each task a proper ID
    for i, task in enumerate(tasks):
        if task['type'] == 'task':
            task_id = f"T{task_counter}"
            task_counter += 1
        elif task['type'] == 'gateway':
            task_id = f"G{gateway_counter}"
            gateway_counter += 1
        elif task['type'] == 'timer_event':
            task_id = f"TM{timer_counter}"
            timer_counter += 1
        elif task['type'] == 'message_event':
            task_id = f"M{message_counter}"
            message_counter += 1
        else:
            task_id = f"E{task_counter}"
            task_counter += 1

        # Create the output task
        output_task = {
            "id": task_id,
            "name": task['name'],
            "actor": task['actor'],
            "type": task['type']
        }

        # Add special attributes based on type
        if task['type'] == 'gateway' and 'branches' in task:
            output_task['branches'] = task['branches']
        elif task['type'] == 'timer_event':
            output_task['timing'] = task.get('timing', {})
        elif task['type'] == 'message_event':
            output_task['direction'] = task.get('direction', 'intermediate')

        # Connect to next task
        if i + 1 < len(tasks):
            next_task_index = i + 1
            # We'll set the next ID after we know what it is
            output_task['next'] = f"TEMP_{next_task_index}"
        else:
            output_task['next'] = 'END'

        # Add confidence info
        output_task['link_confidence'] = task.get('link_confidence', 0.5)
        output_task['connection_type'] = task.get('connection_type', 'document_order')

        sequential_tasks.append(output_task)

    # Fix the 'next' references now that we have all IDs
    for i, task in enumerate(sequential_tasks):
        if task['next'].startswith('TEMP_'):
            next_index = int(task['next'].split('_')[1])
            if next_index < len(sequential_tasks):
                task['next'] = sequential_tasks[next_index]['id']
            else:
                task['next'] = 'END'

    return sequential_tasks


def cluster_actors_into_lanes(actors, max_lanes=8):
    """
    Group similar actors into swimlanes for optimal BPMN diagram organization.

    Analyzes actor names and groups related actors into logical swimlanes based
    on domain similarity and keyword analysis. This reduces visual complexity
    in the final BPMN diagram and improves readability by organizing related
    actors into coherent lanes.

    Args:
        actors (list): List of unique actor names from the workflow
        max_lanes (int, optional): Maximum number of swimlanes to create.
                                  Defaults to 8

    Returns:
        dict: Lane clustering information containing:
            - clusters (dict): Mapping of lane names to actor lists
            - suggested_merges (list): Recommendations for actor groupings
            - reduction (float): Percentage reduction in number of lanes
            - original_count (int): Original number of actors
            - clustered_count (int): Final number of lanes created
    """
    if len(actors) <= max_lanes:
        clusters = {actor: [actor] for actor in actors}
        return {
            "clusters": clusters,
            "suggested_merges": [],
            "reduction": 0,
            "original_count": len(actors),
            "clustered_count": len(clusters)
        }

    # Grouping based on common words
    actor_keywords = {}
    for actor in actors:
        words = re.findall(r'\b[A-Za-z]+\b', actor.lower())
        keywords = [w for w in words if len(w) > 2 and w not in {'the', 'and', 'for', 'with', 'service', 'team'}]
        actor_keywords[actor] = set(keywords)

    # Domain-based clustering
    domain_clusters = {
        "Customer Team": ["customer", "client", "user", "applicant"],
        "Banking Team": ["banking", "core", "account", "transaction"],
        "Risk Team": ["risk", "fraud", "security", "compliance"],
        "Digital Team": ["digital", "online", "web", "portal", "oidc"],
        "KYC Team": ["kyc", "verification", "identity", "due diligence"],
        "Operations Team": ["processing", "monitoring", "validation", "operations"],
        "External Team": ["external", "vendor", "third party"],
        "Notification Team": ["message", "notification", "sms", "email", "alert"]
    }

    clusters = {}
    used_actors = set()

    # Group actors by domain
    for cluster_name, domain_words in domain_clusters.items():
        matching_actors = []
        for actor in actors:
            if actor in used_actors:
                continue
            actor_lower = actor.lower()
            if any(word in actor_lower for word in domain_words):
                matching_actors.append(actor)
                used_actors.add(actor)

        if matching_actors:
            clusters[cluster_name] = matching_actors

    # Handle remaining actors
    remaining_actors = [a for a in actors if a not in used_actors]
    for actor in remaining_actors:
        if len(clusters) < max_lanes:
            clusters[f"{actor} Team"] = [actor]
        else:
            # Add to biggest cluster
            biggest_cluster = max(clusters.keys(), key=lambda k: len(clusters[k]))
            clusters[biggest_cluster].append(actor)

    # Create merge suggestions
    suggested_merges = []
    for cluster_name, cluster_actors in clusters.items():
        if len(cluster_actors) > 1:
            suggested_merges.append({
                "cluster_name": cluster_name,
                "actors": cluster_actors,
                "reason": f"Domain similarity: {cluster_name.split()[0].lower()}"
            })

    reduction_percentage = round((1 - len(clusters) / len(actors)) * 100, 1)

    return {
        "clusters": clusters,
        "suggested_merges": suggested_merges,
        "reduction": reduction_percentage,
        "original_count": len(actors),
        "clustered_count": len(clusters)
    }


def create_llm_prompt(sequential_tasks, actor_info):
    """
    Generate a clean, structured prompt ready for LLM-based BPMN generation.

    Consolidates all workflow information into a comprehensive prompt that can
    be used by large language models to generate BPMN XML or other workflow
    representations. Includes process summaries, detailed steps, actor information,
    and lane clustering recommendations.

    Args:
        sequential_tasks (list): Ordered list of workflow tasks with IDs and connections
        actor_info (dict): Actor consolidation information and mappings

    Returns:
        str: Formatted prompt string containing all necessary information for
             BPMN generation, optimized for LLM processing
    """
    consolidated_actors = list(set(task['actor'] for task in sequential_tasks))
    lane_clustering = cluster_actors_into_lanes(consolidated_actors)

    # Create process summary
    process_summary = create_process_summary(sequential_tasks)
    detailed_steps = create_detailed_steps(sequential_tasks)

    enhanced_output = {
        "process_summary": process_summary,
        "detailed_steps": detailed_steps,
        "granular_tasks": sequential_tasks,
        "actor_consolidation": actor_info,
        "lane_clustering": lane_clustering,
        "metrics": {
            "granularity_level": "sentence",
            "total_actors": len(consolidated_actors),
            "recommended_lanes": len(lane_clustering["clusters"])
        }
    }

    return build_enhanced_llm_prompt(enhanced_output)


def create_process_summary(tasks):
    """
    Generate a high-level summary of the workflow organized by document sections.

    Groups tasks by their original document sections and creates summary blocks
    that provide an overview of the process structure. Each block contains
    information about the main goal, involved actors, complexity, and presence
    of decision points.

    Args:
        tasks (list): Sequential tasks with section metadata

    Returns:
        list: List of summary blocks, each containing:
            - block_id (int): Sequential block identifier
            - phase_or_main_goal (str): Section name or main purpose
            - primary_actors (list): Actors involved in this section
            - contains_gateway (bool): Whether section has decision points
            - complexity_score (int): Number of tasks as complexity indicator
    """
    sections = {}
    for task in tasks:
        section_name = task.get('section_name', 'Unknown')
        if section_name not in sections:
            sections[section_name] = {
                "tasks": [],
                "actors": set(),
                "has_gateways": False
            }
        sections[section_name]["tasks"].append(task)
        sections[section_name]["actors"].add(task['actor'])
        if task['type'] == 'gateway':
            sections[section_name]["has_gateways"] = True

    summary = []
    for i, (section_name, section_data) in enumerate(sections.items()):
        summary.append({
            "block_id": i + 1,
            "phase_or_main_goal": section_name,
            "primary_actors": list(section_data["actors"]),
            "contains_gateway": section_data["has_gateways"],
            "complexity_score": len(section_data["tasks"])
        })

    return summary


def create_detailed_steps(tasks):
    """
    Create a detailed step-by-step breakdown of the workflow.

    Transforms the sequential task list into a structured format suitable for
    detailed analysis and BPMN generation. Each step includes comprehensive
    information about actors, actions, and flow control.

    Args:
        tasks (list): Sequential tasks with full workflow information

    Returns:
        list: List of detailed step dictionaries containing:
            - id (str): Unique task identifier
            - workflow_name (str): Section or phase name
            - actors (list): Actors responsible for this step
            - action (str): Description of the task or action
            - next/branch (str/dict): Flow control information
    """
    detailed_steps = []
    for task in tasks:
        step = {
            "id": task["id"],
            "workflow_name": task.get("section_name", "Process Step"),
            "actors": [task["actor"]],
            "action": task["name"]
        }

        if task["type"] == "gateway" and "branches" in task:
            step["branch"] = task["branches"]
        else:
            step["next"] = task.get("next", "End")

        detailed_steps.append(step)

    return detailed_steps


def explain_bpmn_score(sequential_tasks, bpmn_readiness, quality_warnings):
    """
    Analyze and explain the BPMN readiness score by identifying specific issues.

    Performs diagnostic analysis on the extracted workflow to identify factors
    that affect BPMN generation quality. Checks for common issues like
    disconnected flows, unclear connections, naming problems, and structural
    inconsistencies.

    Args:
        sequential_tasks (list): The complete sequential workflow
        bpmn_readiness (float): Overall readiness score from ambiguity analysis
        quality_warnings (list): List of quality issues identified during processing

    Returns:
        str: Human-readable explanation of the BPMN readiness score including
             specific issues found and their impact on workflow quality
    """
    issues = []

    # Check for problems
    dangling_flows = sum(1 for task in sequential_tasks
                         if task.get('next') == 'END' and task != sequential_tasks[-1])
    if dangling_flows > 0:
        issues.append(f"{dangling_flows} dangling sequence flows")

    low_confidence_connections = sum(1 for t in sequential_tasks
                                     if t.get('link_confidence', 0) < 0.6)
    if low_confidence_connections > len(sequential_tasks) * 0.5:
        issues.append(f"{low_confidence_connections} unclear connections")

    actors = [t['actor'] for t in sequential_tasks]
    unique_actors = set(actors)
    if len(actors) - len(unique_actors) > 3:
        issues.append(f"{len(actors) - len(unique_actors)} duplicate actor assignments")

    technical_actors = [a for a in unique_actors
                        if a.lower() in ['sla', 'json', 'xml', 'bpmn', 'api', 'oidc']]
    if technical_actors:
        issues.append(f"{len(technical_actors)} technical terms as actors")

    timer_events = [t for t in sequential_tasks if t['type'] == 'timer_event']
    empty_timings = sum(1 for t in timer_events if not t.get('timing'))
    if empty_timings > 0:
        issues.append(f"{empty_timings} timer events without timing info")

    if quality_warnings:
        issues.append(f"{len(quality_warnings)} quality gate violations")

    if not issues:
        return f"BPMN readiness score {bpmn_readiness:.2f} - no major structural issues detected."

    return f"BPMN readiness score {bpmn_readiness:.2f} due to: {', '.join(issues)}."


def create_analysis_data(sequential_tasks, actor_info, process_name,
                         quality_warnings, ambiguity_result, cleaning_stats):
    """
    Compile comprehensive analysis data including metrics, quality assessment, and debugging information.

    Aggregates all processing results into a structured analysis dataset that provides
    complete visibility into the workflow extraction process. Includes workflow data,
    quality metrics, extraction statistics, and pipeline metadata for debugging and
    quality assurance purposes.

    Args:
        sequential_tasks (list): Final sequential workflow tasks
        actor_info (dict): Actor consolidation results and mappings
        process_name (str): Process identifier
        quality_warnings (list): Quality issues identified during processing
        ambiguity_result (dict): Results from ambiguity analysis phase
        cleaning_stats (dict): Text cleaning effectiveness metrics

    Returns:
        dict: Comprehensive analysis data structure containing:
            - Workflow data (tasks, summaries, actors)
            - Quality metrics (scores, explanations, statistics)
            - Extraction statistics (counts, confidence scores)
            - Pipeline metadata (version, phases, timestamps)
    """
    consolidated_actors = list(set(task['actor'] for task in sequential_tasks))
    lane_clustering = cluster_actors_into_lanes(consolidated_actors)

    task_count = sum(1 for t in sequential_tasks if t['type'] == 'task')
    gateway_count = sum(1 for t in sequential_tasks if t['type'] == 'gateway')

    bpmn_readiness = ambiguity_result["overall_metrics"]["overall_bpmn_readiness"]
    score_explanation = explain_bpmn_score(sequential_tasks, bpmn_readiness, quality_warnings)

    # Main workflow data
    workflow_data = {
        "process_name": process_name,
        "workflow_summary": {
            "total_tasks": task_count,
            "total_gateways": gateway_count,
            "actors": consolidated_actors,
            "extraction_method": "enhanced_granular",
            "recommended_lanes": len(lane_clustering["clusters"]),
            "lane_clustering": lane_clustering
        },
        "detailed_workflow_steps": sequential_tasks,
        "process_summary": create_process_summary(sequential_tasks),
        "actor_consolidation": actor_info
    }

    # Quality and metrics data
    quality_data = {
        "quality_metrics": {
            "extraction_quality": "high" if bpmn_readiness > 0.7 else "medium" if bpmn_readiness > 0.4 else "low",
            "bpmn_readiness_score": round(bpmn_readiness, 2),
            "score_explanation": score_explanation,
            "actor_reduction_percentage": actor_info.get("actor_resolution", {}).get("statistics", {}).get(
                "reduction_percentage", 0),
            "lane_reduction_percentage": lane_clustering["reduction"],
            "quality_issues_count": len(quality_warnings),
            "cleaning_effective": cleaning_stats.get("cleaning_effective", True)
        },
        "quality_assessment": {
            "cleaning_stats": cleaning_stats,
            "quality_warnings": quality_warnings,
            "ambiguity_analysis": {
                "overall_bpmn_readiness": ambiguity_result["overall_metrics"]["overall_bpmn_readiness"],
                "ready_sections": ambiguity_result["overall_metrics"]["bpmn_ready_sections"],
                "total_sections": ambiguity_result["overall_metrics"]["total_sections"],
                "detailed_results": ambiguity_result
            }
        },
        "extraction_statistics": {
            "tasks_extracted": len(sequential_tasks),
            "sections_processed": len(workflow_data.get("process_summary", [])),
            "granularity_level": "sentence",
            "confidence_scores": [task.get('confidence', 0.0) for task in sequential_tasks]
        },
        "pipeline_metadata": {
            "version": "unified_v2",
            "extraction_timestamp": None,
            "processing_phases": [
                "ingestion", "quality_gate", "extraction",
                "consolidation", "ambiguity_analysis", "lane_clustering", "generation"
            ]
        }
    }

    # Combine everything
    analysis_data = {**workflow_data, **quality_data}
    return analysis_data


def detect_process_name(text):
    """
    Automatically detect the process name from document content.

    Analyzes the beginning of the document to identify likely process names
    based on common patterns and keywords. Looks for lines containing "process"
    or formatted as headers that might indicate the main process being described.

    Args:
        text (str): Document text to analyze

    Returns:
        str: Detected process name or default "Business Process" if none found
    """
    for line in text.split('\n')[:10]:
        line = line.strip()
        if 'process' in line.lower() and len(line.split()) < 8:
            return line
        if line.endswith(':') and len(line.split()) < 6:
            return line.replace(':', '')
    return "Business Process"


def run_bpmn_pipeline(file_path, process_name=None):
    """
    Main entry point for the BPMN extraction pipeline.

    Orchestrates the complete workflow extraction process from file reading
    through final result generation. This is the primary interface function
    that external systems should use to process documents.

    Args:
        file_path (str): Path to the document file to process
        process_name (str, optional): Name for the process. If None, will attempt
                                    to auto-detect from document content

    Returns:
        dict: Result dictionary with keys:
            - success (bool): Processing success status
            - process_name (str): Process identifier
            - llm_prompt (str): Generated LLM prompt
            - analysis (dict): Complete analysis data
            - summary (str): Formatted summary of results
    """
    text = read_file(file_path)

    if not process_name:
        process_name = detect_process_name(text)

    result = process_bpmn_document(text, process_name)

    # Generate formatted summary
    def create_summary(result_data):
        status = "SUCCESS" if result_data["success"] else "FAILED"
        analysis = result_data.get("analysis", {})
        quality = analysis.get("quality_metrics", {}).get("extraction_quality", "unknown").upper()

        summary_data = analysis.get("workflow_summary", {})
        tasks = summary_data.get("total_tasks", 0)
        gateways = summary_data.get("total_gateways", 0)
        actors = len(summary_data.get("actors", []))

        return f"""
BPMN Pipeline Results
========================
{status} | Process: {result_data["process_name"]}
Tasks: {tasks} | Gateways: {gateways} | Actors: {actors}
LLM Prompt: {len(result_data["llm_prompt"]):,} chars (clean, ready for generation)
Analysis Data: {len(str(analysis)):,} chars (metrics, quality, debugging)
Quality: {quality} | BPMN Readiness: {analysis.get('quality_metrics', {}).get('bpmn_readiness_score', 'N/A')}
"""

    # Add summary to result
    result["summary"] = create_summary(result)

    return result


if __name__ == "__main__":
    result = run_bpmn_pipeline("data/sample_specs/third_test.txt")
    print(result["summary"])

    print(f"CLEAN LLM PROMPT ({len(result['llm_prompt'])} chars):")
    print("=" * 60)
    print(result["llm_prompt"])

    print(f"WORKFLOW DATA (up to quality assessment):")
    print("=" * 60)

    if result["success"] and result["analysis"]:
        print(f"STRUCTURED WORKFLOW STEPS:")
        steps = result["analysis"].get("detailed_workflow_steps", [])
        if steps:
            for i, step in enumerate(steps):
                print(f"Step {i + 1}:")
                print(json.dumps(step, indent=2))
                print()
        else:
            print("No workflow steps found")

        print(f"QUALITY METRICS (after workflow data):")
        print("=" * 60)

        if "quality_metrics" in result["analysis"]:
            quality = result["analysis"]["quality_metrics"]
            print(
                f"Quality: {quality.get('extraction_quality', 'unknown')} | BPMN Readiness: {quality.get('bpmn_readiness_score', 0)}")

            if "score_explanation" in quality:
                print(f"Score Analysis: {quality['score_explanation']}")

            print(f"Actor Reduction: {quality.get('actor_reduction_percentage', 0)}%")

            if "lane_reduction_percentage" in quality:
                print(f"Lane Reduction: {quality.get('lane_reduction_percentage', 0)}%")

        if "extraction_statistics" in result["analysis"]:
            stats = result["analysis"]["extraction_statistics"]
            print(f"Tasks Extracted: {stats.get('tasks_extracted', 0)}")

        if "workflow_summary" in result["analysis"]:
            summary = result["analysis"]["workflow_summary"]
            print(f"Total Tasks: {summary.get('total_tasks', 0)}")
            print(f"Total Gateways: {summary.get('total_gateways', 0)}")

            if "lane_clustering" in summary:
                clustering = summary["lane_clustering"]
                print(f"LANE CLUSTERING ({clustering['original_count']} → {clustering['clustered_count']} lanes):")
                for cluster_name, cluster_actors in clustering["clusters"].items():
                    print(f"  {cluster_name}: {', '.join(cluster_actors)}")

            actors = summary.get('actors', [])
            if actors:
                print(f"Actors ({len(actors)}): {', '.join(actors[:5])}")
                if len(actors) > 5:
                    print(f"  ... and {len(actors) - 5} more actors")
    else:
        print("Pipeline failed - no analysis data available")
        if result["analysis"] and result["analysis"].get("error"):
            print(f"Error: {result['analysis']['error']}")