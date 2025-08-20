import json
from typing import List, Dict, Any


def build_enhanced_llm_prompt(enhanced_output: Dict) -> str:
    """
    Build optimized prompt for BPMN generation from workflow data.

    Analyzes the enhanced workflow output to determine appropriate granularity level
    and constructs specialized prompts for either granular sentence-level workflow
    extraction or section-level process modeling. Automatically selects prompt
    strategy based on data availability and metrics.

    Args:
        enhanced_output: Dictionary containing processed workflow data including
                        process summary, detailed steps, metrics, and granular tasks

    Returns:
        Formatted prompt string optimized for BPMN 2.0 XML generation with
        specific instructions for executable workflow modeling
    """
    # Extract key data
    process_summary = enhanced_output.get("process_summary", [])
    detailed_steps = enhanced_output.get("detailed_steps", [])
    metrics = enhanced_output.get("metrics", {})
    granular_tasks = enhanced_output.get("granular_tasks", [])

    # Determine granularity level
    use_granular = len(granular_tasks) > 0 and metrics.get("granularity_level") == "sentence"

    # Build system prompt with executable BPMN requirements
    system_prompt = """You are an expert BPMN 2.0 designer. Your task is to create a complete, executable BPMN 2.0 XML diagram that can be imported and rendered in BPMN modeling tools like Camunda, Signavio, or bpmn.io.

The input provides detailed workflow steps with specific actors and clear decision points. Each step represents a concrete business activity.

CRITICAL REQUIREMENTS FOR EXECUTABLE BPMN:
1. All tasks must be serviceTask or userTask elements with proper lane assignments
2. All sequence flows must have explicit sourceRef and targetRef attributes
3. All gateways must have proper incoming/outgoing flow definitions
4. Lane assignments must be explicit using flowNodeRef elements
5. Process must be marked as isExecutable="true"
6. All IDs must be unique and properly referenced

Your output must be a single, valid BPMN 2.0 XML code block."""

    if use_granular:
        return _build_granular_prompt(system_prompt, granular_tasks, metrics, enhanced_output)
    else:
        return _build_section_prompt(system_prompt, process_summary, detailed_steps, metrics)


def _build_granular_prompt(system_prompt: str, granular_tasks: List[Dict],
                           metrics: Dict, enhanced_output: Dict) -> str:
    """
    Construct specialized prompt for granular sentence-level workflow processing.

    Creates detailed workflow step descriptions from granular task data, including
    task types, actor assignments, timing information, and confidence metrics.
    Optimized for high-precision BPMN generation from sentence-level extracted
    workflow elements.

    Args:
        system_prompt: Base system instructions for BPMN generation
        granular_tasks: List of detailed task dictionaries with type, actor, and flow data
        metrics: Dictionary containing workflow analysis metrics and statistics
        enhanced_output: Complete workflow data for additional context

    Returns:
        Complete prompt string with granular workflow instructions and element mapping
    """

    # Extract unique actors
    unique_actors = list(set(task['actor'] for task in granular_tasks))

    # Create workflow steps section (no emojis, clean format)
    workflow_steps = []
    for task in granular_tasks:
        confidence = task.get('link_confidence', 0.5)
        confidence_indicator = " HIGH-CONF" if confidence > 0.8 else ""

        if task['type'] == 'task':
            next_ref = task.get('next', 'END')
            workflow_steps.append(f"{task['id']}: {task['actor']} → {task['name']} → {next_ref}{confidence_indicator}")

        elif task['type'] == 'timer_event':
            timing = task.get('timing', {})
            duration = timing.get('duration', 'unspecified time')
            next_ref = task.get('next', 'END')
            workflow_steps.append(f"{task['id']}: {task['actor']} → TIMER: {task['name']} ({duration}) → {next_ref}{confidence_indicator}")

        elif task['type'] == 'message_event':
            direction = task.get('direction', 'intermediate')
            next_ref = task.get('next', 'END')
            direction_type = "SEND" if direction == 'outgoing' else "RECEIVE" if direction == 'incoming' else "MESSAGE"
            workflow_steps.append(f"{task['id']}: {task['actor']} → {direction_type}: {task['name']} → {next_ref}{confidence_indicator}")

        else:  # gateway
            workflow_steps.append(f"{task['id']}: {task['actor']} → DECISION: {task['name']}")
            if 'branches' in task:
                for branch in task['branches']:
                    condition = branch.get('condition', 'Condition met')
                    outcome = branch.get('outcome', 'CONTINUE')
                    workflow_steps.append(f"  IF: {condition} → {outcome}")

    workflow_section = "\n".join(workflow_steps)

    # Calculate statistics
    task_count = len([t for t in granular_tasks if t['type'] == 'task'])
    gateway_count = len([t for t in granular_tasks if t['type'] == 'gateway'])
    timer_count = len([t for t in granular_tasks if t['type'] == 'timer_event'])
    message_count = len([t for t in granular_tasks if t['type'] == 'message_event'])
    high_confidence_links = len([t for t in granular_tasks if t.get('link_confidence', 0) > 0.8])

    instructions = f"""
EXECUTABLE BPMN GENERATION INSTRUCTIONS:

Element Summary: {task_count} tasks, {gateway_count} gateways, {timer_count} timers, {message_count} messages. {high_confidence_links} high-confidence links.

Element Mapping:
• Tasks: serviceTask (system actors with "System"/"API"/"Core"/"Digital") or userTask (human actors)
• Timers: intermediateCatchEvent with timerEventDefinition, duration from timing info (PT5M = 5 min)
• Messages: intermediateThrowEvent (SEND) or intermediateCatchEvent (RECEIVE) with messageEventDefinition  
• Gateways: exclusiveGateway with conditional outgoing flows
• Flows: sequenceFlow with sourceRef/targetRef connecting elements (T1→TM1→M1→T2)

Structure Requirements:
• Process marked isExecutable="true"
• One Lane per actor with flowNodeRef elements pointing to their tasks/gateways/events
• All elements connected with sequenceFlow using sourceRef/targetRef
• Start Event connects to first task, End Event for completion
• All IDs unique and properly cross-referenced
• Valid BPMN 2.0 XML with proper namespace declarations

Actors ({len(unique_actors)} lanes):
{chr(10).join([f"{i+1}. {actor}" for i, actor in enumerate(unique_actors)])}

Workflow Steps:
{workflow_section}

Generate complete executable BPMN 2.0 XML diagram with enhanced timer and message events."""

    return system_prompt + instructions


def _build_section_prompt(system_prompt: str, process_summary: List[Dict],
                          detailed_steps: List[Dict], metrics: Dict) -> str:
    """
    Construct prompt for section-level workflow processing from higher-level descriptions.

    Builds prompts from process summary and detailed step data when granular extraction
    is not available or appropriate. Focuses on phase-based workflow organization with
    clear actor assignments and decision point identification.

    Args:
        system_prompt: Base system instructions for BPMN generation
        process_summary: List of process phase dictionaries with actors and goals
        detailed_steps: List of detailed step dictionaries with actions and flows
        metrics: Dictionary containing workflow complexity and structure metrics

    Returns:
        Complete prompt string with section-level workflow instructions and actor mapping
    """

    # Create process summary section
    summary_section = ""
    for block in process_summary:
        actors_str = ", ".join(block.get("primary_actors", ["Unknown"]))
        gateway_indicator = " (Contains Decision Points)" if block.get("contains_gateway", False) else ""
        summary_section += f"• Phase {block.get('block_id', 1)}: {block.get('phase_or_main_goal', 'Process Phase')} [{actors_str}]{gateway_indicator}\n"

    # Create detailed steps section
    steps_section = ""
    for step in detailed_steps:
        actors_str = ", ".join(step.get("actors", ["Unknown"]))
        if step.get("branch"):
            steps_section += f"• {step.get('id', 'X')}: {actors_str} → DECISION: {step.get('action', 'Decision')}\n"
            for branch in step["branch"]:
                condition = branch.get("condition", "Condition")
                steps_section += f"  - {condition} → {branch.get('next', 'Continue')}\n"
        else:
            next_ref = step.get("next", "End")
            steps_section += f"• {step.get('id', 'X')}: {actors_str} → {step.get('action', 'Action')} → {next_ref}\n"

    # Extract all unique actors
    all_actors = set()
    for step in detailed_steps:
        all_actors.update(step.get("actors", []))
    actors_list = "\n".join([f"{i + 1}. {actor}" for i, actor in enumerate(sorted(all_actors))])

    instructions = f"""
INSTRUCTIONS FOR EXECUTABLE SECTION-LEVEL WORKFLOW:
1. PROCESS STRUCTURE: Review the process summary to understand the overall workflow structure
2. WORKFLOW DETAILS: Use the detailed steps to create specific BPMN elements:
   - Each step with "actors" becomes a serviceTask/userTask in a Lane for that actor
   - Steps with "branch" become Exclusive Gateways with conditional flows
   - Steps with "next" become connected via Sequence Flows with sourceRef/targetRef
3. EXECUTABLE REQUIREMENTS: Ensure all elements are properly structured for execution

PROCESS METRICS:
- Total workflow phases: {metrics.get('total_blocks', 0)}
- Unique actors: {len(all_actors)}
- Decision points: {metrics.get('gateway_blocks', 0)}
- Average complexity: {metrics.get('average_complexity', 0):.1f}

PROCESS SUMMARY:
{summary_section.strip()}

ACTORS (create lanes with flowNodeRef for these):
{actors_list}

DETAILED WORKFLOW STEPS:
{steps_section.strip()}

EXECUTABLE BPMN REQUIREMENTS:
- Start with Start Event
- Create one Lane per unique actor with flowNodeRef elements
- Use serviceTask for system actors, userTask for human actors
- Connect all elements with Sequence Flows using sourceRef/targetRef
- Use Exclusive Gateways for decisions with proper incoming/outgoing flows
- End with End Event
- Mark process as isExecutable="true"
- Output valid, executable BPMN 2.0 XML only

Generate the complete, executable BPMN 2.0 XML diagram now."""

    return system_prompt + instructions