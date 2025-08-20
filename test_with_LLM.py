import os
import json
from typing import Dict, Any, Optional, Tuple

# Import the main pipeline function
from unified_pipeline import run_bpmn_pipeline
from ingest.reader import read_file

# Only import Google Gemini for focused testing
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def configure_gemini() -> Tuple[Optional[Any], Optional[str]]:
    """
    Configure and return Google Gemini client for BPMN generation testing.

    Uses Google Gemini 1.5 Flash model for cost-effective BPMN XML generation.
    Requires GOOGLE_API_KEY environment variable to be set for authentication.

    Returns:
        Tuple containing the configured Gemini model and provider string,
        or (None, None) if configuration fails
    """
    if not GEMINI_AVAILABLE:
        print("ERROR: Google Generative AI not installed. Install with: pip install google-generativeai")
        return None, None

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        print("Please set your Google API key: export GOOGLE_API_KEY='your-api-key-here'")
        return None, None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Using Google Gemini 1.5 Flash (fast and cost-effective)")
        return model, "gemini"
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini: {e}")
        return None, None


def call_gemini(client: Any, prompt: str) -> Dict[str, Any]:
    """
    Call Google Gemini with the given prompt and return structured response.

    Sends the prompt to Gemini 1.5 Flash model and captures both the response
    text and token usage metrics for cost analysis and performance evaluation.

    Args:
        client: Configured Gemini GenerativeModel instance
        prompt: Text prompt for BPMN XML generation

    Returns:
        Dictionary containing response text, token counts, and error information
    """
    print("... Sending request to Gemini (this may take a moment) ...")

    try:
        # Count tokens in the input prompt for cost analysis
        prompt_token_count = client.count_tokens(prompt).total_tokens

        # Generate BPMN XML content from the prompt
        response = client.generate_content(prompt)

        # Count tokens in the generated response
        response_token_count = client.count_tokens(response.text).total_tokens

        return {
            "response_text": response.text,
            "prompt_tokens": prompt_token_count,
            "response_tokens": response_token_count,
            "total_tokens": prompt_token_count + response_token_count,
            "error": None
        }

    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}")
        return {
            "response_text": "",
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


def run_gemini_comparison(file_path: str) -> None:
    """
    Run comprehensive comparison between pipeline-structured and raw text prompts using Gemini.

    Executes two BPMN generation approaches: the structured pipeline approach that
    extracts workflow elements and creates targeted prompts, versus direct raw text
    processing. Provides detailed metrics and quality analysis for both approaches.

    Args:
        file_path: Path to the document file for BPMN generation testing
    """
    # Configure Gemini client for testing
    client, provider = configure_gemini()
    if not client:
        return

    print("\n" + "=" * 25 + " TEST 1: PIPELINE PROMPT " + "=" * 25)

    # Execute the structured pipeline workflow extraction
    # The unified_pipeline returns a dictionary, not an object with attributes
    pipeline_result = run_bpmn_pipeline(file_path)

    # Access the LLM prompt from the dictionary result
    pipeline_prompt = pipeline_result.get("llm_prompt")

    if not pipeline_prompt:
        print("ERROR: Pipeline failed to generate structured prompt.")
        print(f"Pipeline result keys: {list(pipeline_result.keys())}")
        return

    # Access analysis data from the pipeline result dictionary
    analysis = pipeline_result.get("analysis", {})
    workflow_summary = analysis.get("workflow_summary", {})
    quality_metrics = analysis.get("quality_metrics", {})

    # Display enhanced pipeline extraction metrics with safe access
    total_tasks = workflow_summary.get('total_tasks', 0)
    total_gateways = workflow_summary.get('total_gateways', 0)
    extraction_quality = quality_metrics.get('extraction_quality', 'Unknown')
    bpmn_readiness = quality_metrics.get('bpmn_readiness_score', 'Unknown')

    print(f"Pipeline extracted: {total_tasks} tasks, {total_gateways} gateways")
    print(f"Quality: {extraction_quality} | BPMN Readiness: {bpmn_readiness}")

    # Display detailed score analysis if available
    if "score_explanation" in quality_metrics:
        print(f"Score Analysis: {quality_metrics['score_explanation']}")

    # Display lane optimization metrics if available
    if "lane_clustering" in workflow_summary:
        clustering = workflow_summary["lane_clustering"]
        print(
            f"Lane Optimization: {clustering['original_count']} → {clustering['clustered_count']} lanes ({clustering['reduction']}% reduction)")

    print(f"Structured prompt length: {len(pipeline_prompt):,} characters")

    # Generate BPMN XML using the structured pipeline prompt
    pipeline_result_data = call_gemini(client, pipeline_prompt)

    print("\n--- PIPELINE PROMPT BPMN OUTPUT ---")
    print(pipeline_result_data["response_text"])
    print("-" * 50)

    # Execute baseline raw text approach for comparison
    print("\n" + "=" * 25 + " TEST 2: RAW TEXT PROMPT (Baseline) " + "=" * 27)

    # Load raw document text without processing
    raw_text = read_file(file_path)

    # Create simple baseline prompt for direct text processing
    raw_text_prompt = (
        "You are an expert BPMN 2.0 designer. Create a standard BPMN 2.0 XML diagram "
        f"based on the following raw text. The output must be a single, valid BPMN 2.0 XML code block.\n\n"
        f"--- RAW TEXT BEGIN ---\n{raw_text}\n--- RAW TEXT END ---"
    )

    print(f"Raw text length: {len(raw_text):,} characters")
    print(f"Raw prompt length: {len(raw_text_prompt):,} characters")

    # Generate BPMN XML using raw text approach
    raw_result_data = call_gemini(client, raw_text_prompt)

    print("\n--- RAW TEXT PROMPT BPMN OUTPUT ---")
    print(raw_result_data["response_text"])
    print("-" * 50)

    # Generate comprehensive comparison analysis
    _display_comparison_analysis(pipeline_result_data, raw_result_data, workflow_summary, quality_metrics)


def _display_comparison_analysis(pipeline_result: Dict[str, Any], raw_result: Dict[str, Any],
                                 workflow_summary: Dict[str, Any], quality_metrics: Dict[str, Any]) -> None:
    """
    Display detailed comparison analysis between pipeline and raw text approaches.

    Presents comprehensive metrics including token usage, efficiency gains,
    extraction quality, and structured analysis of both BPMN generation approaches.

    Args:
        pipeline_result: Results from structured pipeline approach
        raw_result: Results from raw text baseline approach
        workflow_summary: Extracted workflow element summary
        quality_metrics: Quality assessment metrics from pipeline
    """
    print("\n" + "=" * 35 + " COMPARISON SUMMARY " + "=" * 35)
    print(f"| {'Metric':<25} | {'Pipeline Prompt':<20} | {'Raw Text Prompt':<20} |")
    print(f"|{'-' * 27}|{'-' * 22}|{'-' * 22}|")
    print(f"| {'Input Tokens':<25} | {pipeline_result['prompt_tokens']:<20} | {raw_result['prompt_tokens']:<20} |")
    print(
        f"| {'Output Tokens (BPMN)':<25} | {pipeline_result['response_tokens']:<20} | {raw_result['response_tokens']:<20} |")
    print(f"| {'Total Tokens Used':<25} | {pipeline_result['total_tokens']:<20} | {raw_result['total_tokens']:<20} |")
    print("=" * 85)

    # Calculate efficiency metrics for both input and total token usage
    token_saving = raw_result['total_tokens'] - pipeline_result['total_tokens']
    efficiency_gain = (token_saving / raw_result['total_tokens']) * 100 if raw_result['total_tokens'] > 0 else 0

    print(f"\nPIPELINE EXTRACTION METRICS:")
    print(f" Tasks extracted: {workflow_summary.get('total_tasks', 0)}")
    print(f" Gateways extracted: {workflow_summary.get('total_gateways', 0)}")
    print(f" Actors identified: {len(workflow_summary.get('actors', []))}")
    print(f" Extraction quality: {quality_metrics.get('extraction_quality', 'Unknown')}")
    print(f" BPMN readiness: {quality_metrics.get('bpmn_readiness_score', 'Unknown')}")

    print(f"\nTOKEN EFFICIENCY ANALYSIS:")

    # Analyze input token efficiency
    input_token_saving = raw_result['prompt_tokens'] - pipeline_result['prompt_tokens']
    input_efficiency_gain = (input_token_saving / raw_result['prompt_tokens']) * 100 if raw_result[
                                                                                            'prompt_tokens'] > 0 else 0

    if input_token_saving > 0:
        print(f" Input token savings: {input_token_saving:,} tokens ({input_efficiency_gain:.1f}% reduction)")
        print(f" Structured prompt is {input_efficiency_gain:.1f}% more efficient for input processing")
    else:
        print(f" Input token overhead: {abs(input_token_saving):,} tokens ({abs(input_efficiency_gain):.1f}% increase)")
        print(f" Structured approach uses {abs(input_efficiency_gain):.1f}% more input tokens for enhanced structure")

    # Analyze total token efficiency
    if token_saving > 0:
        print(f" Total token savings: {token_saving:,} tokens ({efficiency_gain:.1f}% reduction)")
        print(f" Overall efficiency gain: {efficiency_gain:.1f}%")
    else:
        print(f" Total token overhead: {abs(token_saving):,} tokens ({abs(efficiency_gain):.1f}% increase)")
        print(f" Structured approach uses {abs(efficiency_gain):.1f}% more tokens for improved quality")

    # Display qualitative comparison analysis
    print(f"\nQUALITY COMPARISON ANALYSIS:")
    print(" Pipeline approach: Structured workflow extraction with validated actors and sequential flow")
    print(" Raw text approach: Direct text processing without workflow analysis or validation")
    print(" Recommendation: Compare the BPMN XML outputs above for structural and semantic quality differences")

    # Provide cost analysis for Gemini usage
    _display_cost_analysis(pipeline_result, raw_result)


def _display_cost_analysis(pipeline_result: Dict[str, Any], raw_result: Dict[str, Any]) -> None:
    """
    Display cost analysis for Gemini token usage comparison.

    Provides estimated costs for both approaches using current Gemini pricing
    to help evaluate the cost-effectiveness of the structured pipeline approach.

    Args:
        pipeline_result: Token usage data from pipeline approach
        raw_result: Token usage data from raw text approach
    """
    # Gemini 1.5 Flash pricing (as of current rates)
    input_cost_per_token = 0.000000075  # $0.075 per 1M tokens
    output_cost_per_token = 0.0000003  # $0.30 per 1M tokens

    # Calculate costs for pipeline approach
    pipeline_input_cost = pipeline_result['prompt_tokens'] * input_cost_per_token
    pipeline_output_cost = pipeline_result['response_tokens'] * output_cost_per_token
    pipeline_total_cost = pipeline_input_cost + pipeline_output_cost

    # Calculate costs for raw text approach
    raw_input_cost = raw_result['prompt_tokens'] * input_cost_per_token
    raw_output_cost = raw_result['response_tokens'] * output_cost_per_token
    raw_total_cost = raw_input_cost + raw_output_cost

    cost_difference = raw_total_cost - pipeline_total_cost

    print(f"\nGEMINI COST ANALYSIS (Estimated):")
    print(
        f" Pipeline approach: ${pipeline_total_cost:.6f} (Input: ${pipeline_input_cost:.6f}, Output: ${pipeline_output_cost:.6f})")
    print(f" Raw text approach: ${raw_total_cost:.6f} (Input: ${raw_input_cost:.6f}, Output: ${raw_output_cost:.6f})")

    if cost_difference > 0:
        savings_percentage = (cost_difference / raw_total_cost) * 100
        print(f" Cost savings: ${cost_difference:.6f} ({savings_percentage:.1f}% reduction)")
    else:
        overhead_percentage = (abs(cost_difference) / raw_total_cost) * 100
        print(f" Cost overhead: ${abs(cost_difference):.6f} ({overhead_percentage:.1f}% increase)")


def main() -> None:
    """
    Main execution function for Gemini-based BPMN generation testing.

    Runs comprehensive comparison between structured pipeline approach and
    raw text processing for BPMN XML generation using Google Gemini.
    """
    # Define the test document for BPMN generation
    test_file = "data/sample_specs/third_test.txt"

    if not os.path.exists(test_file):
        print(f"ERROR: Test file '{test_file}' not found.")
        print("Please update the test_file path to point to your document.")
        print("Available options:")
        print("  - Check the file path is correct")
        print("  - Ensure the document exists in the specified location")
        return

    print("Starting Gemini-based BPMN generation comparison...")
    print(f"Test document: {test_file}")

    # Execute the comprehensive comparison analysis
    run_gemini_comparison(test_file)


if __name__ == '__main__':
    main()