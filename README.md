# Preprocessing Prompt Engineering

NLP pipeline for turning unstructured business-process text into structured, **LLM-ready BPMN generation prompts**.

This project uses Natural Language Processing (NLP) to extract actors, actions, sequence logic, gateways, timer/message events, and ambiguity signals from narrative documents (`.txt`, `.docx`, `.pdf`). The output is a high-structure prompt designed to improve BPMN 2.0 XML generation quality.

## Problem and Value

Business process descriptions are often written as long prose with inconsistent structure. Directly sending that raw text to an LLM frequently produces incomplete or invalid BPMN.

This project adds an NLP preprocessing layer that:
- normalizes noisy text,
- validates process signal quality,
- extracts sentence-level workflow semantics,
- consolidates actors for lane design,
- scores ambiguity/BPMN readiness,
- emits a structured prompt for executable BPMN generation.

## What It Does

- Ingests text from `.txt`, `.docx`, and `.pdf`
- Cleans and normalizes process narratives while preserving structure
- Runs quality gates for actor/action and process-signal readiness
- Performs sentence-level workflow extraction with NLP
- Detects tasks, gateways, timer events, and message events
- Consolidates actor variations into canonical lanes
- Runs ambiguity analysis and BPMN-readiness scoring
- Builds an enhanced LLM prompt optimized for BPMN XML generation

## How It Works (NLP Pipeline)

1. **Ingestion** (`ingest/reader.py`): reads source files and normalizes raw text input.
2. **Cleaning** (`ingest/cleaner.py`): removes metadata/boilerplate, fixes formatting artifacts, preserves process structure.
3. **Quality Gate** (`quality_gate/sqg.py`): checks actor-action ratios and process-signal indicators.
4. **Sentence-Level Extraction** (`segmentation/sentence_workflow_extractor.py`): uses spaCy parsing and rules to extract tasks, gateways, timer events, and message events.
5. **Actor Consolidation** (`postprocess/actor_consolidator.py`): resolves actor aliases to canonical business roles.
6. **Ambiguity + BPMN Readiness** (`ambiguity/core.py` + helpers): computes clarity issues and readiness metrics.
7. **Prompt Generation** (`json_gen/prompt_builder.py`): emits a structured BPMN-focused prompt for LLM consumption.

## Architecture (Data Flow)

```text
Document (.txt/.docx/.pdf)
        |
        v
[Ingestion + Cleaning]
        |
        v
[Quality Gate: actor/action + process signals]
        |
        v
[Sentence NLP Extraction]
  - tasks
  - gateways
  - timer events
  - message events
        |
        v
[Actor Consolidation + Lane Clustering]
        |
        v
[Ambiguity Analysis + BPMN Readiness]
        |
        v
[Structured LLM Prompt]
        |
        v
BPMN 2.0 XML generation (LLM side)
```

## Quickstart

### 1. Prerequisites

- Python 3.10+ (project previously tested on Python 3.10.11)
- `pip`

### 2. Install Base Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Install Required NLP Models

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

pip install coreferee
python -m coreferee install en
```

## Run the Pipeline

```bash
python unified_pipeline.py
```

Default sample input in the script:
- `data/sample_specs/third_test.txt`

## Optional: LLM Comparison with Gemini

`test_with_LLM.py` compares:
- pipeline-generated structured prompt vs.
- raw text prompt baseline.

Install optional dependency:

```bash
pip install google-generativeai
```

Set environment variable (see `.env.example`):

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key"

# macOS/Linux
export GOOGLE_API_KEY="your_api_key"
```

Run:

```bash
python test_with_LLM.py
```

## Output Contract (from `run_bpmn_pipeline`)

Main return object includes:

```json
{
  "success": true,
  "process_name": "Business Process",
  "llm_prompt": "<structured BPMN generation prompt>",
  "analysis": {
    "workflow_summary": {},
    "detailed_workflow_steps": [],
    "quality_metrics": {},
    "quality_assessment": {},
    "extraction_statistics": {},
    "pipeline_metadata": {}
  },
  "summary": "..."
}
```

## Portfolio Highlights

- End-to-end NLP preprocessing pipeline for process intelligence
- Hybrid approach: linguistic analysis + business rules + quality scoring
- Production-oriented output contract (`success`, `llm_prompt`, `analysis`)
- Clear modular decomposition by pipeline stage

## Limitations

- Extraction quality depends on input narrative clarity and domain phrasing.
- Rule-driven components are currently tuned for business-process style text.
- BPMN XML generation is delegated to external LLMs and may still require validation.

## Roadmap

- Add reproducible benchmark set with expected BPMN artifacts
- Add automated tests for extraction and ambiguity metrics
- Add optional BPMN XML schema validation post-generation
- Add CI workflow for lint/test/readme checks

## GitHub Push Handoff (Keep GitLab Remote Intact)

Use these commands locally when you are ready to publish to GitHub:

```bash
git remote add github <github-repo-url>
git push github ppe_project:main
```

Alternative (keep branch name):

```bash
git push github ppe_project
```

## License

MIT License. See [`LICENSE`](LICENSE).

## Attribution

Developed as an NLP-focused preprocessing project for BPMN prompt engineering and workflow extraction research/practice.
