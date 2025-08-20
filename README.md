# Preprocessing Prompt Engineering

Turn messy business process text into **structured, LLM-ready prompts** that generate **executable BPMN 2.0 XML**.

It handles:
- Text ingestion from `.txt`, `.docx`, `.pdf`
- Cleaning and structure preservation
- Quality gates for actor/action clarity
- Sentence-level workflow extraction (tasks, gateways, timers, messages)
- Actor consolidation and swimlane clustering
- Ambiguity analysis and recommendations
- Prompt generation for LLMs (tested with Gemini, works with others)

---

## Features

- **Document ingestion**: TXT/DOCX/PDF (`ingest/reader.py`)
- **Cleaning**: preserves process structure (`ingest/cleaner.py`)
- **Quality checks**: semantic gate (`quality_gate/sqg.py`, `ambiguity/business_process.py`)
- **Ambiguity analysis**: lexical, pronouns, coref, flow (`ambiguity/*.py`)
- **Workflow extraction**: sentence-level (`segmentation/sentence_workflow_extractor.py`)
- **Actor consolidation**: canonical roles (`postprocess/actor_consolidator.py`)
- **Prompt builder**: executable BPMN prompt (`json_gen/prompt_builder.py`)
- **Unified pipeline**: orchestration (`unified_pipeline.py`)
- **LLM test harness**: pipeline vs raw prompt (`test_with_LLM.py`)

---

## Installation

```bash
# Python Version 3.10.11 was used for this project
# optional but recommended
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# required language models (not installed by pip)
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

# coreference model required by ambiguity/coref.py
pip install coreferee
python -m coreferee install en
