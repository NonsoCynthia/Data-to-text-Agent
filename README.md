# Are Multi-Agents the New Pipeline Architecture for Data-to-Text Systems?

This repository contains the code and experiments for our research paper:  
**“Are Multi-Agents the new Pipeline Architecture for Data-to-Text Systems?”**  

Our work introduces the first **LLM-based multi-agent framework** for data-to-text generation, combining the strengths of pipeline modularity and end-to-end fluency. The architecture coordinates specialized worker agents (**Content Ordering, Text Structuring, Surface Realization**) under an **Orchestrator**, while **Guardrail agents** provide iterative feedback to ensure fluency, factual consistency, and completeness. The **Finalizer** depending on the quality of the text, may return a more refined or polished final text. 

Evaluation is carried out using Automatic evaluation (BLEU, METEOR, ChrF, TER, BERTScore, COMET) and Human/LLM-as-judge evaluations.  

---

## Project Structure

- **agents/**  
  - `orchestrator.py`: Controls the workflow and assigns tasks to agents  
  - `worker.py`: Implements worker agents (CO, TS, SR)  
  - `guardrail.py`: Implements guardrail agents for feedback validation  
  - `finalizer.py`: Produces validated final outputs  

- **utilities/**  
  - `llm_model.py`: Unified LLM interface  
  - `agent_prompts.py`: Prompt templates for each agent  

- **results/**  
  Stores generated outputs and evaluation files.  

- `run_test.sh`: Automates inference and evaluation runs  
- `run_inference.py`: Runs model inference with selected LLMs  
- `run_evaluation.py`: Computes evaluation metrics  

---

## Setup

### Environment
- Python 3.8 (Ideal for Comet evaluation) or higher  
- Required packages (see `requirements.txt`)  

---

## Installation

Clone this repository:
```bash
git clone https://github.com/NonsoCynthia/Data-to-Text-Agent
cd Data-to-Text-Agent
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run Inference and Evaluation
Make the script executable:
```bash
chmod +x run_test.sh
```

Run the script:
```bash
./run_test.sh
```

By default, results will be saved in the `results/` folder.  

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.  

---