# ACE Agent: Automated Simplification into "Leichte Sprache"

This repository contains the implementation of the **ACE (Adaptive Context-aware Editing) Agent**, a recursive multi-agent system designed to simplify German text into "Leichte Sprache" (Easy Language).

## 🚀 Overview

The system uses LLMs (via Ollama) to iteratively simplify text. There are two variants included:

1.  **3-Node Agent (`agent_3node.py`)**: The advanced architecture featuring acting as a **Generator**, **Reflector** (critic), and **Curator** (memory manager). It evolves its "Strategy Cheatsheet" (memory) over time.
2.  **2-Node Agent (`agent_2node.py`)**: A leaner baseline version (Generator -> Curator) that uses implicit reflection.

## 📋 Requirements

*   **Python 3.10+**
*   **Ollama**: You must have [Ollama](https://ollama.com/) installed and running locally.

### Python Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Models
Pull the recommended models in Ollama before running:
```bash
ollama pull llama3.1:8b
ollama pull mixtral:8x7b
# or any other model you wish to configure
```

## ⚙️ Configuration

**Important:** Before running, check `inference_config.py`. 
You can adjust:
*   `DEFAULT_OLLAMA_URL`: URL of your Ollama server (default: `http://localhost:11434`).
*   `DEFAULT_MODELS`: List of models to use for batch experiments.
*   `DEFAULT_INPUT_DIR`: Folder containing `.txt` files to process.

## 🏃 Usage

### 1. Running on a Single File
You can run an agent directly on a single text file.

**3-Node Agent:**
```bash
python agent_3node.py \
  --input "path/to/input.txt" \
  --output "path/to/output.txt" \
  --level "Level 1 (Leichte Sprache)" \
  --model-name "llama3.1:8b"
```

**2-Node Agent:**
```bash
python agent_2node.py \
  --input "path/to/input.txt" \
  --output "path/to/output.txt" \
  --level "Level 1 (Leichte Sprache)" \
  --model-name "llama3.1:8b"
```

### 2. Batch Processing (Recommended)
Use the `inference_batch_runner.py` to process an entire folder of directory of text documents using multiple models sequentially.

```bash
python inference_batch_runner.py \
  --agent-script agent_3node.py \
  --input-dir ./Recepies-To-Be-EVAL/inferset-sa \
  --models llama3.1:8b mixtral:8x7b
```

This will:
1.  Iterate through every file in the input directory.
2.  Run the selected agent with every specified model.
3.  Save results in `EXPERIMENT_RESULTS/<model_name>/`.
4.  Maintain a persistent memory (SQLite database) for the agent during the run.

## 📂 Key Files

*   `agent_3node.py`: Main agent logic (Generator-Reflector-Curator).
*   `agent_2node.py`: Baseline agent logic (Generator-Curator).
*   `fullrulesfromsa.txt`: The "Seed Memory" containing the official rules for Leichte Sprache.
*   `inference_config.py`: Central configuration file.
*   `inference_batch_runner.py`: Script for running experiments.

## 🧠 Memory & Persistence

The agent uses a "Strategy Cheatsheet" (Memory) that evolves as it processes more files.
*   **Seed Memory:** The agent starts with rules from `fullrulesfromsa.txt` (if provided via `--seed-memory`).
*   **Database:** The evolving memory is stored in `agent_memory.sqlite`.

## 📊 Evaluation

This repository also includes the scripts we used to evaluate the agent's performance (in the `EVAL` folder).

### 1. Requirements

Install the evaluation-specific dependencies:
```bash
cd EVAL
pip install -r requirements.txt
python -m spacy download de_core_news_lg
```

### 2. Metrics
The script `eval_metrics.py` computes:
*   **Readability:** Flesch Reading Ease, Wiener Sachtextformel.
*   **Complexity:** Syllable counts, Zipf frequency.
*   **Rule Violations:** Passive voice, Genitive case, Subjunctive mood, Sentence/Word length.
*   **BERTScore:** Semantic similarity (if reference text is available).

### 3. Running Evaluation
To evaluate a folder of results (e.g., from the agent batch run):

```bash
python EVAL/batch_eval.py \
  --input-dir EXPERIMENT_RESULTS/llama3.1_8b/simplified_texts \
  --output-dir EVAL_RESULTS/llama3.1_8b \
  --spacy-model de_core_news_lg
```

This will generate a `.csv` file with metrics for every text file.

### 4. Summarizing Results
To create a JSON summary and LaTeX tables from the CSVs:

```bash
python EVAL/summarize_results.py \
  --input EVAL_RESULTS/llama3.1_8b.csv \
  --output-dir EVAL_RESULTS/summary/
```

