# When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems

**Paper:** [arXiv:2601.16280](https://arxiv.org/abs/2601.16280)

**Authors:** Donghao Huang, Gauri Malwe, Zhaoxia Wang

**Venue:** 9th International Conference on Artificial Intelligence and Big Data (ICAIBD 2026)

## Overview

This repository contains the code and data for our paper, which introduces a diagnostic framework for evaluating tool invocation reliability in multi-agent LLM systems. Using an SME invoice reconciliation task as a benchmark, we evaluate 1,980 deterministic test cases across open-source and proprietary models on commodity edge hardware. We propose a 12-category failure taxonomy spanning tool initialization, parameter handling, execution, and result interpretation, and show that models as small as Qwen2.5:14b can achieve 96.6% task success at 7.3s latency on a single RTX 4090.

## Repository Structure

```text
.
├── app.py                  # Main application entry point
├── dataset/                # Synthetic dataset (emails, transactions, ground truth)
│   └── attachments/        # Invoice image attachments (990 JPEGs)
├── modelfiles/             # Ollama modelfiles for functionary models
├── notebooks/              # Analysis and evaluation notebooks
├── requirements.txt
├── results/                # Evaluation results and precomputed metrics
│   ├── Anthropic/          # Claude 3.5 / 3.7 Sonnet results
│   ├── H100/               # H100 open-source model results
│   ├── H100_r2/            # H100 functionary model results
│   ├── M3-Max/             # Apple M3 Max results
│   ├── OpenAI/             # GPT-4o / GPT-4.1 family results
│   ├── RTX-4090_r2/        # RTX 4090 qwen2.5:14b results (error analysis)
│   ├── RTX-4090_r3/        # RTX 4090 final results (paper)
│   ├── RTX-A6000/          # RTX A6000 results (error analysis)
│   ├── RTX-A6000_r3/       # RTX A6000 final results (paper)
│   └── metrics*.csv        # Precomputed metrics per platform
├── scripts/                # Evaluation shell scripts
└── src/                    # Source code
    ├── data/               # Database scripts and processed outputs
    ├── llm/                # Agent definitions and LangGraph workflows
    └── misc/               # Metrics calculation and utilities
```

## Setup

### Prerequisites

#### libmagic

**macOS:**
```bash
brew install libmagic
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libmagic1
```

**Windows:**
```bash
pip install python-magic-bin
```

#### Python Packages

Recommended Python version: 3.12

```bash
pip install -r requirements.txt
```

#### Ollama

Install [Ollama](https://github.com/ollama/ollama/blob/main/README.md), then pull the vision model used for OCR:

```bash
ollama pull llama3.2-vision:11b
```

### Environment File

Copy `.env.example` to `.env` and fill in the required values:

```bash
OPENAI_API_KEY="your-api-key-here"
ANTHROPIC_API_KEY="your-api-key-here"
PYTHONPATH="."
BASE_URL="http://localhost:11434/v1"   # Ollama API endpoint
MODEL="qwen2.5:7b"                     # Fallback model
FINANCE_CLERK_MODEL="qwen2.5:7b"       # Agent that triggers the OCR tool
SUPERVISOR_MODEL="qwen2.5:7b"          # Supervisor agent
SQL_MODEL="qwen2.5:7b"                 # SQL agents
VISION_BASE_URL="http://localhost:11434/v1"
VISION_MODEL="llama3.2-vision:11b"     # OCR / vision model
```

## Running the Application

```bash
python app.py
```

### ReconApp Parameters

| Parameter             | Type | Description                                             | Default                       |
| --------------------- | ---- | ------------------------------------------------------- | ----------------------------- |
| `supervisor_model`    | str  | LLM model for the supervisor                            | `env["SUPERVISOR_MODEL"]`     |
| `sql_model`           | str  | LLM model for the SQL agents                            | `env["SQL_MODEL"]`            |
| `finance_clerk_model` | str  | LLM model for the finance clerk (triggers the OCR tool) | `env["FINANCE_CLERK_MODEL"]`  |
| `vision_model`        | str  | Vision model for OCR                                    | `env["VISION_MODEL"]`         |
| `max_retries`         | int  | Max retries when LangGraph encounters an error          | `3`                           |
| `batch_size`          | int  | Batch size for querying the email database              | `10`                          |
| `tool_based`          | bool | Use SQL query tool instead of SQL agents                | `False`                       |
| `reset_db_state`      | bool | Reset the database before running                       | `False`                       |

**Example:**

```python
ReconApp(
    supervisor_model=os.environ["SUPERVISOR_MODEL"],
    sql_model=os.environ["SQL_MODEL"],
    finance_clerk_model=os.environ["FINANCE_CLERK_MODEL"],
    vision_model=os.environ["VISION_MODEL"],
    max_retries=3,
    batch_size=10,
    tool_based=True,
    reset_db_state=True,
).run()
```

## Reproducing Paper Results

### Evaluating OpenAI Models

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
./scripts/eval-gpt.sh gpt-4o-mini
./scripts/eval-gpt.sh gpt-4o
./scripts/eval-gpt.sh gpt-4.1-nano
./scripts/eval-gpt.sh gpt-4.1-mini
./scripts/eval-gpt.sh gpt-4.1
```

### Evaluating Anthropic Models

Set `ANTHROPIC_API_KEY` in `.env`, then run:

```bash
./scripts/eval-gpt.sh claude-3-5-sonnet-20241022
./scripts/eval-gpt.sh claude-3-7-sonnet-20250219
```

### Evaluating Open-Source Models

First, ensure `llama3.2-vision:11b` is running as the vision model (see Ollama setup above).

**RTX 4090 (24GB VRAM)** — evaluates qwen2.5:3b, 7b, 14b:
```bash
./scripts/eval-rtx-4090.sh
```

**RTX A6000 (48GB VRAM)** — evaluates qwen2.5:3b, 7b, 14b, 32b, 72b, functionary-small, functionary-medium:
```bash
./scripts/eval-rtx-a6000.sh
```

**Apple M3 Max** — evaluates qwen2.5:3b, 7b, 14b, functionary-small:
```bash
./scripts/eval-m3-max.sh
```

For functionary models, create the Ollama modelfile first:
```bash
./scripts/create-models.sh
```

### Calculating Metrics

After a run, calculate metrics from the result databases:

```bash
python src/misc/calc_metrics.py results/RTX-A6000_r3 RTX-A6000_r3
```

This reads the `emails.db` files under each model subdirectory and writes `results/metrics_RTX-A6000_r3.csv`.

### Visualizing Results

Open and run the final analysis notebook:

```text
notebooks/99r3Data_Analysis_all_models-full.ipynb
```

This merges the per-platform metrics CSVs and produces all plots and tables in the paper.

## Dataset

The synthetic dataset is located in `dataset/`:

| File                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| `emails.csv`          | 990 synthetic emails with optional attachments   |
| `transactions.csv`    | Invoice/transaction records                      |
| `ground_truth.csv`    | Ground truth reconciliation labels               |
| `attachments/`        | 990 synthetic invoice JPEG images                |

### Email Table Schema

| Header               | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `email_id`           | Unique email ID                                                                               |
| `sender_email`       | Sender address                                                                                |
| `recipient_email`    | Recipient address                                                                             |
| `subject`            | Email subject                                                                                 |
| `email_body`         | Email body text                                                                               |
| `filename`           | Attachment filename (empty if none)                                                           |
| `timestamp`          | Email timestamp (ISO format)                                                                  |
| `process_status`     | `NOT_STARTED` / `NOT_INVOICE` / `ERROR` / `API_ERROR` / `RECURSION_LIMIT_REACHED` / `SUCCESS` |
| `response`           | Final LLM response text                                                                       |
| `start_time`         | Processing start time (ISO format)                                                            |
| `end_time`           | Processing end time (ISO format)                                                              |
| `full_logs`          | JSON list of all LLM exchange logs                                                            |
| `total_time`         | Total processing time (seconds)                                                               |
| `successful_requests`| Number of successful LLM API calls                                                            |
| `total_tokens`       | Total tokens used                                                                             |
| `prompt_tokens`      | Prompt tokens used                                                                            |
| `completion_tokens`  | Completion tokens used                                                                        |
| `total_cost`         | Estimated cost (OpenAI/Anthropic models only)                                                 |

### Transactions Table Schema

| Header                 | Description                                 |
| ---------------------- | ------------------------------------------- |
| `invoice_id`           | Invoice ID                                  |
| `bank_name`            | Bank to pay the invoice to                  |
| `transaction_id`       | Transaction ID                              |
| `transaction_date`     | Transaction date                            |
| `amount`               | Transaction amount                          |
| `recipient_name`       | Person to make payment to                   |
| `sender_name`          | Invoice pending collection from             |
| `reconciliation_state` | `PAID` or `UNPAID`                          |
| `email_details`        | Email details used to reconcile the invoice |

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@misc{huang2026agentsfailactdiagnostic,
      title={When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems},
      author={Donghao Huang and Gauri Malwe and Zhaoxia Wang},
      year={2026},
      eprint={2601.16280},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.16280}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The dataset (`dataset/`) and results (`results/`) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
