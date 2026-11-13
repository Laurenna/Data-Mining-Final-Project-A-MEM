# A-Mem Reproduction and Full-History Control

Data Mining Homework 3: a final project proposal on memory selection for conversational agents.

## Experiment scope

- Ten LoCoMo conversations, retaining all 5,882 utterances.
- Twenty uniformly sampled questions per conversation: 200 questions total.
- Question-selection seed: 20261110.
- Both methods use `deepseek-flash` through the official DeepSeek API, with thinking disabled.
- A-Mem uses k=10 initial memories followed by the evaluated implementation's link expansion.
- The control supplies the complete corresponding conversation, retaining the same question instructions and category-5 option order.
- F1 uses sets of normalized tokens. BLEU-1 uses NLTK tokenization and smoothing method 1.
- Category 5 is scored against `Not mentioned in the conversation`.

This is a sampled method reproduction under modified settings, not exact replication of all published results. MemoryBank, improved ranking, deduplication and ablations are future work, not completed experiments.

## Results

| Method | F1 | BLEU-1 |
|---|---:|---:|
| A-Mem | 0.4860 | 0.4302 |
| Full-history control | 0.6131 | 0.5607 |

These are average overlap scores, not exact-answer accuracy. The full-history control scores higher overall; this does not establish why A-Mem performed worse. Results by category and conversation are in `results/comparison/comparison.csv`.

## Repository structure

- `amem/`: adapted public A-Mem source, DeepSeek adapter, checkpointed runner, and sampled evaluation runner.
- `full_history/`: paired full-history runner, its separate adapter, diagnostic wrapper, and offline tests.
- `analysis/`: independent scoring audit and paired comparison scripts.
- `results/`: compressed predictions, original manifests, summary scores, and comparison tables.
- `report/`: NeurIPS-format LaTeX source, style file, and PDF.
- `data/`: dataset placement instructions; dataset not bundled.

The two adapters are separate versions and must stay in their respective directories. Do not combine them. Historical manifests retain original local paths and hashes as provenance; those paths are not required to run the commands below. Memory checkpoints and complete API ledgers are retained locally and are not included in this repository.

## Environment

Commands below are run from the repository root. The experiments used Python 3.12 and NLTK 3.10.4. Other dependencies are listed, but not all were version-pinned; exact environment replication is not claimed.

For offline result checking and the full-history runner:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r full_history/requirements.txt
python -m nltk.downloader punkt punkt_tab
python unpack_results.py
```

For A-Mem, additionally install:

```bash
python -m pip install -r amem/requirements.txt -r full_history/requirements.txt
```

The upstream A-Mem modules can download embedding models and NLTK resources when imported. The full-history control does not use an embedding model.

## Inspect saved results without API calls

First unpack the compressed answer files with `python unpack_results.py`. To reproduce the paired tables (no dataset or API key required):

```bash
python analysis/compare_results.py \
  --amem results/amem/answers.jsonl \
  --full-history results/full_history/answers.jsonl \
  --output-dir outputs/comparison
```

This validates matching question sets and prompt instructions, then summarizes saved scores. To independently recompute the scores, place the matching dataset in `data/locomo10.json` and run:

```bash
python analysis/audit_evaluation.py \
  --answers results/amem/answers.jsonl \
  --dataset data/locomo10.json \
  --output-dir outputs/amem_audit

python analysis/audit_evaluation.py \
  --answers results/full_history/answers.jsonl \
  --dataset data/locomo10.json \
  --output-dir outputs/full_history_audit
```

## Run new experiments (paid requests only with --run)

Set `DEEPSEEK_API_KEY` in your terminal environment. Never put the actual key in code, README, or committed files. The scripts do not load a repository `.env` automatically.

A-Mem offline plan:

```bash
python amem/run_sampled_locomo.py \
  --source-dir amem \
  --dataset data/locomo10.json \
  --output-dir outputs/amem_sampled200 \
  --max-total-calls 1000
```

Add `--run` to start paid requests. Increase the cumulative cap explicitly to continue after it is reached. A cap of 1000 is a starting budget, not enough to build all ten histories. This fresh-run command does not reuse the earlier pilot checkpoint. Full-history construction is retained; only QA is sampled. Preserve the output directory and repeat the command to resume completed checkpoints.

The control uses the included original A-Mem answers and selection manifest, so it can run without rebuilding A-Mem memories:

```bash
python full_history/run_full_history.py \
  --dataset data/locomo10.json \
  --amem-dir results/amem \
  --output-dir outputs/full_history_sampled200 \
  --max-calls 300
```

Add `--run --max-new-questions 5` for a five-question paid smoke run. Then omit `--max-new-questions` to complete the remaining questions. Use `full_history/diagnose_full_history.py` in place of the runner if raw response diagnostics are needed. This preserves request content and validation rules.

Each completed control answer is saved immediately. Ctrl+C preserves completed work. A failed/interrupted request may still incur charges. Limits count cumulative requests, including retries, rather than money. No answer is selected based on its score. Nonzero temperatures and provider model aliases mean exact numerical reruns are not guaranteed.

## Offline tests

```bash
python -m unittest discover -s full_history -p 'test_full_history.py'
python amem/test_sampled_runner.py amem data/locomo10.json
```

## Report

Upload `report/main.tex` and `report/neurips_2025.sty` to Overleaf, or compile with `latexmk -pdf main.tex` inside `report/`. The template uses preprint mode. Fill in the author field before final submission; this draft deliberately does not invent a student ID. Submit the single PDF and this repository's actual URL through the course submission system.

## Sources and attribution

A-Mem source was adapted from [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory). Its MIT copyright and license are preserved in `amem/LICENSE`. Adaptations include provider integration, validation, checkpointing, sampled evaluation and local evaluation handling. Do not describe the entire system as independently written from scratch.

- Xu et al. (2025), [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110).
- Maharana et al. (2024), [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753).
- LoCoMo data: [official repository](https://github.com/snap-research/locomo), with the evaluated file also distributed in the A-Mem repository. Respect the source dataset's terms.

Codex/ChatGPT assisted with explanation, coding, debugging, evaluation analysis, and report drafting.
