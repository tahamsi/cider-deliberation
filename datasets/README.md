# Datasets

`download_preprocess.py` downloads supported Hugging Face datasets and writes normalized JSONL files.

Supported names:

- `mmlu_pro`
- `gsm8k`
- `math500`
- `truthfulqa_mc`
- `strategyqa`
- `medqa`
- `medmcqa`
- `aime2024`

Use `--seed` with `--max_examples` for reproducible sampling. Loader failures are raised with dataset-specific context.
