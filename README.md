# smol-llm

A tiny GPT-style causal language model built from scratch for local experiments.

This repo is intentionally small and plain. The goal is to test tokenizer choices, dataset quality, model sizes, and training behavior without turning the project into a framework-shaped swamp.

## First setup

```bash
cd /home/dsmason321/repos/smol-llm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Stream FineNews text

This streams FineNews English rows from Hugging Face and writes only article text locally.

```bash
python scripts/stream_finenews_text.py \
  --start-month 2021-01 \
  --end-month 2025-12
```

Completed months are skipped automatically. Use `--overwrite` to rebuild them.

## Build a small local sample

```bash
python scripts/sample_corpus.py \
  --input-dir data/processed/finenews_text \
  --output data/processed/sample_100mb.txt \
  --target-mb 100
```

## Train tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/processed/sample_100mb.txt \
  --output-dir data/tokenizer \
  --vocab-size 16000
```

## Prepare binary training data

```bash
python scripts/prepare_data.py \
  --input data/processed/sample_100mb.txt \
  --tokenizer data/tokenizer/tokenizer.json \
  --out-dir data/processed
```

## Smoke train

```bash
python scripts/train.py --config configs/smoke.yaml
```

## Generate

```bash
python scripts/generate.py \
  --config configs/smoke.yaml \
  --checkpoint checkpoints/latest.pt \
  --prompt "The government announced"
```

## Repo layout

```text
configs/              model and run configs
data/                 local data, mostly gitignored
scripts/              command line tools
smol_llm/             core package
tests/                smoke tests
checkpoints/          local model checkpoints, gitignored
logs/                 local logs, gitignored
```
