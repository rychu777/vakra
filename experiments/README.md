# Experiments

All experiments run on the **same fixed splits** drawn from the public train set, through one script
that records the configuration of every run. This keeps results from different people and different
agent variants comparable item by item.

## One-time setup

```bash
pip install -e ".[init]" -r requirements_benchmark.txt   # pins mcp<2; with mcp 2.x the agent silently loops
python benchmark_setup.py --download-data
docker compose up -d
python scripts/make_dev_split.py                          # rebuilds data/dev and data/dev_mini from experiments/splits/
```

| Split | Items per domain | Capability 1 | 2 | 3 | 4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| `dev` (numbers for the paper) | ≤10 | 319 | 387 | 185 | 320 | 1211 |
| `dev_mini` (quick checks while developing) | ≤3 | 97 | 119 | 74 | 106 | 396 |

`dev_mini` is a subset of `dev`. The manifests in `experiments/splits/` hold only UUIDs and are committed;
the data itself stays in the git-ignored `data/`. **Do not resample** (`--resample`): it changes the items
and makes earlier results incomparable.

**`talkingdata` is left out of capability 1** (`EXCLUDED_DOMAINS` in `scripts/run_experiment.py`), so an
experiment on `dev` covers 309 items there (1201 in total). Before each item, the capability 1 server loads
the item's tables into memory; `talkingdata` has a table of 32.5M rows, and the server runs out of memory in
Docker and is killed, whatever the agent does. The domain stays in the split and can be brought back on a
machine with more memory. Capabilities 2 and 4 keep their `talkingdata` items.

VAKRA has no official dev split, only train and test. If a method uses train examples (few-shot prompts,
example retrieval, prompt tuning), keep the `dev` items out of that pool.

## Running an experiment

```bash
python scripts/run_experiment.py --name baseline_qwen25-7b --model qwen2.5:7b \
    --ollama-base-url http://127.0.0.1:11434
```

- `--name` names the experiment; everything goes to `output/experiments/<name>/`.
- Defaults: all four capabilities, the `dev` split, `--max-iterations 10`, `--top-k-tools 128`,
  `--timeout 300`, temperature 0.
- `--split dev_mini --capabilities 2` gives a quick run on one capability.
- Extra `benchmark_runner.py` flags go after `--`, e.g. `-- --phoenix`.
- `--dry-run` prints what would run and changes nothing.

**Resuming:** run the same command again. The runner skips domains that already have a predictions file.
Domains where **every** item failed (e.g. because Ollama was down) are listed in a warning; delete their
files to run them again.

**The configuration is enforced:** if the model or agent settings differ from an existing experiment,
the script refuses to start. A new configuration needs a new `--name`. This includes the model's Ollama
digest: the same tag (e.g. `qwen2.5:7b`) pulled at another time or in another quantization is a different
model, so everyone working on one experiment needs the same digest (`ollama list` shows its first 12 characters).

## Sharing the work

Everyone runs the **same** experiment (same name and flags) on their own domains, without evaluation:

```bash
python scripts/run_experiment.py --name baseline_qwen25-7b --model qwen2.5:7b --capabilities 2 \
    --domain books --domain hockey --skip-eval
```

Then the `predictions/<domain>.json` files are copied into one directory and one person evaluates:

```bash
python scripts/run_experiment.py --name baseline_qwen25-7b --eval-only
```

## Evaluation

The judge is a model on the Ollama server given by `--ollama-base-url`. By default it is `gpt-oss:20b`,
the smaller model from the family of the organizers' judge, on the same server as the agent:

```bash
ollama pull gpt-oss:20b        # once, about 14 GB
python scripts/run_experiment.py --name baseline_qwen25-7b --eval-only
```

Once an agent is ready, it is evaluated with the organizers' judge model, `gpt-oss:120b`, on the GPU server:

```bash
python scripts/run_experiment.py --name <experiment> --eval-only --judge-model gpt-oss:120b \
    --ollama-base-url http://<gpu-server>:11434
```

- Keep the judge different from the agent model, so the agent does not grade its own answers.
- Compare only scores from the same judge. Each judge's results go to their own directory
  (`eval/<judge>/`), and `summary.json` lists them separately.
- `--re-eval` recomputes the scores, e.g. after predictions were replaced.

**A failed judge call scores 0.** The evaluator does not stop when the judge fails (server down,
unparsable output); it scores that turn 0. The summary counts these errors per judge. If there are any,
fix the cause and run again with `--eval-only --re-eval`.

## Output

```
output/experiments/<name>/
  config.json        configuration and run history (git commit, mcp and Ollama versions, host, domains, judge)
  summary.json       per capability: items, successes, errors, cost (time, LLM calls, tokens), scores per judge
  capability_N_.../
    predictions/     predictions in the submission format (+ *_tools.json, run.log); each record
                     also has duration_s, llm_calls, input_tokens and output_tokens
    runner.log       full agent log
    eval/<judge>/    per-item evaluator results (results.json) and eval.log, e.g. eval/gpt-oss-20b/
```

Error analysis: `scripts/error_analysis_script.py` (inputs: `data/dev/<capability>/input`, `.../output`,
`predictions/`, `eval/<judge>/results.json`).
