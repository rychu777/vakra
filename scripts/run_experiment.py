#!/usr/bin/env python3
"""Run a VAKRA experiment (agent run and evaluation) on a fixed split.

An experiment is a named configuration (model, agent settings, split). All of its files
live in output/experiments/<name>/, so a run can be resumed, split across people by
domain, and compared with other experiments:

    config.json                    experiment config and a provenance record per invocation
    summary.json                   item counts and scores per capability
    <capability>/predictions/      runner output in the submission format, plus run.log
    <capability>/runner.log        full runner output
    <capability>/eval/<judge>/     results.json and eval.log, kept apart per judge

Examples:
    python scripts/run_experiment.py --name baseline_qwen25-7b --model qwen2.5:7b \\
        --ollama-base-url http://127.0.0.1:11435
    python scripts/run_experiment.py --name smoke --split dev_mini --capabilities 2 \\
        --model qwen2.5:7b --skip-eval
    python scripts/run_experiment.py --name baseline_qwen25-7b --eval-only
    python scripts/run_experiment.py --name baseline_qwen25-7b --eval-only --judge-model gpt-oss:120b \\
        --ollama-base-url http://<gpu-server>:11434
    python scripts/run_experiment.py --name traced --model qwen2.5:7b -- --phoenix

Running the same command again resumes: the runner skips domains that already have
predictions and the evaluator skips domains already present in results.json.
The judge is an Ollama model (--judge-model, default gpt-oss:20b) on the same server as the agent.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG_PATH = REPO_ROOT / "benchmark" / "mcp_connection_config.yaml"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "gpt-oss:20b"
JUDGE_ERROR_MARKER = "Judge Error"
COST_FIELDS = ("duration_s", "llm_calls", "input_tokens", "output_tokens")


class Capability(NamedTuple):
    directory: str
    evaluator_name: str


CAPABILITIES = {
    1: Capability("capability_1_bi_apis", "capability_bi_apis"),
    2: Capability("capability_2_dashboard_apis", "capability_dashboard_apis"),
    3: Capability("capability_3_multihop_reasoning", "capability_multihop_reasoning"),
    4: Capability("capability_4_multiturn", "capability_multiturn"),
}

# Domains left out of every experiment, by capability id. They stay in the split, so they can be
# brought back on a machine with more memory. In capability 1, get_data for talkingdata loads joins
# of app_events (32.5M rows) into memory and the MCP server is killed by the out-of-memory killer
# of the Docker VM at about 22 GB, so these items fail whatever the agent does.
EXCLUDED_DOMAINS = {1: ["talkingdata"]}

RUNNER_PREFIX_PATTERN = re.compile(r"^\[capability_\d\]\s*")
PROGRESS_PATTERN = re.compile(r"# DOMAIN:|\[\d+/\d+\] Query:|Status: error|Total items:|Successful:|Failed:|^Error:")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    arguments = sys.argv[1:]
    runner_extra_args: list[str] = []
    if "--" in arguments:
        separator_index = arguments.index("--")
        runner_extra_args = arguments[separator_index + 1:]
        arguments = arguments[:separator_index]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Experiment name (directory under output/experiments/)")
    parser.add_argument(
        "--capabilities",
        type=int,
        nargs="+",
        choices=sorted(CAPABILITIES),
        default=sorted(CAPABILITIES),
        help="Capability ids to run (default: all)",
    )
    parser.add_argument("--split", default="dev", help="Split under data/ made by make_dev_split.py (default: dev)")
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Run only these domains (repeatable); capabilities without them are skipped",
    )
    parser.add_argument("--provider", default="ollama", help="LLM provider for benchmark_runner.py (default: ollama)")
    parser.add_argument("--model", help="Model name (required unless --eval-only)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations per query, 0 means the runner default (default: 10)",
    )
    parser.add_argument("--top-k-tools", type=int, default=128, help="Tool shortlist size (default: 128)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Agent timeout per query in seconds (default: 300)")
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help=f"Ollama server for the agent and the local judge (default: OLLAMA_BASE_URL or {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model on the Ollama server (default: {DEFAULT_JUDGE_MODEL})",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--skip-eval", action="store_true", help="Only run the agent")
    mode_group.add_argument("--eval-only", action="store_true", help="Only evaluate existing predictions")
    parser.add_argument("--re-eval", action="store_true", help="Delete each evaluated capability's results.json first")
    parser.add_argument("--force", action="store_true", help="Continue even if the config differs from config.json")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without changing anything")

    args = parser.parse_args(arguments)
    if not args.eval_only and not args.model:
        parser.error("--model is required unless --eval-only is given")
    return args, runner_extra_args


def judge_id(args: argparse.Namespace) -> str:
    """Directory name for one judge's results, e.g. gpt-oss-20b."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", args.judge_model)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def domain_files(directory: Path) -> list[Path]:
    """Domain JSON files in a directory, without the runner's *_tools.json sidecars."""
    return sorted(path for path in directory.glob("*.json") if not path.stem.endswith("_tools"))


def selected_domains(capability_id: int, input_dir: Path, requested_domains: set[str]) -> list[str]:
    """Domains of the split to run, without EXCLUDED_DOMAINS, limited to --domain if given."""
    available_domains = {path.stem for path in domain_files(input_dir)} - set(EXCLUDED_DOMAINS.get(capability_id, []))
    if requested_domains:
        return sorted(requested_domains & available_domains)
    return sorted(available_domains)


def count_items(input_dir: Path, domains: set[str]) -> int:
    return sum(
        len(read_json(path)) for path in domain_files(input_dir) if not domains or path.stem in domains
    )


def ollama_url(args: argparse.Namespace) -> str:
    return args.ollama_base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL


def container_names() -> dict[int, str]:
    tasks = yaml.safe_load(MCP_CONFIG_PATH.read_text(encoding="utf-8"))["tasks"]
    return {int(capability_id): task["container_name"] for capability_id, task in tasks.items()}


def is_container_running(container: str) -> bool:
    try:
        inspection = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return inspection.returncode == 0 and inspection.stdout.strip() == "true"


def ollama_get(base_url: str, path: str) -> dict:
    """Read an Ollama API endpoint. Used only for metadata, never loads a model."""
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=5) as response:
        return json.load(response)


def ollama_models(base_url: str) -> dict[str, dict]:
    return {entry["name"]: entry for entry in ollama_get(base_url, "/api/tags").get("models", [])}


def find_model(models: dict[str, dict], name: str) -> dict | None:
    return models.get(name) or models.get(f"{name}:latest")


def ollama_model_digest(base_url: str, name: str) -> str | None:
    """The digest pins the exact weights and quantization behind a tag such as qwen2.5:7b."""
    try:
        entry = find_model(ollama_models(base_url), name)
    except (OSError, ValueError):
        return None
    return entry["digest"] if entry else None


def ollama_version(base_url: str) -> str | None:
    try:
        return ollama_get(base_url, "/api/version").get("version")
    except (OSError, ValueError):
        return None


def collect_problems(args: argparse.Namespace, split_dir: Path, evaluate: bool) -> list[str]:
    problems = []
    mcp_version = importlib.metadata.version("mcp")
    if int(mcp_version.split(".")[0]) >= 2:
        problems.append(
            f"mcp {mcp_version} is installed, but the tool wrapper needs mcp 1.x "
            "(with 2.x the tools lose their parameter schemas); run: pip install \"mcp<2\""
        )

    containers = container_names()
    for capability_id in args.capabilities:
        input_dir = split_dir / CAPABILITIES[capability_id].directory / "input"
        if not domain_files(input_dir):
            problems.append(f"no split data in {input_dir}; run: python scripts/make_dev_split.py")
        if not is_container_running(containers[capability_id]):
            problems.append(f"container '{containers[capability_id]}' is not running; run: docker compose up -d")

    required_ollama_models = []
    if not args.eval_only and args.provider == "ollama":
        required_ollama_models.append(args.model)
    if evaluate:
        required_ollama_models.append(args.judge_model)
    if required_ollama_models:
        base_url = ollama_url(args)
        try:
            available_models = ollama_models(base_url)
        except (OSError, ValueError) as error:
            problems.append(
                f"Ollama at {base_url} is not reachable ({error.__class__.__name__}); start it or fix --ollama-base-url"
            )
        else:
            for model in dict.fromkeys(required_ollama_models):
                if find_model(available_models, model) is None:
                    problems.append(f"model '{model}' is not available on Ollama at {base_url}; run: ollama pull {model}")
    return problems


def experiment_config(args: argparse.Namespace, runner_extra_args: list[str]) -> dict:
    manifest_path = REPO_ROOT / "experiments" / "splits" / f"{args.split}.json"
    return {
        "split": args.split,
        "split_manifest_sha256": file_sha256(manifest_path),
        "provider": args.provider,
        "model": args.model,
        "model_digest": ollama_model_digest(ollama_url(args), args.model)
        if args.provider == "ollama" and args.model else None,
        "temperature": args.temperature,
        "max_iterations": args.max_iterations or None,
        "top_k_tools": args.top_k_tools,
        "agent_timeout_s": args.timeout,
        "runner_extra_args": runner_extra_args,
        "excluded_domains": {
            CAPABILITIES[capability_id].directory: domains for capability_id, domains in EXCLUDED_DOMAINS.items()
        },
    }


def config_differences(stored_config: dict, new_config: dict) -> dict[str, tuple]:
    keys = sorted(stored_config.keys() | new_config.keys())
    return {
        key: (stored_config.get(key), new_config.get(key))
        for key in keys
        if stored_config.get(key) != new_config.get(key)
    }


def record_invocation(config_path: Path, stored_experiment: dict | None, args: argparse.Namespace,
                      config: dict, evaluate: bool) -> None:
    if args.eval_only:
        mode = "eval-only"
    elif evaluate:
        mode = "run+eval"
    else:
        mode = "run"

    invocation = {
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "capabilities": args.capabilities,
        "domains": sorted(args.domain) if args.domain else "all",
        "config": config,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain", "--untracked-files=no")),
        "mcp_version": importlib.metadata.version("mcp"),
        "python": platform.python_version(),
        "host": platform.node(),
        "ollama_base_url": ollama_url(args),
        "ollama_version": ollama_version(ollama_url(args)),
        "judge_model": args.judge_model if evaluate else None,
        "judge_model_digest": ollama_model_digest(ollama_url(args), args.judge_model) if evaluate else None,
    }
    experiment = stored_experiment or {"experiment": args.name, "config": config, "runs": []}
    experiment["runs"].append(invocation)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")


def capability_env(capability_id: int, capability_split_dir: Path, args: argparse.Namespace, config: dict) -> dict:
    env = dict(os.environ)
    env[f"CAPABILITY_{capability_id}_DIR"] = str(capability_split_dir)
    env["AGENT_TIMEOUT_SECONDS"] = str(config["agent_timeout_s"])
    env["JUDGE_BACKEND"] = "ollama"
    env["JUDGE_MODEL"] = args.judge_model
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if args.ollama_base_url:
        env["OLLAMA_BASE_URL"] = args.ollama_base_url
    return env


def run_logged(command: list[str], env: dict, log_path: Path, label: str) -> int:
    """Run a command, append its whole output to log_path, print progress lines. Return the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                log_file.write(line)
                message = RUNNER_PREFIX_PATTERN.sub("", line.strip())
                if message and not message.startswith("{") and PROGRESS_PATTERN.search(message):
                    print(f"  [{label}] {message[:160]}", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def run_agent(capability_id: int, config: dict, domains: list[str], capability_split_dir: Path,
              capability_output_dir: Path, env: dict, dry_run: bool) -> bool:
    command = [
        sys.executable, "benchmark_runner.py",
        "--capability_id", str(capability_id),
        "--provider", config["provider"],
        "--model", config["model"],
        "--temperature", str(config["temperature"]),
        "--top-k-tools", str(config["top_k_tools"]),
        "--output", str(capability_output_dir / "predictions"),
    ]
    if config["max_iterations"]:
        command += ["--max-iterations", str(config["max_iterations"])]
    for domain in domains:
        command += ["--domain", domain]
    command += config["runner_extra_args"]

    item_count = count_items(capability_split_dir / "input", set(domains))
    print(f"\n[capability {capability_id}] agent run on {item_count} items")
    if dry_run:
        print(f"  CAPABILITY_{capability_id}_DIR={capability_split_dir}")
        print(f"  {' '.join(command)}")
        return True

    log_path = capability_output_dir / "runner.log"
    exit_code = run_logged(command, env, log_path, f"capability {capability_id}")
    if exit_code != 0:
        print(f"  runner exited with code {exit_code}; see {log_path.relative_to(REPO_ROOT)}")
    return exit_code == 0


def run_evaluator(capability_id: int, domains: list[str], capability_split_dir: Path,
                  capability_output_dir: Path, env: dict, args: argparse.Namespace) -> bool:
    predictions_dir = capability_output_dir / "predictions"
    judge_dir = capability_output_dir / "eval" / judge_id(args)
    results_path = judge_dir / "results.json"
    command = [
        sys.executable, "evaluator/evaluator.py",
        "--capability_name", CAPABILITIES[capability_id].evaluator_name,
        "--gt_root", str(capability_split_dir / "output"),
        "--pred_root", str(predictions_dir),
        "--output", str(results_path),
    ]
    if domains:
        command += ["--domains", *domains]

    print(f"\n[capability {capability_id}] evaluation with the judge {args.judge_model}")
    if args.dry_run:
        print(f"  {' '.join(command)}")
        return True
    if not predictions_dir.exists():
        print("  no predictions yet, skipped")
        return True
    if args.re_eval and results_path.exists():
        results_path.unlink()

    judge_dir.mkdir(parents=True, exist_ok=True)
    log_path = judge_dir / "eval.log"
    exit_code = run_logged(command, env, log_path, f"evaluation {capability_id}")
    if exit_code != 0:
        print(f"  evaluator exited with code {exit_code}; see {log_path.relative_to(REPO_ROOT)}")
    return exit_code == 0


def prediction_stats(predictions_dir: Path) -> dict:
    stats = {"predicted": 0, "success": 0, "error": 0, "failed_domains": []}
    stats.update(dict.fromkeys(COST_FIELDS, 0))
    if not predictions_dir.exists():
        return stats
    for path in domain_files(predictions_dir):
        records = read_json(path)
        error_count = sum(record.get("status") != "success" for record in records)
        stats["predicted"] += len(records)
        stats["success"] += len(records) - error_count
        stats["error"] += error_count
        if records and error_count == len(records):
            stats["failed_domains"].append(path.stem)
        for cost_field in COST_FIELDS:
            stats[cost_field] += sum(record.get(cost_field, 0) for record in records)
    return stats


def build_summary(experiment_dir: Path, split_dir: Path) -> dict:
    summary = {}
    for capability_id, capability in CAPABILITIES.items():
        capability_output_dir = experiment_dir / capability.directory
        if not capability_output_dir.exists():
            continue
        input_dir = split_dir / capability.directory / "input"
        row = {"expected": count_items(input_dir, set(selected_domains(capability_id, input_dir, set())))}
        row.update(prediction_stats(capability_output_dir / "predictions"))
        row["scores"] = {}
        for results_path in sorted((capability_output_dir / "eval").glob("*/results.json")):
            results_text = results_path.read_text(encoding="utf-8")
            results = json.loads(results_text).get("summary", {})
            row["scores"][results_path.parent.name] = {
                "evaluated": results.get("n_samples"),
                "correct": results.get("n_correct"),
                "score": results.get("mean_dialogue_score"),
                "missing_predictions": results.get("n_missing_predictions"),
                "judge_errors": results_text.count(JUDGE_ERROR_MARKER),
            }
        summary[capability.directory] = row
    return summary


def print_summary(summary: dict, summary_path: Path) -> None:
    print(f"\n{'capability':34} {'expected':>8} {'predicted':>9} {'success':>7} {'error':>5} "
          f"{'s/item':>7} {'calls/item':>10} {'in tok/item':>11} {'out tok/item':>12}")
    for capability_dir, row in summary.items():
        items = max(row["predicted"], 1)
        print(f"{capability_dir:34} {row['expected']:8} {row['predicted']:9} {row['success']:7} {row['error']:5} "
              f"{row['duration_s'] / items:7.1f} {row['llm_calls'] / items:10.1f} "
              f"{row['input_tokens'] / items:11.0f} {row['output_tokens'] / items:12.0f}")

    judges = sorted({judge for row in summary.values() for judge in row["scores"]})
    for judge in judges:
        print(f"\nJudge {judge}:")
        print(f"  {'capability':34} {'evaluated':>9} {'score':>7} {'judge errors':>12}")
        for capability_dir, row in summary.items():
            scores = row["scores"].get(judge)
            if scores is None:
                continue
            score = scores["score"]
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            print(f"  {capability_dir:34} {scores['evaluated'] or 0:9} {score_text:>7} {scores['judge_errors']:12}")

    for capability_dir, row in summary.items():
        if row["failed_domains"]:
            print(f"\nWARNING {capability_dir}: every item failed in {row['failed_domains']}. "
                  f"Resuming does not retry them; if it was an infrastructure problem, delete those files "
                  f"from {capability_dir}/predictions/ and run again.")
        for judge, scores in row["scores"].items():
            if scores["judge_errors"]:
                print(f"\nWARNING {capability_dir}, judge {judge}: {scores['judge_errors']} judge calls failed "
                      f"and were scored 0 (server down, unparsable output). Fix the cause and "
                      f"run again with --eval-only --re-eval.")
    print(f"\nSummary: {summary_path.relative_to(REPO_ROOT)}")


def main() -> int:
    args, runner_extra_args = parse_args()
    experiment_dir = REPO_ROOT / "output" / "experiments" / args.name
    config_path = experiment_dir / "config.json"
    stored_experiment = read_json(config_path) if config_path.exists() else None
    if args.eval_only and stored_experiment:
        args.split = stored_experiment["config"]["split"]
    split_dir = REPO_ROOT / "data" / args.split

    evaluate = not args.skip_eval
    problems = collect_problems(args, split_dir, evaluate)
    if problems and not args.dry_run:
        print("Cannot start:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    for problem in problems:
        print(f"WARNING (dry run): {problem}")

    config = experiment_config(args, runner_extra_args)
    if args.eval_only and stored_experiment:
        config = stored_experiment["config"]
    elif stored_experiment:
        differences = config_differences(stored_experiment["config"], config)
        if differences:
            print(f"The config differs from {config_path.relative_to(REPO_ROOT)}:")
            for key, (stored_value, new_value) in differences.items():
                print(f"  {key}: {stored_value!r} -> {new_value!r}")
            if not args.force:
                print("Use a new --name, or --force to mix configurations deliberately.")
                return 2

    print(f"Experiment {args.name} -> {experiment_dir.relative_to(REPO_ROOT)}")
    print(f"  split={config['split']} provider={config['provider']} model={config['model']} "
          f"max_iterations={config['max_iterations']} top_k_tools={config['top_k_tools']} "
          f"timeout={config['agent_timeout_s']}s")
    if not args.dry_run:
        record_invocation(config_path, stored_experiment, args, config, evaluate)

    requested_domains = set(args.domain) if args.domain else set()
    failed_steps = 0
    for capability_id in args.capabilities:
        capability = CAPABILITIES[capability_id]
        capability_split_dir = split_dir / capability.directory
        capability_output_dir = experiment_dir / capability.directory
        excluded_domains = EXCLUDED_DOMAINS.get(capability_id)
        if excluded_domains:
            print(f"\n[capability {capability_id}] excluded domains: {', '.join(excluded_domains)}")
        domains = selected_domains(capability_id, capability_split_dir / "input", requested_domains)
        if not domains:
            print(f"\n[capability {capability_id}] no domains to run here, skipped")
            continue

        env = capability_env(capability_id, capability_split_dir, args, config)
        if not args.eval_only:
            agent_ok = run_agent(
                capability_id, config, domains, capability_split_dir, capability_output_dir, env, args.dry_run
            )
            if not agent_ok:
                failed_steps += 1
                continue
        if evaluate:
            evaluation_ok = run_evaluator(
                capability_id, domains, capability_split_dir, capability_output_dir, env, args
            )
            if not evaluation_ok:
                failed_steps += 1

    if args.dry_run:
        return 0

    summary = build_summary(experiment_dir, split_dir)
    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, summary_path)
    return 1 if failed_steps else 0


if __name__ == "__main__":
    sys.exit(main())
