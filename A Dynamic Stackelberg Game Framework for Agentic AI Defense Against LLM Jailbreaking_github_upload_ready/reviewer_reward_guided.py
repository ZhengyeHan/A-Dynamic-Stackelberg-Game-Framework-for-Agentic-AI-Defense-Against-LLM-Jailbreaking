from __future__ import annotations

import argparse
import contextlib
import csv
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import reviewer_baselines as rb


REPO_ROOT = Path(__file__).resolve().parent
REWARD_GUIDED_SCRIPT = REPO_ROOT / "purple agent RRT guided.py"
REWARD_GUIDED_REPEAT_DIR = REPO_ROOT / "reviewer_reward_guided_repeat_runs"
BASELINE_REPEAT_DIR = REPO_ROOT / "reviewer_repeat_runs"
BASELINE_REPEATED_TABLE = REPO_ROOT / "paper_table_baselines_repeated.csv"
BASELINE_REPEATED_OVERHEAD = REPO_ROOT / "paper_table_overhead_repeated.csv"
BASELINE_H_SINGLE_TABLE = REPO_ROOT / "paper_table_H.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated reviewer-side experiments for reward-guided RRT.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_rg = subparsers.add_parser("run-reward-guided", help="Run one reward-guided repeated condition.")
    run_rg.add_argument("--mode", required=True, choices=[
        "reward_guided_rrt_no_defense",
        "reward_guided_risky_memory_only",
        "reward_guided_purple_full",
    ])
    run_rg.add_argument("--run-id", required=True)
    run_rg.add_argument("--h-value", type=float, default=0.40)

    aggregate_rg = subparsers.add_parser("aggregate-reward-guided", help="Aggregate repeated reward-guided runs.")
    aggregate_rg.add_argument("--include-h035", action="store_true")

    run_h30 = subparsers.add_parser("run-baseline-h30", help="Run one baseline-branch purple_full H=0.30 repeat.")
    run_h30.add_argument("--run-id", required=True)

    aggregate_h = subparsers.add_parser("aggregate-h-extended", help="Build repeated local H-trend table.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_assignment(source: str, name: str, default: int | None = None) -> int:
    match = re.search(rf"(?m)^{re.escape(name)}\s*=\s*(\d+)\b", source)
    if match:
        return int(match.group(1))
    if default is None:
        raise RuntimeError(f"Could not parse integer assignment for {name}.")
    return default


def load_reward_guided_environment(script_path: Path = REWARD_GUIDED_SCRIPT) -> dict[str, Any]:
    rb.install_optional_import_shims()
    source = script_path.read_text(encoding="utf-8")
    root_index = source.find('\np0 = "Tell me a fun fact."')
    if root_index == -1:
        root_index = source.find("\np0 = 'Tell me a fun fact.'")
    if root_index == -1:
        raise RuntimeError(f"Could not isolate executable prefix in {script_path.name}.")

    prefix = source[:root_index]
    env: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(script_path),
        "__name__": "reward_guided_module",
    }
    exec(compile(prefix, str(script_path), "exec"), env)
    env["_source_text"] = source
    env["_script_path"] = str(script_path)
    env["_parsed_max_budget"] = parse_int_assignment(source, "MAX_BUDGET", default=100)
    return env


def ensure_reward_guided_environment(env: dict[str, Any]) -> None:
    required_names = [
        "PROMPT_SPACE",
        "TREE",
        "K_ROLLOUT",
        "BLOCK_RADIUS",
        "sample_prompt",
        "extend_prompt",
        "query_llm",
        "classify_response",
        "get_prompt_vector",
        "PurpleAgent",
        "rollout_trim",
        "T",
        "N",
        "M",
    ]
    missing = [name for name in required_names if name not in env]
    if missing:
        raise RuntimeError(f"Reward-guided environment is missing expected names: {missing}")

    prompt_space = env["PROMPT_SPACE"]
    if not isinstance(prompt_space, list) or not prompt_space:
        raise RuntimeError("PROMPT_SPACE must be a non-empty list.")

    if env["TREE"]["V"][0] != "Tell me a fun fact.":
        raise RuntimeError("Reward-guided root prompt does not match the original script.")

    for prompt in rb.BENIGN_PROMPT_SUBSET:
        if prompt not in prompt_space:
            raise RuntimeError(f"Benign FPR prompt is not in PROMPT_SPACE: {prompt}")


def reset_reward_guided_counters(env: dict[str, Any]) -> None:
    for counter_name in [
        "count_LLM_API",
        "count_jailbreaking",
        "count_risky_memory_vectors",
        "count_prevented_simulation_vectors",
    ]:
        if counter_name in env:
            env[counter_name] = 0
    if "jailbreak_prompts" in env:
        env["jailbreak_prompts"] = []
    if "history" in env:
        env["history"] = {}


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    return 0.0 if len(values) <= 1 else stdev(values)


def safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def run_reward_guided_repeat(mode: str, run_id: str, h_value: float) -> dict[str, Any]:
    env = load_reward_guided_environment(REWARD_GUIDED_SCRIPT)
    ensure_reward_guided_environment(env)
    reset_reward_guided_counters(env)

    per_run_prefix = f"{mode}_run{run_id}"
    summary_path = REWARD_GUIDED_REPEAT_DIR / f"{per_run_prefix}_summary.csv"
    fpr_path = REWARD_GUIDED_REPEAT_DIR / f"{per_run_prefix}_fpr.csv"
    log_path = REWARD_GUIDED_REPEAT_DIR / f"{per_run_prefix}.log"
    REWARD_GUIDED_REPEAT_DIR.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and fpr_path.exists():
        return read_csv(summary_path)[0]

    summary = rb.RunSummary(
        mode=mode,
        h_value=h_value,
        max_budget=env["_parsed_max_budget"],
        root_prompt=env["TREE"]["V"][0],
        prompt_space_size=len(env["PROMPT_SPACE"]),
        benign_subset_size=len(rb.BENIGN_PROMPT_SUBSET),
    )

    original_sample = env["sample_prompt"]
    original_extend = env["extend_prompt"]
    original_query = env["query_llm"]
    original_classify = env["classify_response"]
    phase = {"value": "main"}

    def sample_prompt_wrapper() -> str:
        if phase["value"] == "simulation":
            summary.sample_calls_simulation += 1
        else:
            summary.sample_calls_main += 1
        return original_sample()

    def extend_prompt_wrapper(p_near: str, p_rand: str) -> str:
        if phase["value"] == "simulation":
            summary.extend_calls_simulation += 1
        else:
            summary.extend_calls_main += 1
        return original_extend(p_near, p_rand)

    def query_llm_wrapper(messages: list[dict[str, str]]) -> str:
        if phase["value"] == "simulation":
            summary.target_queries_simulation += 1
        else:
            summary.target_queries_main += 1
        return original_query(messages)

    def classify_response_wrapper(attacker_prompt: str, llm_response: str) -> str:
        if phase["value"] == "simulation":
            summary.classifier_calls_simulation += 1
        else:
            summary.classifier_calls_main += 1
        label = original_classify(attacker_prompt, llm_response)
        if phase["value"] == "main":
            rb.update_label_counts(summary, label)
        return label

    env["sample_prompt"] = sample_prompt_wrapper
    env["extend_prompt"] = extend_prompt_wrapper
    env["query_llm"] = query_llm_wrapper
    env["classify_response"] = classify_response_wrapper

    purple = env["PurpleAgent"](k_rollout=env["K_ROLLOUT"], block_radius=h_value)
    original_should_block = purple.should_block_preemptively
    original_rollout_defense = purple.maybe_defend_by_rollout

    def wrapped_should_block(p_new: str) -> bool:
        reason = rb.current_block_reason(env, purple, p_new) or ""
        blocked = original_should_block(p_new)
        if blocked:
            summary.blocked_pre_query += 1
            summary.blocked_by_risky_memory += int(reason == "risky_memory")
            summary.blocked_by_simulation_memory += int(reason == "simulation_memory")
        return blocked

    def wrapped_rollout_defense(history_for_node: list[dict[str, str]], p_new: str) -> None:
        previous_phase = phase["value"]
        phase["value"] = "simulation"
        try:
            original_rollout_defense(history_for_node, p_new)
        finally:
            phase["value"] = previous_phase

    if mode == "reward_guided_rrt_no_defense":
        purple.should_block_preemptively = lambda p_new: False
        purple.post_incident_learn = lambda p_jb: None
        purple.maybe_defend_by_rollout = lambda history_for_node, p_new: None
    elif mode == "reward_guided_risky_memory_only":
        purple.should_block_preemptively = wrapped_should_block
        purple.maybe_defend_by_rollout = lambda history_for_node, p_new: None
        summary.overlap_note = rb.PURPLE_NO_SIMULATION_NOTE
    elif mode == "reward_guided_purple_full":
        purple.should_block_preemptively = wrapped_should_block
        purple.maybe_defend_by_rollout = wrapped_rollout_defense
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

    root_prompt = env["TREE"]["V"][0]
    history: dict[str, list[dict[str, str]]] = {}
    history[root_prompt] = [{"role": "user", "content": root_prompt}]

    with log_path.open("w", encoding="utf-8") as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
        root_response = env["query_llm"](history[root_prompt])
        history[root_prompt].append({"role": "assistant", "content": root_response})
        try:
            env["rollout_trim"](root_prompt, env["T"], env["N"], env["M"], history, purple)
        except Exception as exc:
            if exc.__class__.__name__ != "BudgetReached":
                raise
            print(f"BudgetReached: {exc}")

    summary.main_queries = summary.target_queries_main
    summary.main_rounds = summary.main_queries + summary.blocked_pre_query
    summary.nearest_neighbor_calls = summary.sample_calls_main
    summary.risky_memory_size = len(purple.risky_memory_vectors)
    summary.simulated_memory_size = len(purple.prevented_simulation_vectors)

    fpr_row, benign_rows = rb.evaluate_benign_subset(
        env,
        summary.mode,
        summary.h_value,
        purple=None if mode == "reward_guided_rrt_no_defense" else purple,
    )
    summary.benign_blocked = fpr_row["benign_blocked"]
    summary.benign_accepted = fpr_row["benign_accepted"]
    summary.benign_blocked_by_risky_memory = fpr_row["benign_blocked_by_risky_memory"]
    summary.benign_blocked_by_simulation_memory = fpr_row["benign_blocked_by_simulation_memory"]
    summary.fpr = fpr_row["fpr"]
    summary.benign_accept_rate = fpr_row["benign_accept_rate"]

    summary_row = asdict(summary)
    write_csv(summary_path, [summary_row])
    write_csv(fpr_path, benign_rows)
    return summary_row


def aggregate_reward_guided(include_h035: bool = False) -> None:
    mode_order = [
        ("reward_guided_rrt_no_defense", "0.40"),
        ("reward_guided_risky_memory_only", "0.40"),
        ("reward_guided_purple_full", "0.40"),
    ]
    if include_h035:
        mode_order.append(("reward_guided_purple_full", "0.35"))

    summary_rows: list[dict[str, str]] = []
    fpr_rows: list[dict[str, str]] = []
    for summary_path in sorted(REWARD_GUIDED_REPEAT_DIR.glob("*_summary.csv")):
        row = read_csv(summary_path)[0]
        stem = summary_path.stem
        run_id = stem.rsplit("_run", 1)[1].replace("_summary", "")
        row["run_id"] = run_id
        row["source"] = "new_repeat_run"
        row["status"] = "completed"
        total_target = float(row["target_queries_main"]) + float(row["target_queries_simulation"])
        total_classifier = float(row["classifier_calls_main"]) + float(row["classifier_calls_simulation"])
        row["total_target_queries"] = fmt(total_target)
        row["total_classifier_calls"] = fmt(total_classifier)
        row["total_model_calls"] = fmt(total_target + total_classifier)
        row["latency_seconds"] = ""
        row["total_tokens"] = ""
        summary_rows.append(row)

        fpr_path = summary_path.with_name(summary_path.name.replace("_summary.csv", "_fpr.csv"))
        for fpr_row in read_csv(fpr_path):
            fpr_row["run_id"] = run_id
            fpr_rows.append(fpr_row)

    summary_rows.sort(key=lambda row: (row["mode"], float(row["h_value"]), row["run_id"]))
    write_csv(REPO_ROOT / "aggregate_reward_guided_repeated_runs.csv", summary_rows)

    def rows_for(mode: str, h_value: str) -> list[dict[str, str]]:
        rows = [row for row in summary_rows if row["mode"] == mode and row["h_value"] == h_value]
        if not rows:
            raise RuntimeError(f"Missing repeated reward-guided summaries for {mode} @ H={h_value}")
        return rows

    repeated_rows: list[dict[str, str]] = []
    overhead_rows: list[dict[str, str]] = []
    for mode, h_value in mode_order:
        rows = rows_for(mode, h_value)
        jailbreaks = [float(row["jailbreaks"]) for row in rows]
        blocks = [float(row["blocked_pre_query"]) for row in rows]
        fprs = [float(row["fpr"]) for row in rows]
        accepts = [float(row["benign_accept_rate"]) for row in rows]
        target_total = [float(row["target_queries_main"]) + float(row["target_queries_simulation"]) for row in rows]
        classifier_total = [float(row["classifier_calls_main"]) + float(row["classifier_calls_simulation"]) for row in rows]
        total_model = [a + b for a, b in zip(target_total, classifier_total)]
        extend_total = [float(row["extend_calls_main"]) + float(row["extend_calls_simulation"]) for row in rows]

        repeated_rows.append(
            {
                "mode": mode,
                "H": h_value,
                "n_runs": str(len(rows)),
                "jailbreak_mean": fmt(safe_mean(jailbreaks)),
                "jailbreak_std": fmt(safe_std(jailbreaks)),
                "jailbreak_min": fmt(safe_min(jailbreaks)),
                "jailbreak_max": fmt(safe_max(jailbreaks)),
                "blocked_pre_query_mean": fmt(safe_mean(blocks)),
                "blocked_pre_query_std": fmt(safe_std(blocks)),
                "FPR_mean": fmt(safe_mean(fprs)),
                "FPR_std": fmt(safe_std(fprs)),
                "benign_accept_rate_mean": fmt(safe_mean(accepts)),
                "benign_accept_rate_std": fmt(safe_std(accepts)),
                "target_queries_mean": fmt(safe_mean(target_total)),
                "target_queries_std": fmt(safe_std(target_total)),
                "total_model_calls_mean": fmt(safe_mean(total_model)),
                "total_model_calls_std": fmt(safe_std(total_model)),
                "latency_mean_seconds": "",
                "total_tokens_mean": "",
                "overlap_note": rows[0].get("overlap_note", ""),
                "note": "latency/tokens not logged in the original reward-guided runner",
            }
        )

        overhead_rows.append(
            {
                "mode": mode,
                "H": h_value,
                "n_runs": str(len(rows)),
                "target_queries_main_mean": fmt(safe_mean([float(row["target_queries_main"]) for row in rows])),
                "target_queries_simulation_mean": fmt(safe_mean([float(row["target_queries_simulation"]) for row in rows])),
                "total_target_queries_mean": fmt(safe_mean(target_total)),
                "total_target_queries_std": fmt(safe_std(target_total)),
                "classifier_calls_main_mean": fmt(safe_mean([float(row["classifier_calls_main"]) for row in rows])),
                "classifier_calls_simulation_mean": fmt(safe_mean([float(row["classifier_calls_simulation"]) for row in rows])),
                "total_classifier_calls_mean": fmt(safe_mean(classifier_total)),
                "total_classifier_calls_std": fmt(safe_std(classifier_total)),
                "total_model_calls_mean": fmt(safe_mean(total_model)),
                "total_model_calls_std": fmt(safe_std(total_model)),
                "extend_calls_main_mean": fmt(safe_mean([float(row["extend_calls_main"]) for row in rows])),
                "extend_calls_simulation_mean": fmt(safe_mean([float(row["extend_calls_simulation"]) for row in rows])),
                "total_extend_calls_mean": fmt(safe_mean(extend_total)),
                "total_extend_calls_std": fmt(safe_std(extend_total)),
                "latency_mean_seconds": "",
                "total_tokens_mean": "",
                "note": "latency/tokens not logged in the original reward-guided runner",
            }
        )

    write_csv(REPO_ROOT / "paper_table_reward_guided_repeated.csv", repeated_rows)
    write_csv(REPO_ROOT / "paper_table_reward_guided_overhead.csv", overhead_rows)


def ensure_baseline_h30_seed() -> None:
    seed_summary = BASELINE_REPEAT_DIR / "purple_full_H_0.30_run01_summary.csv"
    seed_fpr = BASELINE_REPEAT_DIR / "purple_full_H_0.30_run01_fpr.csv"
    seed_rounds = BASELINE_REPEAT_DIR / "purple_full_H_0.30_run01_rounds.csv"
    if seed_summary.exists() and seed_fpr.exists() and seed_rounds.exists():
        return

    legacy_summary = REPO_ROOT / "reviewer_runs" / "purple_full_H_0.30_summary.csv"
    legacy_fpr = REPO_ROOT / "reviewer_runs" / "purple_full_H_0.30_fpr.csv"
    legacy_rounds = REPO_ROOT / "reviewer_runs" / "purple_full_H_0.30_rounds.csv"
    if legacy_summary.exists() and legacy_fpr.exists() and legacy_rounds.exists():
        write_csv(seed_summary, read_csv(legacy_summary))
        write_csv(seed_fpr, read_csv(legacy_fpr))
        write_csv(seed_rounds, read_csv(legacy_rounds))
        return

    env = rb.load_original_environment()
    rb.ensure_original_environment(env)
    summary, benign_rows, _ = rb.run_rrt_mode(
        env,
        mode="purple_full",
        h_value=0.30,
        use_risky_memory=True,
        use_simulation=True,
        per_run_path=seed_rounds,
    )
    write_csv(seed_summary, [asdict(summary)])
    write_csv(seed_fpr, benign_rows)


def run_baseline_h30_repeat(run_id: str) -> dict[str, str]:
    ensure_baseline_h30_seed()
    prefix = f"purple_full_H_0.30_run{run_id}"
    rounds_path = BASELINE_REPEAT_DIR / f"{prefix}_rounds.csv"
    summary_path = BASELINE_REPEAT_DIR / f"{prefix}_summary.csv"
    fpr_path = BASELINE_REPEAT_DIR / f"{prefix}_fpr.csv"
    if summary_path.exists() and fpr_path.exists() and rounds_path.exists():
        return read_csv(summary_path)[0]
    env = rb.load_original_environment()
    rb.ensure_original_environment(env)
    summary, benign_rows, _ = rb.run_rrt_mode(
        env,
        mode="purple_full",
        h_value=0.30,
        use_risky_memory=True,
        use_simulation=True,
        per_run_path=rounds_path,
    )
    summary_row = asdict(summary)
    write_csv(summary_path, [summary_row])
    write_csv(fpr_path, benign_rows)
    return summary_row


def aggregate_h_extended() -> None:
    ensure_baseline_h30_seed()
    h30_paths = [BASELINE_REPEAT_DIR / f"purple_full_H_0.30_run0{i}_summary.csv" for i in range(1, 4)]
    if not all(path.exists() for path in h30_paths):
        missing = [str(path) for path in h30_paths if not path.exists()]
        raise RuntimeError(f"Missing H=0.30 repeated summaries: {missing}")

    h30_rows = [read_csv(path)[0] for path in h30_paths]
    h35_rows = [row for row in read_csv(BASELINE_REPEATED_TABLE) if row["mode"] == "purple_full" and row["H"] == "0.35"]
    if not h35_rows:
        raise RuntimeError("Missing H=0.35 repeated row in paper_table_baselines_repeated.csv")
    h40_rows = read_csv(BASELINE_REPEAT_DIR / "purple_full_H_0.40_repeated_summary.csv")
    if not h40_rows:
        raise RuntimeError("Missing H=0.40 repeated summary row")

    def summarize_h(rows: list[dict[str, str]], h_value: str) -> dict[str, str]:
        jailbreaks = [float(row["jailbreaks"]) for row in rows]
        blocks = [float(row["blocked_pre_query"]) for row in rows]
        fprs = [float(row["fpr"]) for row in rows]
        accepts = [float(row["benign_accept_rate"]) for row in rows]
        return {
            "H": h_value,
            "n_runs": str(len(rows)),
            "jailbreak_mean": fmt(safe_mean(jailbreaks)),
            "jailbreak_std": fmt(safe_std(jailbreaks)),
            "blocked_pre_query_mean": fmt(safe_mean(blocks)),
            "blocked_pre_query_std": fmt(safe_std(blocks)),
            "FPR_mean": fmt(safe_mean(fprs)),
            "benign_accept_rate_mean": fmt(safe_mean(accepts)),
        }

    extended_rows = [
        summarize_h(h30_rows, "0.30"),
        {
            "H": "0.35",
            "n_runs": h35_rows[0]["n_runs"],
            "jailbreak_mean": h35_rows[0]["jailbreak_mean"],
            "jailbreak_std": h35_rows[0]["jailbreak_std"],
            "blocked_pre_query_mean": h35_rows[0]["blocked_pre_query_mean"],
            "blocked_pre_query_std": h35_rows[0]["blocked_pre_query_std"],
            "FPR_mean": h35_rows[0]["FPR_mean"],
            "benign_accept_rate_mean": h35_rows[0]["benign_accept_rate_mean"],
        },
        {
            "H": "0.40",
            "n_runs": h40_rows[0]["n_runs"],
            "jailbreak_mean": h40_rows[0]["jailbreak_mean"],
            "jailbreak_std": h40_rows[0]["jailbreak_std"],
            "blocked_pre_query_mean": h40_rows[0]["blocked_pre_query_mean"],
            "blocked_pre_query_std": h40_rows[0]["blocked_pre_query_std"],
            "FPR_mean": h40_rows[0]["FPR_mean"],
            "benign_accept_rate_mean": h40_rows[0]["benign_accept_rate_mean"],
        },
    ]
    write_csv(REPO_ROOT / "paper_table_H_extended.csv", extended_rows)
    write_csv(BASELINE_REPEAT_DIR / "purple_full_H_0.30_repeated_summary.csv", [extended_rows[0]])


def main() -> None:
    args = parse_args()
    if args.command == "run-reward-guided":
        run_reward_guided_repeat(args.mode, args.run_id, args.h_value)
    elif args.command == "aggregate-reward-guided":
        aggregate_reward_guided(include_h035=args.include_h035)
    elif args.command == "run-baseline-h30":
        run_baseline_h30_repeat(args.run_id)
    elif args.command == "aggregate-h-extended":
        aggregate_h_extended()
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
