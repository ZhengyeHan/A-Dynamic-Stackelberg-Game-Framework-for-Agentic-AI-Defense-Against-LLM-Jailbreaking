# Reproducibility Notes

## Prerequisites

1. Create a Python environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Fill in the credentials required for the scripts you plan to run.
4. Optional: replace `PROMPT_SPACE_PATH` with an approved restricted prompt-set JSON file if you need a closer approximation to the original private experiments.

## Public baseline workflow

Regenerate the baseline summary tables:

```bash
python reviewer_baselines.py
```

This writes the baseline aggregate tables at the repository root, including:

- `paper_table_baselines.csv`
- `paper_table_fpr.csv`
- `paper_table_H.csv`
- `paper_table_overhead.csv`

## Public repeated reward-guided workflow

Run three repeats per condition:

```bash
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_rrt_no_defense --run-id 01 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_rrt_no_defense --run-id 02 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_rrt_no_defense --run-id 03 --h-value 0.40

python reviewer_reward_guided.py run-reward-guided --mode reward_guided_risky_memory_only --run-id 01 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_risky_memory_only --run-id 02 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_risky_memory_only --run-id 03 --h-value 0.40

python reviewer_reward_guided.py run-reward-guided --mode reward_guided_purple_full --run-id 01 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_purple_full --run-id 02 --h-value 0.40
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_purple_full --run-id 03 --h-value 0.40
```

Aggregate the repeated runs:

```bash
python reviewer_reward_guided.py aggregate-reward-guided
```

This produces the retained reward-guided summary tables:

- `paper_table_reward_guided_repeated.csv`
- `paper_table_reward_guided_overhead.csv`

Optional local H-extension workflow:

```bash
python reviewer_reward_guided.py run-baseline-h30 --run-id 02
python reviewer_reward_guided.py run-baseline-h30 --run-id 03
python reviewer_reward_guided.py aggregate-h-extended
```

## Output notes

- Re-running the reviewer scripts recreates `reviewer_runs`, `reviewer_repeat_runs`, and `reviewer_reward_guided_repeat_runs`.
- Those directories are intentionally ignored in `.gitignore` because they contain generated artifacts rather than source files.
- The repository keeps a small set of `paper_table_*.csv` files as public-safe example outputs, but omits intermediate aggregate CSVs and bulky figure assets.
- The public default prompt library is redacted, so the exact paper numbers from the private prompt set should not be expected unless you supply an approved replacement via `PROMPT_SPACE_PATH`.
