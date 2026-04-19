# Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking

This repository is the final public-release code package for our research on Stackelberg-style defenses against LLM jailbreak attacks. It keeps the core experiment scripts, reviewer-side evaluation pipeline, and minimal public-safe data/templates needed to understand and run the method without exposing private infrastructure or unsafe raw artifacts, and it is released under the MIT License.

The original internal attack prompt library was not suitable for open release. By default, the code uses `data/public_prompt_space.json`, a redacted public-safe substitute. This means the repository is designed to reproduce the public workflow and code structure, not the exact private results obtained with the restricted prompt set.

## Project overview

The repository contains four main experiment entry points:

- `main.py`: baseline RRT-style prompt search.
- `purple agent.py`: Purple-style defense with risky-memory and simulation-memory filtering.
- `RRT reward guided.py`: reward-guided RRT baseline.
- `purple agent RRT guided.py`: reward-guided Purple variant.

The reviewer-side scripts aggregate those runs into compact paper-style result tables:

- `reviewer_baselines.py`
- `reviewer_reward_guided.py`

## Repository structure

- `public_release_utils.py`: shared environment and prompt-space helpers.
- `data/`: public prompt/data templates only.
- `docs/reproducibility.md`: command-line reproduction notes.
- `paper_table_*.csv`: small public-safe example result tables retained for reference.
- `SANITIZATION_REPORT.md`: summary of what was removed or redacted.
- `LICENSE`: MIT License for the public code release.

Placeholder directories such as `reviewer_runs/` are intentionally kept only as markers for generated outputs and are ignored by Git.

## Installation

1. Create a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

1. Copy `.env.example` to `.env`.
2. Fill in only the provider credentials needed for the scripts you want to run.
3. Optional: if you have approved access to a restricted prompt set, set `PROMPT_SPACE_PATH` to that JSON file.

The retained scripts fail fast with a clear error if a required variable is missing.

## Running the key scripts

Run the baseline reviewer sweep:

```bash
python reviewer_baselines.py
```

Run one reward-guided repeated condition:

```bash
python reviewer_reward_guided.py run-reward-guided --mode reward_guided_purple_full --run-id 01 --h-value 0.40
```

Aggregate repeated reward-guided runs:

```bash
python reviewer_reward_guided.py aggregate-reward-guided
```

The fuller public workflow is documented in `docs/reproducibility.md`.

## Public reproducibility scope

- The repository keeps a small set of public-safe `paper_table_*.csv` files as example outputs.
- Running the reviewer scripts regenerates the same categories of tables locally.
- Raw round-by-round prompt trajectories, generated harmful content, large paper figures, and diagram source files were intentionally omitted from the public package.
- Exact reproduction of the private paper numbers requires a restricted prompt library that is not included here.

## What was omitted from the public release

- Real API keys, tokens, gateway endpoints, and private service configuration.
- Private prompt libraries and raw harmful generations.
- Draft/manuscript assets, internal diagram sources, billing files, IDE state, and other lab-only artifacts.
- Intermediate aggregate files and bulky figure assets that are not needed to understand or run the released code.

## License

This repository is released under the MIT License. See `LICENSE`.
The copyright line in `LICENSE` is intentionally left as a placeholder so the project owner can fill in the correct attribution before publication.

For a compact audit trail, see `SANITIZATION_REPORT.md`.
