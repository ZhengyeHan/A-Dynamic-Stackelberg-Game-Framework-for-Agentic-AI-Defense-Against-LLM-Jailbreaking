from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PUBLIC_PROMPT_SPACE_PATH = DATA_DIR / "public_prompt_space.json"
BENIGN_PROMPTS_PATH = DATA_DIR / "heldout_benign_prompts.json"
DEFAULT_ROOT_PROMPT = "Tell me a fun fact."

load_dotenv(REPO_ROOT / ".env", override=False)


def _load_text_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise RuntimeError(f"Expected a JSON list of strings in {path}.")
    return data


def load_prompt_space() -> list[str]:
    configured_path = os.getenv("PROMPT_SPACE_PATH", "")
    prompt_path = Path(configured_path).expanduser() if configured_path else PUBLIC_PROMPT_SPACE_PATH
    if not prompt_path.is_absolute():
        prompt_path = REPO_ROOT / prompt_path

    prompt_space = _load_text_list(prompt_path)
    if DEFAULT_ROOT_PROMPT not in prompt_space:
        raise RuntimeError(
            f"The prompt space file {prompt_path} must include the root prompt {DEFAULT_ROOT_PROMPT!r}."
        )
    return prompt_space


def load_benign_prompt_subset() -> list[str]:
    return _load_text_list(BENIGN_PROMPTS_PATH)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    raise RuntimeError(
        f"Missing required environment variable {name}. Copy .env.example to .env and fill in the values you need."
    )


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return default


def create_openai_client(
    *,
    api_key_env: str,
    base_url_env: str | None = None,
    default_base_url: str | None = None,
) -> OpenAI:
    kwargs: dict[str, str] = {"api_key": require_env(api_key_env)}
    base_url = get_env(base_url_env, default_base_url) if (base_url_env or default_base_url) else None
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
