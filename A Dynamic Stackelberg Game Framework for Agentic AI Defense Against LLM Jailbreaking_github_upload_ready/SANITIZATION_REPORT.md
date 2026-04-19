# Sanitization Report

## Sensitive content found

- Hardcoded API credentials in the main experiment entry points.
- A private Portkey-compatible gateway URL and gateway credential in the `free-API` folder.
- Local IDE metadata containing absolute user-home paths and a project identifier.
- Billing receipts, draft/manuscript folders, and local-only research artifacts unrelated to the public code release.
- Raw round-level jailbreak traces, logs, and generated outputs containing unsafe prompt content and model generations.

## What was changed

- Replaced hardcoded credentials in the retained Python scripts with environment-variable loading via `public_release_utils.py`.
- Added `.env.example` so public users can supply their own credentials without exposing real values.
- Replaced the original restricted prompt library with `data/public_prompt_space.json`, a redacted public prompt file.
- Added `data/restricted_prompt_space.template.json` to show the file shape expected for approved non-public prompt sets.
- Removed the private `free-API` implementation files and replaced that folder with a placeholder note.
- Removed local IDE state, caches, billing files, manuscript folders, raw round-by-round output directories, intermediate aggregate CSV files, large figure binaries, and diagram source assets from the final upload-ready copy.

## Manual inputs still required

- Valid API credentials for the providers used by the scripts you want to run.
- An approved restricted prompt set if you need to reproduce the original private attack library rather than the redacted public substitute.
- A user-supplied Portkey-compatible gateway configuration only if you plan to rebuild the omitted free-gateway experiments locally.

## Public-release limitations

- Exact reproduction of the original paper numbers that depended on the restricted prompt library is not possible from this release alone.
- The public release preserves the core scripts, small public-safe table outputs, and the reviewer-side aggregation pipeline, but omits raw unsafe traces, private-service variants, and paper-only artwork.
