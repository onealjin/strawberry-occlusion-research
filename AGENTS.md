# Agent Instructions

## Project

This is a non-confidential public research codebase for occlusion-aware strawberry calyx/stem segmentation and cutline estimation.

## Confidentiality Rules

- Use only public, synthetic, toy, or sanitized data.
- Do not read, request, copy, commit, upload, or reference private company data.
- Do not use raw private images, masks, industrial project files, production machine settings, patent notes, internal reports, proprietary thresholds, or internal metrics unless the user explicitly says they are cleared.
- Never hard-code private local data paths.
- Public code must not depend on private local folders.
- Tests must use synthetic, toy, temporary, or public-safe data.

## Tool Roles

- Codex is the primary implementation agent.
- Claude Code is the secondary reviewer, debugger, and rescue agent.
- Claude Code should usually diagnose, review, or propose minimal fixes before editing.
- Avoid style-only rewrites unless they improve correctness, privacy safety, maintainability, or research clarity.

## Coding Preferences

- Prefer simple, readable Python and PyTorch.
- Use modular files under src/strawberry_occlusion/.
- Use pathlib for paths.
- Use pytest for tests.
- Avoid unnecessary dependencies.
- Do not add real images, masks, model weights, private artifacts, or large binary files.

## Standard Commands

Run tests with: python -m pytest

Check repository state with: git status

## Expected Workflow

Before editing:
- Summarize the intended change.
- List likely files to modify.

After editing:
- Run relevant tests.
- Summarize what changed.
- Mention tests not run.
- Mention any privacy or data-leakage risks.
