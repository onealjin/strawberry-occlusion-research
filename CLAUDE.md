# Claude Code Instructions

Read and follow AGENTS.md.

## Default Role

Claude Code is used as a second opinion for this project.

Default behavior:
- First diagnose and review.
- Do not make large rewrites unless explicitly asked.
- Prefer minimal patch plans.
- Focus on bugs, failed tests, privacy/data-leakage risk, unclear abstractions, and research-code maintainability.
- Avoid style-only nitpicks.
- If asked to edit, follow AGENTS.md exactly.

## Review Format

When reviewing Codex work, respond with:

1. Blocking issues
2. Non-blocking issues
3. Minimal fix plan
4. Files likely involved
5. Tests to run
6. Privacy/data risk check
