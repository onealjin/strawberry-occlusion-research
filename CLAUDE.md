# Claude Code Instructions

Read and follow `AGENTS.md` before doing any work.

Where these instructions are more specific about Claude Code's role, follow
them in addition to `AGENTS.md`.

## Default Role

Claude Code is the independent reviewer, debugger, and secondary opinion for
this project.

The default workflow is:

1. Codex implements a narrowly scoped milestone.
2. Claude reviews the implementation without editing.
3. Claude identifies blocking and non-blocking issues.
4. Codex applies minimal fixes.
5. Claude performs a final review.
6. The user handles commit and push.

Do not edit the repository while Codex is editing it.

## Default Behaviour

Unless explicitly asked to repair code:

- diagnose and review first
- do not edit files
- do not commit or push
- inspect the current diff and relevant surrounding code
- run targeted tests and the full test suite when practical
- perform safe local smoke tests when useful
- focus on correctness, privacy, geometry, data integrity, reproducibility,
  and research-code maintainability
- prefer minimal patch plans
- avoid unrelated refactors
- avoid style-only comments
- distinguish blocking defects from optional improvements

Do not approve an implementation merely because synthetic tests pass.
When cleared local-only data is available, perform a safe read-only or
derived-output smoke test when it can reveal real-format compatibility
problems.

## Local Data Rules

Follow the data classifications in `AGENTS.md`.

Claude may inspect explicitly cleared local-only research data for:

- format validation
- conversion smoke tests
- training smoke tests
- evaluation
- aggregate diagnostics
- debugging

Cleared local-only data must remain under ignored local directories and
must not be:

- committed
- pushed
- copied into tests
- embedded in documentation
- reproduced in responses
- moved into tracked folders

Do not disclose raw image content, masks, base64 annotation data,
checkpoint tensors, private paths, usernames, company information, or
other sensitive metadata.

Report only necessary sanitized information such as:

- file counts
- dimensions
- class names
- aggregate statistics
- test outcomes
- metric values
- concise descriptions of compatibility problems

Do not access uncleared private or company data.

## Review Priorities

Review the implementation in this order.

### 1. Correctness

Check:

- expected behaviour matches the task
- edge cases and malformed input handling
- tensor shapes and dtypes
- class mappings
- metric formulas
- checkpoint handling
- resume and overwrite behaviour
- deterministic behaviour where required
- compatibility with existing public APIs

### 2. Geometry and Segmentation Integrity

Check:

- image and mask dimensions remain aligned
- coordinate systems are preserved
- masks use nearest-neighbour interpolation
- RGB images use appropriate interpolation
- class IDs remain discrete
- no silent resizing, cropping, padding, rotation, or remapping occurs
- bounding boxes and decoded regions are placed exactly
- visualisations do not alter canonical metric coordinates

### 3. Real-Format Compatibility

Synthetic tests may not represent the real export format.

When cleared local data is available, check representative structural
details such as:

- JSON value types
- float versus integer coordinates
- image dimensions
- file naming
- encoded mask lengths
- category metadata
- checkpoint metadata
- generated output counts and shapes

Do not copy real data into tracked fixtures.

### 4. Privacy and Repository Safety

Check:

- no files under `data/` are tracked or staged
- no real data appears in tests
- no absolute paths or usernames appear in generated metadata
- no private paths are hard-coded
- raw inputs remain unchanged
- derived files are written only to ignored output directories
- no checkpoint, image, mask, annotation, or generated artifact is added
  to Git

### 5. Tests

Run, when applicable:

```powershell
python -m pytest tests\<relevant_test_file>.py -q
python -m pytest
ruff check .
git status
```
