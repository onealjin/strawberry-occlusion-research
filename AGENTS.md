# Agent Instructions

## Project

This is a public research codebase for occlusion-aware strawberry
calyx/stem segmentation, uncertainty analysis, synthetic occlusion
experiments, and cutline estimation.

The public repository must remain reproducible using public, synthetic,
toy, sanitized, or procedurally generated examples.

## Data Classification

Treat project data as belonging to one of three categories.

### 1. Public-safe data

This includes:

- public datasets
- synthetic data
- toy examples
- procedurally generated examples
- explicitly sanitized examples
- temporary synthetic test fixtures

Public-safe data may be used in source code, documentation, examples, and
tests when appropriate.

### 2. Explicitly cleared local-only research data

The user may explicitly clear particular local files or datasets for
research use.

Agents may inspect and process such cleared data locally for:

- dataset conversion
- smoke testing
- training
- evaluation
- debugging
- aggregate analysis

Cleared local-only data must still not be:

- committed
- pushed
- uploaded elsewhere
- copied into tracked tests
- embedded in documentation
- reproduced in responses
- included in generated public artifacts

Only sanitized aggregate information may be reported, such as file counts,
image dimensions, class names, metric values, and validation results.

Cleared local datasets and generated artifacts must remain under ignored
local directories, normally `data/`.

Clearance for local research use does not automatically mean clearance for
public redistribution.

### 3. Uncleared private or company data

Do not read, request, copy, process, commit, upload, summarize, or reference
uncleared private or company data.

This includes:

- raw private images and masks
- VisionMaster or VisionTrain project files
- industrial project files
- production machine settings
- patent notes
- internal reports
- proprietary thresholds
- internal metrics
- confidential source code
- private model weights

Do not access such data unless the user explicitly states that the specific
material is cleared.

## Path and Metadata Safety

Never hard-code private local data paths in public code.

Public code must not depend on a particular user's computer, username,
company folder, or private dataset location.

Use command-line arguments, configuration objects, or relative paths.

Generated public metadata must not contain:

- absolute paths
- usernames
- company names or paths
- private folder names
- raw annotation content
- base64 image or mask data
- checkpoint tensors

Use `pathlib` for filesystem paths.

## Repository and Artifact Safety

Do not add real images, masks, annotations, model weights, checkpoints,
training outputs, evaluation outputs, or large binary artifacts to Git.

Local data and generated artifacts should remain under ignored directories,
normally:

- `data/vt_exports_raw/`
- `data/vt_normalized/`
- `data/vt_qa/`
- `data/training_runs/`
- `data/evaluation_runs/`

Before finishing a task involving local data, check that nothing under
`data/` is tracked or staged.

Do not modify raw input data. Derived data must be written to a separate
output directory.

Refuse to overwrite generated outputs by default unless the user explicitly
requests replacement or an `--overwrite` option is provided.

## Tests

Tests must use only:

- synthetic data
- toy data
- temporary files under `tmp_path`
- public-safe fixtures

Never copy real local images, masks, annotation JSON, checkpoints, or
metadata into tests.

Tests should be deterministic, small, and fast enough to run on CPU.

Tests must not require proprietary SDKs, private services, interactive GUI
windows, or external network access.

Do not download pretrained model weights during tests.

## Tool Roles

### ChatGPT

ChatGPT is responsible for:

- research planning
- architecture decisions
- debugging explanations
- milestone design
- experiment interpretation
- deciding Codex tasks
- MASc and research strategy

### Codex

Codex is the primary implementation agent.

Codex should:

- inspect the repository before editing
- summarize the intended change
- list likely files to modify
- implement narrowly scoped tasks
- add or update tests
- run relevant tests and lint checks
- report changed files and command results

Codex must not commit or push.

### Claude Code

Claude Code is the secondary reviewer, debugger, and rescue agent.

Claude should normally:

- review Codex's implementation without editing
- run tests and smoke checks
- identify correctness, privacy, and maintainability problems
- recommend minimal fixes
- avoid style-only rewrites

Claude should edit only when explicitly assigned a repair task.

Claude must not commit or push.

Do not allow Codex and Claude to edit the repository simultaneously.

## Coding Preferences

Prefer:

- simple, readable Python
- simple, readable PyTorch
- modular files under `src/strawberry_occlusion/`
- explicit types and validation
- deterministic behavior where practical
- reusable importable functions
- small command-line entry points
- `pathlib`
- `pytest`
- clear error messages
- minimal dependencies

Avoid:

- unnecessary abstractions
- duplicated implementations
- hidden global state
- undocumented coordinate transformations
- silent class remapping
- silent geometry changes
- unnecessary dependencies
- style-only rewrites

Reuse existing models, metrics, inference utilities, dataset loaders, and
visualization functions when they already implement the required behavior.

## Geometry and Segmentation Safety

For segmentation masks:

- preserve discrete class IDs
- use nearest-neighbor interpolation
- never use bilinear interpolation on class masks
- validate mask dimensions and allowed values
- preserve image/mask coordinate alignment
- document every resize, crop, pad, rotation, or coordinate transformation

For RGB images, bilinear resizing is acceptable when required.

Do not silently alter the segmentation ontology or class mapping. Any class
mapping change must be explicit and documented.

## Experiment Reproducibility

Training and evaluation code should record enough information to reproduce
a run, including where applicable:

- model name
- class mapping
- input dimensions
- train/validation split identifier
- random seed
- optimizer
- learning rate
- loss configuration
- batch size
- epoch count
- device type
- checkpoint-selection metric

Generated metadata should use sanitized relative identifiers rather than
absolute local paths.

Do not present validation results as independent test results.

## Git Safety

Agents must not run:

- `git commit`
- `git push`
- destructive Git reset commands
- commands that remove untracked user data

The user performs Git commits and pushes through PowerShell.

Before completion, report:

- files changed
- tests run
- tests not run
- lint results
- Git status
- privacy or data-leakage risks
- whether local data was accessed
- confirmation that nothing was committed or pushed

## Standard Commands

Run tests:

```powershell
python -m pytest
```

Run formatting and lint checks:

```powershell
ruff format .
ruff check .
```

Check repository state:

```powershell
git status
```
