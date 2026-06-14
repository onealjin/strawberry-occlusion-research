\# AGENTS.md



This repository develops a non-confidential research prototype for occlusion-aware strawberry calyx/stem segmentation and cutline estimation.



\## Confidentiality rules

\- Do not add, generate, commit, summarize, or expose real JIATONG images, videos, labels, production metrics, machine parameters, VisionMaster projects, or patent notes.

\- Use only public datasets, synthetic data, toy masks, or sanitized examples in this repository.

\- Private JIATONG data lives outside this repo and must be accessed only through local paths configured by the user.

\- Do not hard-code private paths, thresholds, customer names, machine settings, or proprietary workflows.



\## Project goal

Build a modular research pipeline:

image -> visible segmentation -> occlusion-state classification -> amodal attachment-zone estimation -> cutline geometry -> uncertainty-aware decision -> JSON output.



\## Coding rules

\- Prefer simple, readable PyTorch and Python.

\- Keep modules small and testable.

\- Use config files instead of hard-coded constants.

\- Any model should support ONNX export.

\- Any inference output should be serializable to JSON.

\- Add tests for geometry, cutline metrics, post-processing, and ONNX parity where possible.



\## Commands

\- Install: `pip install -e .`

\- Test: `pytest`

\- Format: `ruff format .`

\- Lint: `ruff check .`



\## Done means

\- Code runs without syntax errors.

\- Relevant tests are added or updated.

\- No private data, paths, or proprietary JIATONG details are committed.

\- The change is documented if it affects training, inference, or evaluation.

