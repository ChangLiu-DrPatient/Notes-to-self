# Deprecated HRLib Experiment Paths

This directory keeps historical HRLib experiments that are no longer part of
the active default pipeline.

## Why deprecated

- MiniLM score-gated retrieval is the active default path.
- Cross-encoder reranking and BGE-M3-heavy experiment matrices were evaluated but
  did not provide strong enough improvements to become default.
- To keep the active code path minimal, reranking helpers and runbooks were
  removed from the main `examples/hrlib` workflow and archived here.

## Current status

- Files here are preserved for historical reference and reproduction of old
  experiments.
- The active runbook intentionally omits BGE-M3 and rerank setup to keep the
  default workflow minimal.
- Active scripts in `examples/hrlib` no longer expose rerank mode.
- Common historical embedders referenced in archived experiments:
  `BAAI/bge-m3` and `Qwen/Qwen3-Embedding-0.6B`.
