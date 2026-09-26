# CHANNEL unified research archive — 2026-09-26

This release accompanies the unified archival commit for the CHANNEL / RAL-MoE research project.

It contains five local-workspace data packages, seventeen selected cluster archive packages, and three additional formal-model/prediction packages. Git tracks paper sources, code, reports, conclusions, tables, figures, configurations, logs, manifests, and compact results. Release assets hold models, POD/ROM assets, raw or necessary evaluation data, and other large files.

Start with `README.md` and `RESULTS_INDEX.md`. Restore large files with:

```sh
python tools/restore_archive.py --scope all
```

The archive is curated rather than a full machine image. Repeated intermediate checkpoints, screening candidates, rebuildable caches, runtime environments, dependencies, and temporary files are listed with reasons under `archive/*-manifest.json`. Three uploaded public copies had credential literals redacted; the original local and cluster workspaces were not changed.

Verification evidence is recorded in `archive/final-verification.json`. Scientific results retain their original protocols and limitations; metrics with different references or aggregation rules must not be combined.
