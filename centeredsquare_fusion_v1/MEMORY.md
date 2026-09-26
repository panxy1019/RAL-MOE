# Project memory

- 2026-09-20: The canonical SH/HP scientific plotting workflow is documented in `PLOTTING_MIGRATION_GUIDE.md`; exact local asset hashes and renamed-output aliases are in `PLOTTING_FILE_MANIFEST.json`. Use `plot_boundary_fusion_fields.py` as the active implementation and treat `reference_plotting/plot_field_figures.py` as read-only layout reference. Source: verified local files and frozen figure manifests. Confidence: high.
- 2026-09-20: The local reference VTK is the durable source. Its historical virtual-machine path is no longer present, and the historical training-server location could not be reverified because the old login credential no longer authenticates. Do not store credentials in project memory. Confidence: high.
