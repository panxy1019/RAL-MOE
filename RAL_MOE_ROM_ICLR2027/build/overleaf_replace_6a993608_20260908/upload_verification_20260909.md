# Overleaf replacement verification — 2026-09-09

## Result

Completed replacement in https://www.overleaf.com/project/6a9936085c574b84a0ab1597 .
The existing project title, `PMD-Galerkin-Pan_ICLR`, was preserved. The earlier Overleaf project was not changed.

- User explicitly confirmed clearing the original 18 files.
- The original files were deleted after verifying the exact deletion list and available recovery copies.
- The validated RAL-MoE-ROM manuscript was uploaded as 8 root files and the `appendix`, `sections`, and `figures` directories.
- An initial ZIP upload was stored as an ordinary file by Overleaf. That agent-created temporary ZIP was removed after the source-directory upload.
- Final visible file tree contains exactly the 31 expected manuscript files, with no original-project files or temporary ZIP left.
- `main.tex` was set as the main document.
- A new online compile produced the RAL-MoE-ROM manuscript: 23 pages, updated Figure 1 caption on page 3.
- A browser screenshot of the newly compiled page 3 visually confirmed the latest two-panel flowchart is rendered.
- Online logs: 0 errors, 1 float-placement warning (`h` changed to `ht` in `appendix/circular_cylinder.tex`), and 21 underfull-box/typesetting information entries.

## Recovery

Local original-project backup: `original_PMD_Galerkin_Pan_ICLR.zip` in this directory.

- 18 files; 15,135,984 bytes.
- ZIP SHA256: `634de446560390f5c8b5cedcb19c3f4782953a9c44ee43c3839418be42f260e2`.
- CRC and exact file list were independently checked.
- Original Overleaf version was also labelled `Before RAL-MoE-ROM replacement 2026-09-09` before deletion.
- The online history at that time showed the original project version from 3 September, 4:55 pm.
- A fresh source-download action was attempted on 9 September, but a new local download was not confirmed; recovery evidence is the verified existing ZIP plus the saved online history label.

## Uploaded local source

- Source package: `../overleaf_upload_user_figure_20260908.zip`.
- Package SHA256: `601733a2408cc91facd43a71d198a0da619cba411fdd3ca012f5ab9d5b036d60`.
- Actual upload directory: `../user_figure_20260908/upload_extracted/`.
- All 31 extracted file hashes were checked against `../user_figure_20260908/upload_manifest.json` immediately before upload.
- Latest figure: `figures/ral_moe_rom_overview_user_20260908.png`.
- Figure SHA256: `5f53933dafd79441424e031e4f1cd0da72065b44383224186ac9bcbd02274eff`.

## Verification limits

Online verification used the visible file tree, main-document action, newly generated PDF preview, and compiler logs. No byte-for-byte hash comparison of re-downloaded online sources/PDF was performed. No scientific content or experimental data was changed during this upload turn. Existing local PDFs, prior figures, and backups were preserved.
