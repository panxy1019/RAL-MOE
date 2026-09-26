# Recent related work: verified provenance

Verified on 2026-09-06 for the final P0/P1 revision. This is an internal evidence record, not manuscript text.

## Scope and result

- Added exactly the four requested papers to `references.bib` and cited them in the final paragraph of `sections/related_work.tex`.
- Each entry is the unmodified official proceedings BibTeX export, including its official citation key. No bibliographic metadata were guessed. The lowercase `dong, huanshuo` in the MoE-POT export is retained exactly as supplied by the proceedings.
- Replaced the broad claim about most MoEs consuming and producing a shared representation with the qualified statement that many formulations route within a shared representation or common output space.
- Added concise distinctions among constraint decomposition, implicit representation decomposition, neural-operator pretraining, and invariant operator-expert forecasting. The manuscript's distinction is the outer routing of complete autonomous projected ROMs with independently constructed reduced coordinates, followed by physical-space assembly; it does not claim that every prior MoE shares one latent space.
- No title, abstract, method, or experimental performance values were edited by this literature subtask.

## Official source and export records

### Chalapathi, Du, and Krishnapriyan (ICLR 2024)

Title: *Scaling physics-informed hard constraints with mixture-of-experts*.

- [Official proceedings record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9aeda582add763c41c7b39691ce19ab0-Abstract-Conference.html)
- [Official BibTeX export](https://proceedings.iclr.cc/paper_files/paper/3994-/bibtex)
- Key: `ICLR2024_9aeda582`
- Export SHA256: `b3bdb781f381a1112740ccda999f41597585e0446d924fc5810f10ee3114fa37`
- Evidence: the proceedings abstract describes hard physical constraints solved over decomposed smaller domains by differentiable-optimization experts. The added text claims localized hard constraints, not autonomous local reduced dynamical models.

### Ben-Shabat et al. (NeurIPS 2024)

Title: *Neural Experts: Mixture of Experts for Implicit Neural Representations*.

- [Official proceedings record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b83fae17d73b079b1b98ab200276db9f-Abstract-Conference.html)
- [Official BibTeX export](https://proceedings.neurips.cc/paper_files/paper/26665-/bibtex)
- Key: `NEURIPS2024_b83fae17`
- Export SHA256: `67260e9ece7195f0356dc33f802e1d97ac4caee09d6d97d6f5f3e0a5c494204e`
- Evidence: the proceedings abstract describes jointly learned domain subdivision and local piecewise implicit functions, with gating pretraining and conditioning. The related-work text characterizes the work as local implicit representation rather than dynamical ROM assembly.

### Wang et al. (NeurIPS 2025)

Title: *Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training*.

- [Official proceedings record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d23a9991a6f64482bf395628e279f5f-Abstract-Conference.html)
- [Official BibTeX export](https://proceedings.neurips.cc/paper_files/paper/29160-/bibtex)
- [Official paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2d23a9991a6f64482bf395628e279f5f-Paper-Conference.pdf)
- Key: `NEURIPS2025_2d23a999`
- Export SHA256: `f4c5f0a9c399ce46a8c711ad0a39e7fd1507e9ea4808f23539786e67b9d4007e`
- Evidence: the abstract and Section 4 describe neural-operator pretraining with shared experts and sparse routed experts inside operator blocks. Figure 3 locates the MoE in the operator architecture; Section 5.4 analyzes router decisions. The manuscript cites the shared/routed architecture without importing its performance numbers.

### Li et al. (ICLR 2026)

Title: *Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning*.

- [Official proceedings record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html)
- [Official BibTeX export](https://proceedings.iclr.cc/paper_files/paper/9075-/bibtex)
- [Official paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf)
- Key: `ICLR2026_754612bd`
- Export SHA256: `74e05c64e6c9e992d73cb389a446315914ab9c7a5cd7cae6ebe1b642a1e3a6e2`
- Evidence: Section 3.1 describes operator experts and a fusion network conditioned on physical parameters; it permits simple addition or an extra network depending on operator composition. The added wording therefore uses “parameter-conditioned fusion,” not an inaccurate claim that every fusion stage is a nonlinear learned network. The target is invariant out-of-distribution PDE forecasting.

## Retrieval and verification

1. Located and read each official proceedings record; followed the actual BibTeX link in its HTML. For MoE-POT, the official proceedings search returned the paper's hash URL.
2. Retrieved exports with PowerShell `Invoke-WebRequest`; responses use a byte content type, decoded with `[System.Text.Encoding]::UTF8.GetString(...)`.
3. Re-fetched all four exports after editing. After normalizing only file line endings, every complete official export occurs exactly in `references.bib`: **4/4 PASS**.
4. Checked all bibliography keys: **no duplicates**. Checked every citation key in `sections/related_work.tex`: **all resolve**.
5. Full manuscript compilation and page/visual QA are performed by the main task, since this subtask owns only the two literature files and this evidence record.

OpenReview served a public-browser verification challenge, but no access bypass was attempted. The official proceedings were available and supplied all four verified exports, so no bibliography item is blocked or sourced from third-party metadata.

## Current ICLR 2027 AI disclosure policy (read-only verification)

- [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines): an AI use statement is mandatory and excluded from the main-text page limit. Submission main text is limited to nine pages.
- [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors): disclose AI use both in the paper and the submission form. Exact boilerplate wording is not mandatory. Authors remain responsible for all contents. Required-disclosure categories include method implementation and results interpretation; recommended categories include code, figures, literature analysis/search, and language editing.
- [Official ICLR 2027 style ZIP](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip): the `iclr2027/iclr2027_conference.tex` example uses `\subsection*{AI use statement}` and specifies no more than one page. The archive was read in memory, without replacing any project style file.
- Main task was notified of the exact heading, flexible content format, and separate submission-form requirement. No AI statement was edited by this literature subtask. Authors must personally confirm any statement claiming their review/verification is complete.
