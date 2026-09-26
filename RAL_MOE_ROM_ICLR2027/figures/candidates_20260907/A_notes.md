# Candidate A: compact horizontal flow

Design: a horizontal regime-selection / local-evolution / reconstruction / physical-fusion pipeline. One shallow lower inset explains the sparse velocity closure. Colors encode routing (purple), learned dynamics (warm), physics (blue), and physical assembly (teal). Curves and grids are schematic glyphs, not simulated fields.

Semantics retained:

- Only the physical parameter enters E2; its trajectory preference is fixed.
- The outer selector permits Top-1 or an adjacent S-H / H-P Top-2 pair.
- Physical history initializes both selected native encoders and separately supplies a chart-independent descriptor to the correction weight. The weight label explicitly includes parameter, normalized regime probability, and descriptor.
- Selected charts evolve independently in their native coordinates; each decoder reconstructs its field before physical fusion. The second chart is Top-2 only. Top-1 assembly is the identity. There is no cross-latent mixing or fused-state feedback.
- Inside one chart, a Top-1 group supplies a shared expert in parallel with Top-2 routed experts. The resulting sparse velocity closure adds to Galerkin dynamics before velocity RK integration.

Intentional simplifications: the compact inset omits the expert's nonlinear / affine / bilinear internal branches, the channel-wise feature construction, and the pressure algebraic map. The `a` labels and Velocity label identify the integrated state; no pressure ODE is implied. Expert indices 1 and 2 symbolize the selected experts, not fixed globally active expert identities. The normalized selected-pair probability is denoted by bar-pi without its r subscript for space. Caption should explain these omissions if this variant is selected.

Suggested caption: **RAL-MoE-ROM.** A parameter-only router selects one regime-local chart or an admissible adjacent pair. Selected charts encode the same physical history and evolve independently; reconstructed fields are assembled in physical space with a history-conditioned convex weight. Within each chart, a group-selected shared expert and two routed experts form a sparse velocity closure that augments Galerkin dynamics before time integration. Pressure is recovered algebraically (not shown).

Exports are 12 x 5.2 inches. Minimum source text is 18.5 pt, equivalent to 8.50 pt at a 397 pt figure width. SVG retains editable text; PDF uses TrueType fonts. There are no raster assets.
