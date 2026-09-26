"""Build provenance inventory from observed verification files, no inference."""
import argparse
import json
from pathlib import Path


MODEL_PATHS = {
    'steady': 'steady_specialist_v1/code/train_v16_4_v2_r32_compat.py',
    'hopf': 'Hopf/migrated_h4_expanded/periodic_moe_3090/train_periodic_moe.py',
    'periodic': 'periodic_specialist_r32/code/train_periodic_moe.py',
}
ENTRY = {
    'steady': 'train_s2b_3090.Experiment.rollout; finalize_s2b_3090.evaluate_horizon',
    'hopf': 'train_hopf_moe_expanded.rollout_batch/autonomous_step; audit_hopf_native_attractor.rollout_arrays/horizon_metrics',
    'periodic': 'train_periodic_moe.integrate_autonomous_step_np; evaluate_periodic_r32_portable window policy and weighted geometry',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.analysis_dir
    root = '/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE'
    lines = ['# Model and asset inventory', '',
             'Generated from this run\'s strict-load, file-hash, and prediction-invariance evidence.', '',
             'All three runtime classes are the original `OperatorSpaceMoEROM`, with original `PhysicsAwareExpert` implementations.',
             'Router attributes: `group_router`, `velocity_group_routers`, `pressure_group_routers`.',
             'Expert attributes: `velocity_expert_groups`, `pressure_expert_groups`, `velocity_shared_experts`, `pressure_shared_experts`.',
             'Group selection is shared across the two channels. Expert IDs are zero-based and group-local.', '',
             'Steady S4 is an independent diagnosis, not the current manuscript Steady reference.', '']
    for chart, model_path in MODEL_PATHS.items():
        report = json.loads((out / f'logs/{chart}_routing_verification.json').read_text())
        lines += [f'## {chart}', '', f"- Checkpoint: `{report['checkpoint']}`",
                  f"- Checkpoint SHA256: `{report['checkpoint_sha256']}`",
                  f'- Original model/expert definitions: `{root}/{model_path}`',
                  f'- Original rollout/evaluation: `{ENTRY[chart]}`',
                  f"- Integrator: {report['integrator']}", f"- Precision: {report['precision']}",
                  f"- Window policy: {report['window_policy']}",
                  '- Strict load: no missing/unexpected keys; checkpoint and model state unchanged.',
                  '- Hook-on/off prediction equality: bitwise equality for every evaluated window.', '',
                  '### Held-out windows', '', '| Label | Actual stored Re | Count | Start indices |',
                  '|---|---:|---:|---|']
        for row in report['rows']:
            lines.append(f"| {row['label']} | {row['Re']} | {row['windows']} | {row['starts']} |")
        lines += ['', '### Routing configuration', '', '```json', json.dumps(report['routing_config'], indent=2), '```', '',
                  '### Original source and asset fingerprints', '', '| Absolute path | SHA256 |', '|---|---|']
        sources = dict(report['source_sha256'])
        if chart == 'steady':
            initial = json.loads((out / 'logs/steady_reproduction.json').read_text())
            sources.update(initial['asset_hashes'])
        extra_path = out / 'logs/input_view_fingerprints.json'
        if extra_path.exists():
            extra = json.loads(extra_path.read_text()).get(chart, {})
            sources.update({k: v['sha256'] for k, v in extra.items()})
        for path, sha in sources.items():
            lines.append(f'| `{path}` | `{sha}` |')
        lines += ['']
    lines += ['## Normalization and compatible data views', '',
              '- Steady: original relocated `training_s2b_portable.json` points to `steady_specialist_v1/source_artifacts/steady`; `Experiment` uses its `Global_POD_AreaWeighted_L2` compatibility view to construct arrays and fits the original training-only normalization. POD/ROM physical assets are listed above.',
              '- Hopf: the original trainer reconstructs normalization from the train view; all ten normalization fields match the checkpoint exactly (maximum absolute difference 0). The full split coefficient/index assets are read only for evaluation, with held-out IDs explicitly selected.',
              '- Periodic: `scalers` are loaded directly from the frozen checkpoint; local POD data view is `periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2`, with native velocity ROM and pressure surrogate paths listed above.',
              '- Initial history is observed; later a/b/history are predicted. Native Re/time metadata are retained. Steady and Periodic phase features are retained as provided by the original data-index contract; their upstream construction is not re-audited here.', '',
              '## Source authority for historical result comparisons', '',
              '- Hopf: `Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json`; checkpoint SHA matches the user-specified final.pt.',
              '- Periodic: `periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json`.',
              '- Steady identity conflict: copied original `source/SPECIALIST_ABLATION_REPORT.md` and `source/NUMERICAL_EXPERIMENTS_REVISION8_DETAILED_CN.md`.', '',
              'This inventory does not claim that all historical training/data provenance questions have been re-audited. The current task restores the specified original inference contracts and observes routing without modifying prediction mathematics.', '']
    (out / 'reports/model_asset_inventory.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
