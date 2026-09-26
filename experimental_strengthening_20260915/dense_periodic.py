"""Isolated, total-parameter-matched dense closure in the native Periodic trainer."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))

def patch(trainer, contract_path):
    original = trainer.OperatorSpaceMoEROM
    class DenseROM(original):
        def __init__(self, *args, **kwargs):
            cfg = inspect.signature(original.__init__).bind(None, *args, **kwargs)
            cfg.apply_defaults()
            cfg = cfg.arguments
            super().__init__(*args, **kwargs)
            target = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # Remove all learned sparse routing and all experts, not just mask their use.
            for name in ('group_router', 'velocity_group_routers', 'pressure_group_routers',
                         'velocity_expert_groups', 'pressure_expert_groups',
                         'velocity_shared_experts', 'pressure_shared_experts'):
                delattr(self, name)
            common = sum(p.numel() for p in self.parameters() if p.requires_grad)
            input_dim = cfg['hidden_dim'] + cfg['out_dim'] + cfg['pressure_dim']
            output_dim = cfg['out_dim'] + cfg['pressure_dim']
            # Three hidden layers, one joint correction with velocity/pressure output slices.
            def count(w):
                return (input_dim+1)*w + 2*(w+1)*w + (w+1)*output_dim
            width = min(range(16, 8193), key=lambda w: abs(common+count(w)-target))
            self.dense_correction = nn.Sequential(
                nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(cfg['dropout']),
                nn.Linear(width, width), nn.GELU(), nn.Dropout(cfg['dropout']),
                nn.Linear(width, width), nn.GELU(), nn.Dropout(cfg['dropout']),
                nn.Linear(width, output_dim))
            self.num_experts = 1
            self.num_shared_experts = 0
            self.num_regime_groups = 1
            self.shared_per_group = 0
            self.experts_per_group = 1
            actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
            assert abs(actual-target)/target < .005
            assert not any('expert' in n or 'group_router' in n for n, _ in self.named_parameters())
            write(contract_path, dict(reference_total_trainable=target, dense_total_trainable=actual,
                common_parameters=common, dense_width=width, hidden_layers=3,
                relative_parameter_difference=(actual-target)/target,
                matching='total allocated trainable parameters, NOT active parameters',
                correction='one joint dense GELU network, velocity and pressure output slices',
                removed='all group/channel routers and shared/routed experts',
                retained='native encoder, inputs, Galerkin, pressure base, closure confidence, optimizer, curriculum, splits, validation selector'))

        def group_router_outputs(self, x):
            # Constant one-column compatibility metadata, no learned router.
            return x.new_ones((x.shape[0], 1)), x.new_zeros((x.shape[0], 1))

        def forward(self, x, return_expert_stack=True, pressure_state_override=None, return_closure_params=False):
            h, attr = self._condition_attractor(self._encode(x))
            _, state = self._state_slices(x)
            if pressure_state_override is not None:
                state = pressure_state_override
            output = self.dense_correction(torch.cat([h, state], dim=1))
            params = self._closure_params(h)
            params.update(attr)
            gate = self.rom_regime_gate(h)
            if gate is not None:
                params['rom_gate'] = gate
            ones = x.new_ones((x.shape[0], 1))
            result = (output[:, :self.out_dim], output[:, self.out_dim:], [ones, ones],
                      x.new_empty((0, 1, self.out_dim)))
            return (*result, params) if return_closure_params else result

    def zero_init(model, scalers, logit):
        with torch.no_grad():
            layer = model.dense_correction[-1]
            layer.weight.zero_()
            layer.bias.copy_(torch.cat([-scalers['rhs_op_mean']/scalers['rhs_op_scale'],
                                       -scalers['pressure_mean']/scalers['pressure_scale']]))
            model.closure_confidence_head[-1].weight.zero_()
            model.closure_confidence_head[-1].bias.fill_(float(logit))
    trainer.OperatorSpaceMoEROM = DenseROM
    trainer.initialize_physical_zero_residual = zero_init
    trainer.expert_operator_diversity_analysis = lambda *a, **k: {
        'applicable': False, 'reason': 'single dense correction has no expert pairs'}
    return DenseROM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT/'periodic_specialist_r32'
    source = base/'code/train_periodic_moe.py'
    spec = importlib.util.spec_from_file_location('dense_native', source)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    patch(trainer, args.output/'architecture.json')
    checkpoint = torch.load(base/'checkpoint/FINAL_PERIODIC_SPECIALIST.pt', map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    values = dict(vars(saved) if hasattr(saved, '__dict__') else saved)
    original = dict(values)
    values.update(data_root=base/'assets/Global_POD_AreaWeighted_L2',
        tensor_path=base/'assets/velocity_rom_periodic.npz',
        pressure_surrogate_path=base/'assets/pressure_poisson_surrogate_periodic.npz',
        output_dir=args.output, experiment_name='Dense_totalmatched_Circular_P',
        seed=1600, resume_checkpoint=None, eval_only_checkpoint=None,
        swanlab_mode='disabled', swanlab_required=False, swanlab_log_dir=args.output/'swanlog')
    # Only architecture-specific router penalties are removed.
    for key in list(values):
        if key.startswith('lambda_') and any(t in key for t in ('gate', 'router', 'balance', 'entropy', 'diversity', 'group')):
            values[key] = 0.0
    if args.smoke:
        values.update(epochs=1, min_epochs=1)
    write(args.output/'protocol.json', dict(config=values, changes={k: dict(old=original.get(k), new=v)
        for k, v in values.items() if str(v) != str(original.get(k))}, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        smoke_only=args.smoke, reference_checkpoint_epoch=checkpoint.get('epoch')))
    trainer.parse_args = lambda: argparse.Namespace(**values)
    # Native trainer uses legacy np.random state; adapt only JSON-compatible scalar representation if needed.
    trainer.main()
    write(args.output/'TRAINING_COMPLETED.json', {'completed': True, 'smoke_only': args.smoke})

if __name__ == '__main__':
    main()
