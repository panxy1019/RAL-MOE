"""Passive forward hooks. Actual combined gates are the statistical authority."""
import gzip
import hashlib
import json

import numpy as np
import torch


def model_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def softmax(x, temperature):
    z = np.asarray(x, dtype=np.float64) / temperature
    z -= np.max(z, axis=-1, keepdims=True)
    y = np.exp(z)
    return y / y.sum(axis=-1, keepdims=True)


class RoutingLogger:
    def __init__(self, model, chart, stages):
        self.model, self.chart, self.stages = model, chart, stages
        self.records, self.handles, self.cache = [], [], {}
        self.context = None
        self.config = {k: getattr(model, k) for k in (
            'num_regime_groups', 'experts_per_group', 'shared_per_group', 'top_k',
            'group_top_k', 'temperature', 'group_temperature', 'gate_floor',
            'group_gate_floor', 'shared_scale', 'routed_scale')}
        assert self.config['shared_per_group'] == 1
        assert self.config['top_k'] == 2 and self.config['group_top_k'] == 1
        assert self.config['gate_floor'] == self.config['group_gate_floor'] == 0

    def begin(self, re_value, trajectory, starts, times, horizon):
        self.context = (float(re_value), str(trajectory), [int(s) for s in starts],
                        [float(t) for t in times], horizon)
        self.calls = 0

    def attach(self):
        def capture(key):
            def hook(module, inputs, output):
                self.cache[key] = output.detach().float().cpu().numpy().copy()
            return hook
        self.handles.append(self.model.group_router.register_forward_hook(capture('group')))
        for channel, routers in [('u', self.model.velocity_group_routers),
                                 ('p', self.model.pressure_group_routers)]:
            for group, router in enumerate(routers):
                self.handles.append(router.register_forward_hook(capture((channel, group))))
        self.handles.append(self.model.register_forward_hook(self.observe))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def observe(self, module, inputs, output):
        re_value, trajectory, starts, times, horizon = self.context
        step, stage = self.calls // self.stages + 1, self.calls % self.stages + 1
        assert step <= horizon
        self.calls += 1
        group_logits = self.cache['group']
        group_probs = softmax(group_logits, self.config['group_temperature'])
        G, E = self.config['num_regime_groups'], self.config['experts_per_group']
        group_ids = []
        for channel, gate in zip(('u', 'p'), output[2]):
            gates = gate.detach().float().cpu().numpy().reshape(len(starts), G, E + 1)
            assert np.isfinite(gates).all() and (gates >= 0).all()
            # BF16 gate construction is rounded; do not renormalize away this fact.
            assert np.max(np.abs(gates.sum((1, 2)) - 1)) < .02
            ids = gates.sum(2).argmax(1)
            group_ids.append(ids)
            for i, group in enumerate(ids):
                active = np.flatnonzero(gates[i, group, 1:] > 0)
                assert len(active) == 2
                assert np.count_nonzero(gates[i].sum(1)) == 1
                top = active[np.argsort(-gates[i, group, 1:][active], kind='stable')]
                weights = gates[i, group, 1:][top]
                local_logits = self.cache[(channel, int(group))][i]
                self.records.append({
                    'chart': self.chart, 'split': 'heldout', 'Re': re_value,
                    'trajectory_id': trajectory, 'window_id': f'{trajectory}:{starts[i]}',
                    'start_index': starts[i], 'start_time': times[i],
                    'horizon': horizon, 'rollout_step': step, 'stage': stage,
                    'channel': channel, 'used_in_update': channel == 'u' or stage == 1,
                    'primary_macro_sample': stage == 1,
                    'group_id': int(group), 'group_logits': group_logits[i].tolist(),
                    'group_dense_probs_float64_recomputed': group_probs[i].tolist(),
                    'expert_top_ids': top.tolist(), 'expert_top_weights': weights.tolist(),
                    'shared_weight': float(gates[i, group, 0]),
                    'combined_gates': gates[i].reshape(-1).tolist(),
                    'selected_group_expert_logits': local_logits.tolist(),
                    'selected_group_dense_probs_float64_recomputed': softmax(local_logits, self.config['temperature']).tolist(),
                })
        assert np.array_equal(*group_ids), 'Shared outer group router must agree between channels'
        self.cache.clear()

    def finish_window_batch(self, horizon):
        assert self.calls == horizon * self.stages, (self.calls, horizon, self.stages)

    def save(self, path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for record in self.records:
                f.write(json.dumps(record, allow_nan=False) + '\n')
        return len(self.records)
