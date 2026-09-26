from pathlib import Path
s=Path(__file__).with_name('train_h3h4.py').read_text(encoding='utf8')
for token in ('H3H4_strict_train_only','fluc_modal_u','nf_derivative','gradient fail closed','Hopf-aware'):
    assert token in s, token
assert 'HELDOUT' in s and 'evaluate_heldout' not in s
print('H3H4_STATIC_PASS')
