"""User-requested parallel scheduling; preserve frozen training and test protocol."""
import os, signal, threading, subprocess, sys, shutil, traceback
from pathlib import Path
import fno_experiment as f

original = f.FOUT
parallel = f.OUT / 'fno_P_parallel2026'
pid = 64616
expected = str(f.OUT / 'code/fno_experiment.py')
assert expected in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
assert not (original/'formal/1600/summary.json').exists(), 'Reinspect scheduling before launching'
assert not parallel.exists(), 'Do not launch duplicate worker'
parallel.mkdir()
for name in ['protocol_frozen.json', 'normalization.npz', 'raw_development.npy']:
    (parallel/name).symlink_to(original/name)
f.write(parallel/'scheduling.json', dict(started=f.time.strftime('%FT%T%z'),
    original_campaign_pid=pid, seed=2026, reason='explicit user requested parallel seed2026',
    training_protocol_changed=False, timing_note='training overlaps seed1600; final latency remains serial'))

def retire_serial_campaign():
    # Wait for the completed seed1600 summary, never interrupt that training.
    target=original/'formal/1600/summary.json'
    while not target.exists():
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Original campaign exited before seed1600 summary')
        f.time.sleep(.2)
    # Original process may have begun its redundant 2026 initialization;
    # outputs are isolated from this worker and are retained below.
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    assert expected in command
    os.kill(pid, signal.SIGTERM)
    for _ in range(300):
        if not Path(f'/proc/{pid}').exists():break
        f.time.sleep(.2)
    assert not Path(f'/proc/{pid}').exists(), 'Campaign termination not confirmed'
    f.write(parallel/'serial_campaign_retired.json',dict(time=f.time.strftime('%FT%T%z'),
        seed1600_summary=str(target),reason='avoid duplicate seed2026 and resume testing after parallel completion'))

errors=[]
def watch():
    try: retire_serial_campaign()
    except BaseException: errors.append(traceback.format_exc())
watcher=threading.Thread(target=watch,daemon=True)
watcher.start()
try:
    data=f.Data()  # Reads original training-only cache and normalization.
    protocol=f.json.loads((original/'protocol_frozen.json').read_text())
    cfg=f.json.loads((original/'selected_config.json').read_text())['architecture']
    f.FOUT=parallel
    f.train(data,cfg,2026,'formal',protocol['formal_steps'],protocol['formal_eval_every'])
    del data
    f.gc.collect();f.torch.cuda.empty_cache()
    watcher.join()
    if errors:raise RuntimeError(errors[0])
    target=original/'formal/2026'
    if target.exists():
        target.rename(parallel/'redundant_serial_2026_retained')
    shutil.copytree(parallel/'formal/2026',target)
    f.write(parallel/'PROMOTED.json',dict(time=f.time.strftime('%FT%T%z'),destination=str(target)))
    # All three summaries now exist: campaign skips training and runs tests.
    subprocess.run([sys.executable,expected,'campaign'],check=True)
    if (original/'PREDICTIONS_COMPLETED.json').exists():
        subprocess.run([sys.executable,str(f.OUT/'code/finalize_fno.py')],check=True)
except BaseException:
    f.write(parallel/'ERROR.json',dict(time=f.time.strftime('%FT%T%z'),traceback=traceback.format_exc()))
    raise
