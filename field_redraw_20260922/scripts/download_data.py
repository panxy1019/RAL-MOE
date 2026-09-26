"""Download only this task's new snapshots and topology; no source assets changed."""
import os
from pathlib import Path
import paramiko
p=Path(__file__).resolve().parents[1]/'data';p.mkdir(exist_ok=True)
remote='/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/field_redraw_20260922/data'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.10.164.243',port=20381,username='root',password=os.environ['FIELD_SSH_PASSWORD'],timeout=20)
try:
 with c.open_sftp() as s:
  for name in s.listdir(remote):
   if name.endswith('.json') or (name.endswith('.npz') and not (p/name).exists()):
    s.get(remote+'/'+name,str(p/name));print(name,(p/name).stat().st_size)
finally:c.close()
