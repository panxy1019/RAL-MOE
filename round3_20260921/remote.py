"""Scoped SSH/SFTP bridge, authenticating with the existing user key."""
import argparse, sys
from pathlib import Path
import paramiko
p=argparse.ArgumentParser();p.add_argument('action',choices=['exec','put','get']);p.add_argument('source');p.add_argument('target',nargs='?');p.add_argument('--vm',action='store_true');a=p.parse_args()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.232.130' if a.vm else '10.10.164.243',port=22 if a.vm else 20381,username='ray' if a.vm else 'root',key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=15,look_for_keys=False,allow_agent=False)
try:
    if a.action=='exec':
        cmd=Path(a.source[1:]).read_text(encoding='utf-8') if a.source.startswith('@') else a.source
        _,out,err=c.exec_command(cmd,timeout=3600)
        for line in out:print(line,end='',flush=True)
        sys.stderr.write(err.read().decode(errors='replace'))
        sys.exit(out.channel.recv_exit_status())
    else:
        with c.open_sftp() as s:
            if a.action=='put':s.put(a.source,a.target)
            else:
                Path(a.target).parent.mkdir(parents=True,exist_ok=True);s.get(a.source,a.target)
finally:c.close()
