"""Scoped SSH/SFTP helper. Credentials are read only from the environment."""
import argparse
import os
import sys
import paramiko

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['exec', 'get', 'put'])
p.add_argument('source')
p.add_argument('destination', nargs='?')
a = p.parse_args()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.10.164.243', port=20381, username='root',
          password=os.environ['FIELD_SSH_PASSWORD'], timeout=20)
try:
    if a.mode == 'exec':
        _, out, err = c.exec_command(a.source, timeout=120)
        sys.stdout.buffer.write(out.read())
        sys.stderr.buffer.write(err.read())
        sys.exit(out.channel.recv_exit_status())
    else:
        with c.open_sftp() as s:
            if a.mode == 'get':
                s.get(a.source, a.destination)
            else:
                s.put(a.source, a.destination)
finally:
    c.close()
