import paramiko, concurrent.futures
from pathlib import Path
def probe(item):
    name,host,port,user=item
    c=paramiko.SSHClient(); c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host,port=port,username=user,key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=8,auth_timeout=8,banner_timeout=8,look_for_keys=False,allow_agent=False)
        _,o,e=c.exec_command('hostname; command -v perl; command -v python3; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader',timeout=10)
        return name,o.read().decode(),e.read().decode()
    except Exception as e:return name,type(e).__name__,str(e)
    finally:c.close()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(probe,[('4090','10.10.164.247',11219,'root'),('3090','10.10.164.243',11219,'root'),('old','10.10.164.243',20381,'root'),('VM','192.168.232.130',22,'ray')]):print(r,flush=True)
