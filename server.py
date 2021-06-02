from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('user@server:path')

with SCPClient(ssh.get_transport()) as scp:
    scp.put('my_file.txt', 'my_file.txt') # Copy my_file.txt to the server