import os

import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, key_file):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.RSAKey.from_private_key_file(key_file)

    client.connect(server, port, user, pkey=private_key)
    return client
def scp_transfer_files(ssh_client, local_path, remote_path):
    with SCPClient(ssh_client.get_transport()) as scp:
        # Check if the local_path is a directory and use recursive if it is
        if os.path.isdir(local_path):
            scp.put(local_path, remote_path, recursive=True)  # Upload the entire directory
        else:
            scp.put(local_path, remote_path)  # Upload a single file

# ssh_client = create_ssh_client('192.168.1.109', 22, 'bonsaiheart', '/home/bonsai/.ssh/id_rsa')
# scp_transfer_files(ssh_client, '/home/bonsai/Python_Projects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv',
#                            r"PycharmProjects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv")
# ssh_client.close()