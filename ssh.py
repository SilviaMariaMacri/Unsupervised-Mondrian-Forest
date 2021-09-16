import paramiko   
from scp import SCPClient

# declare credentials   
host = '137.204.48.15'   
username = 'silviamaria.macri@studio.unibo.it'   
password = 'd(PF66gn'   

# connect to server   
con = paramiko.SSHClient()   
con.load_system_host_keys()   
con.connect(host, username=username, password=password)

   
# copy the file accross  
with SCPClient(con.get_transport()) as scp:   
    scp.put('prova1.py', 'prova_copia.py')