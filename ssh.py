import paramiko   
from scp import SCPClient

# declare credentials   
host = '137.204.48.15'   
username = 'silviamaria.macri@studio.unibo.it'   
password = 'd(PF66gn'   
port = 22





ssh = paramiko.SSHClient()
ssh.load_system_host_keys() 
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)


#cd  "C:\Users\silvi\Desktop\Fisica\TESI\tesi"
#copy the file accross  
#with SCPClient(ssh.get_transport()) as scp:
#	scp.put('Matrix.py', 'Matrix.py')

#%%
command = "rm prova1.py" #rimuovere file
stdin, stdout, stderr = ssh.exec_command(command)
lines = stdout.readlines()
print(lines)

with SCPClient(ssh.get_transport()) as scp:
	scp.put('prova1.py','prova1.py')

command = "python prova1.py"
stdin, stdout, stderr = ssh.exec_command(command)
lines = stdout.readlines()
print(lines)



#%%
from fabric.api import env, run

env.host_string = '137.204.48.15'
env.user = 'silviamaria.macri@studio.unibo.it'
env.password = 'd(PF66gn'

run('ls -l')