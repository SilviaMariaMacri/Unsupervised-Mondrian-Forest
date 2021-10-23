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
command = "rm trova_partizioni_vicine.py" #rimuovere file
stdin, stdout, stderr = ssh.exec_command(command)
lines = stdout.readlines()
print(lines)

with SCPClient(ssh.get_transport()) as scp:
	scp.put('trova_partizioni_vicine.py','trova_partizioni_vicine.py')

command = "python3 trova_partizioni_vicine.py"
stdin, stdout, stderr = ssh.exec_command(command)
lines1 = stdout.readlines()
print(lines1)
lines2 = stderr.readlines()
print(lines2)

#%% aggiungere box 

vertices = []
for i in range(len(part)):
    poly = part['polytope'][i]
    vert = pypoman.compute_polytope_vertices(np.array(poly['A']),np.array(poly['b']))
    # ordino vertici: (per più dimensioni da errore?)
    # compute centroid
    cent=(sum([v[0] for v in vert])/len(vert),sum([v[1] for v in vert])/len(vert))
    # sort by polar angle
    vert.sort(key=lambda v: math.atan2(v[1]-cent[1],v[0]-cent[0]))
    vertices.append(vert)
part['box'] = vertices

