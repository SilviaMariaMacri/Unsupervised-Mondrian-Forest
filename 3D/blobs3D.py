import FileFinale
from sklearn import datasets

import sys
blobs3D,lifetime,exp,metric,number_of_iterations= sys.argv

name = 'blobs3D_lambda'+lifetime+'_exp'+exp+'_'
varied = datasets.make_blobs(n_samples=200,n_features=3,cluster_std=[1.0, 2.5, 0.5],random_state=10)#random_state=150
X=varied[0]
y=varied[1]
t0 = 0


FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )



'''
Traceback (most recent call last):
  File "blobs3D.py", line 19, in <module>
    int(number_of_iterations)
  File "/home/STUDENTI/silviamaria.macri/FileFinale.py", line 45, in FunzioneFinale
    list_m_leaf,list_p = trova_partizioni_vicine.Classification_BU(m,part,metric)
  File "/home/STUDENTI/silviamaria.macri/trova_partizioni_vicine.py", line 628, in Classification_BU
    m_leaf,p =  MergePart_SingleData(m,part)
  File "/home/STUDENTI/silviamaria.macri/trova_partizioni_vicine.py", line 337, in MergePart_SingleData
    data2 = pd.DataFrame(m_leaf[p[p['part_number']==j].index[0]]).copy()
  File "/home/STUDENTI/silviamaria.macri/.local/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 4604, in
__getitem__
    return getitem(key)
IndexError: index 0 is out of bounds for axis 0 with size 0







part_leaf_true = part.query('leaf==True')
part_leaf_true.index = np.arange(len(part_leaf_true))
for i in range(len(part_leaf_true)):
    for j in part_leaf_true['neighbors'].iloc[i]:
        if part[part['part_number']==j]['leaf'][0]==False:
            print(part_leaf_true['part_number'].iloc[i],'   ',j)'''