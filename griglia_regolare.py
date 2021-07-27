# griglia regolare

x = np.arange(-1.5,2,0.6)
y = np.arange(-2.5,1.5,0.5)





xmin = []
xmax = []
ymin = []
ymax = []
vertici = []
box = []
part_number = []
count=0
for i in range(len(x)-1):
	x0 = x[i]
	x1 = x[i+1]
	for j in range(len(y)-1):
		y0 = y[j]
		y1 = y[j+1]
		vertici.append([[[x0,y0],[x1,y0]],[[x1,y0],[x1,y1]],[[x1,y1],[x0,y1]],[[x0,y1],[x0,y0]]])
		box.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
		part_number.append(count)
		count += 1
		xmin.append(x0)
		xmax.append(x1)
		ymin.append(y0)
		ymax.append(y1)

part = {'part_number':part_number,'min0':xmin,'max0':xmax,'min1':ymin,'max1':ymax,'vertici':vertici,'box':box}	
part = pd.DataFrame(part)
part['leaf'] = True



		

neighbors = []
	
for i in range(len(part)):
	
	p = part.copy()	
	
	for j in range(2):
		p['min'+str(j)+'_'+str(i)] = p['min'+str(j)].iloc[i]
		p['max'+str(j)+'_'+str(i)] = p['max'+str(j)].iloc[i]
			
	p=p.eval('vicino = ( ((min0==max0_'+str(i)+') and (min1==min1_'+str(i)+') and (max1==max1_'+str(i)+')) or ((max0==min0_'+str(i)+') and (min1==min1_'+str(i)+') and (max1==max1_'+str(i)+')) or ((min1==max1_'+str(i)+') and (min0==min0_'+str(i)+') and (max0==max0_'+str(i)+')) or ((max1==min1_'+str(i)+') and (min0==min0_'+str(i)+') and (max0==max0_'+str(i)+')))' )
		
	p=p.query('vicino==True')
			
	
		
	neighbors.append(list(p['part_number']))
	
	

part['neighbors'] = neighbors




sns.set_style('whitegrid')
fig,ax = plt.subplots()
		
	
for i in range(len(part)):
	box_new = part['box'].iloc[i]
	p = Polygon(box_new, facecolor = 'none', edgecolor='b')
	ax.add_patch(p)
			
	b = pd.DataFrame(box_new)
	x_avg = np.mean(b[0])
	y_avg = np.mean(b[1])
	ax.text(x_avg,y_avg,part['part_number'].iloc[i])
			
			
	ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
	
	
	
	