#!/cbica/home/bertolem/anaconda3/bin/python
import numpy as np
import pandas as pd
import club
import os
import graph_metrics
import write_brains
from multiprocessing import Pool
import igraph
from igraph import VertexClustering
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
import seaborn as sns
import itertools 


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


import statsmodels.api as sm

from scipy.stats import pearsonr  
from statsmodels.stats.mediation import Mediation
import statsmodels.genmod.families.links as links
import matplotlib.pylab as plt
import matplotlib as mpl

import random

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Palatino"
plt.rcParams['font.serif'] = "Palatino"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Palatino:italic'
plt.rcParams['mathtext.bf'] = 'Palatino:bold'
plt.rcParams['mathtext.cal'] = 'Palatino'
path = '/cbica/home/bertolem/Palatino/Palatino.ttf'
mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = 'serif'
sns.set(style='white',font='Palatino')
# this is where the developmental matrices are
global matrix_path
matrix_path = '/cbica/projects/pinesParcels/dropbox/PNC_schaef400'

# load the age, executive function, motion stuff
global sub_df
sub_df = pd.read_csv('/cbica/projects/pinesParcels/dropbox/scanid_Age_Sex_Motion_EF.csv')

# make loading/saving things easier
global homedir
homedir = '/cbica/home/bertolem/diverse_dev/'

# cut off for the club
global rank_threshold
rank_threshold = 80

# from the adult HCP
global adult_pc
adult_pc = np.load('/%s/data/hcp_pc.npy'%(homedir)).mean(axis=0)
global adult_pc_club
adult_pc_club = adult_pc>np.percentile(adult_pc,rank_threshold)

# from the adult HCP
global adult_degree
adult_degree = np.load('/%s/data/hcp_strength.npy'%(homedir)).mean(axis=0)
global adult_degree_club
adult_degree_club = adult_degree>np.percentile(adult_degree,rank_threshold)

global atlas_path
atlas_path = '/%s/Schaefer2018_400Parcels_17Networks_order.dlabel.nii'%(homedir)

global d_color
global r_color
global dd_color
global rr_color
d_color = np.array([119,136,153,255])/255.
dd_color = np.array([112,128,144,255])/255.
r_color = np.array([112, 144, 144,255])/255.
rr_color = np.array([78,101,101,255])/255.


global full_width
full_width = 7.44094

def vcorrcoef(X,y):
	Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
	ym = np.mean(y)
	r_num = np.sum((X-Xm)*(y-ym),axis=1)
	r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
	r = r_num/r_den
	return r

def yeo_partition(n_networks=17):
	if n_networks == 17:
		full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,
		'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	if n_networks==7:
		full_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,
		'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
		name_dict = {0:'Visual',1:'Sensory\nMotor',2:'Dorsal\nAttention',3:'Ventral\nAttention',4:'Limbic',5:'Control',6:'Default'}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('//cbica/home/bertolem/deep_prediction//Schaefer2018_400Parcels_17Networks_order.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name']
	yeo_colors = pd.read_csv('//cbica/home/bertolem/deep_prediction//Schaefer2018_400Parcels_17Networks_order.txt',sep='\t',header=None,names=['name','r','g','b','0'])
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.
	for i,n in enumerate(yeo_df):
		if n_networks == 17:
			membership[i] = n.split('_')[2]
			membership_ints[i] = int(full_dict[n.split('_')[2]])
		if n_networks == 7:
			membership_ints[i] = int(full_dict[n.split('_')[2]])
			membership[i] = name_dict[membership_ints[i]]
	return membership,membership_ints,yeo_colors

membership,membership_ints,yeo_colors = yeo_partition(7)
names = ['VisCent','VisPeri','SomMotA','SomMotB','DorsAttnA','DorsAttnB','SalVentAttnA','SalVentAttnB','Limbic','ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','TempPar']

def load_hcp_subjects():
	df = pd.read_csv('/cbica/home/bertolem/deep_prediction/S1200.csv')
	subjects = df.Subject[df['3T_Full_MR_Compl'].values==True].values
	for s in subjects:
		if os.path.exists('/cbica/home/bertolem/deep_prediction//all_matrices/%s_matrix.npy'%(s)) == False:
			subjects = subjects[subjects!=s]
	return subjects

def load_hcp_matrices(subjects):
	matrices = []
	for s in subjects: 
		matrices.append(np.load('//cbica/home/bertolem/deep_prediction//all_matrices/%s_matrix.npy' %(s)))
	return matrices

def hcp_metrics_multi_funct(m):
	m = m + m.transpose()
	m = np.tril(m,-1)
	m = m + m.transpose()
	q = np.zeros((5))
	pc = np.zeros((5,400))
	strength = np.zeros((5,400))
	for idx,cost in enumerate([0.15,0.1,0.05,0.025,0.01]):
		graph = graph_metrics.matrix_to_igraph(m.copy(), cost, binary=False, check_tri=False, interpolation='midpoint', normalize=False, mst=True, test_matrix=False)
		vc = VertexClustering(graph,membership=membership_ints,modularity_params={'weights':'weight'})
		pc[idx] = graph_metrics.part_coef(np.array(graph.get_adjacency(attribute='weight').data),membership_ints)
		q[idx] = vc.modularity
		strength[idx] = vc.graph.strength(weights='weight')
	return np.nanmean(pc,axis=0),np.nanmean(q),np.nanmean(strength,axis=0)

def hcp_metrics():
	subjects = load_hcp_subjects()
	matrices = load_hcp_matrices(subjects)
	pool = Pool(38)
	results = pool.map(hcp_metrics_multi_funct,matrices)
	pc = []
	q = []
	s = []
	for r in results:
		pc.append(r[0])
		q.append(r[1])
		s.append(r[2])
	np.save('/%s/data/hcp_pc.npy'%(homedir),pc)
	np.save('/%s/data/hcp_q.npy'%(homedir),q)
	np.save('/%s/data/hcp_strength.npy'%(homedir),s)

def metrics_multi_funct(subject):
	matrix = np.loadtxt('%s/%s_Schaefer400_network.txt'%(matrix_path,subject))
	diverse_clubness = np.zeros((5))
	rich_clubness = np.zeros((5))
	diverse_2_other = np.zeros((5))
	rich_2_other = np.zeros((5))
	q = np.zeros((5))
	pc = np.zeros((5,400))
	strength = np.zeros((5,400))
	for idx,cost in enumerate([0.15,0.1,0.05,0.025,0.01]):
		graph = graph_metrics.matrix_to_igraph(matrix, cost, binary=False, check_tri=False, interpolation='midpoint', normalize=False, mst=True, test_matrix=False)
		vc = VertexClustering(graph,membership=membership_ints,modularity_params={'weights':'weight'})
		pc[idx] = graph_metrics.part_coef(np.array(graph.get_adjacency(attribute='weight').data),membership_ints)
		q[idx] = vc.modularity
		strength[idx] = vc.graph.strength(weights='weight')
		dc = matrix[np.ix_(adult_pc_club,adult_pc_club)][np.triu_indices(80,1)].sum()
		rc = matrix[np.ix_(adult_degree_club,adult_degree_club)][np.triu_indices(80,1)].sum()
		diverse_clubness[idx] = dc
		rich_clubness[idx] = rc
		dc_2_other = matrix[np.ix_(adult_pc_club,adult_pc_club==False)].sum()/2.
		rc_2_other = matrix[np.ix_(adult_degree_club,adult_degree_club==False)].sum()/2.
		diverse_2_other = dc_2_other
		rich_2_other = rc_2_other
	return [q,pc,strength,rich_clubness,diverse_clubness,rc_2_other,dc_2_other]

def run_metrics():
	pool = Pool(38)
	club_results = pool.map(metrics_multi_funct,sub_df['masteref.bblid'].values)
	n_subs = len(sub_df['masteref.bblid'].values)
	diverse_clubness = np.zeros((n_subs,5))
	rich_clubness = np.zeros((n_subs,5))
	diverse_other = np.zeros((n_subs,5))
	rich_other = np.zeros((n_subs,5))
	q = np.zeros((n_subs,5))
	pc = np.zeros((n_subs,5,400))
	strength = np.zeros((n_subs,5,400))
	for s in range(n_subs):
		q[s],pc[s],strength[s],rich_clubness[s],diverse_clubness[s],rich_other[s],diverse_other[s] = club_results[s]
	np.save('/%s/data/q.npy'%(homedir),q)
	np.save('/%s/data/pc.npy'%(homedir),pc)
	np.save('/%s/data/rich.npy'%(homedir),rich_clubness)
	np.save('/%s/data/diverse.npy'%(homedir),diverse_clubness)
	np.save('/%s/data/rich_other.npy'%(homedir),rich_other)
	np.save('/%s/data/diverse_other.npy'%(homedir),diverse_other)
	np.save('/%s/data/strength.npy'%(homedir),strength)

def remove(remove_me,y):
	y_model = LinearRegression().fit(remove_me,y) 
	y_predict = y_model.predict(remove_me) # predicted values
	y_residual =  y - y_predict # residual values
	return y_residual

def age_by_club_brains():
	global adult_degree_club
	global adult_pc_club
	colors = np.zeros((400,4))
	for i in range(400):
		if adult_pc_club[i] and adult_degree_club[i]:
			colors[i] = [1,1,1,1]
			continue
		if adult_pc_club[i]:
			colors[i] = d_color
		if adult_degree_club[i]:
			colors[i] = r_color
	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/clubs_mask'%(homedir))
	
	names = ['diverse','rich']
	for idx,club in enumerate([adult_pc_club,adult_degree_club]):
		colors = np.zeros((400,4))
		for i in range(len(club)):
			if idx == 0:
				if club[i]: 
					colors[i] = dd_color
			if idx == 1:
				if club[i]: 
					colors[i] = rr_color
		write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/%s_club_mask'%(homedir,names[idx]))


	ages = sub_df['masteref.Age'].values
	motion = sub_df['masteref.Motion'].values
	pc = np.load('/%s/data/pc.npy'%(homedir))
	pc = remove(motion.reshape(-1,1),pc.mean(axis=1))

	strength = np.load('/%s/data/strength.npy'%(homedir))
	strength = remove(motion.reshape(-1,1),strength.mean(axis=1))

	"""
	simple regression way, predict value at each age, save a few ages
	"""

	clf = make_pipeline(StandardScaler(), SVR(kernel='poly',degree=3))
	predicted_ages = np.linspace(np.min(ages),np.max(ages),len(range(np.min(ages),np.max(ages)))+1).reshape(-1,1)
	predicted_pc = np.zeros((400,len(predicted_ages)))

	for i in range(400):
		X,y = ages.reshape(-1,1),pc[:,i].reshape(-1,1)
		clf.fit(X, y)
		predicted_pc[i] = clf.predict(predicted_ages).flatten()

	clf = make_pipeline(StandardScaler(), SVR(kernel='poly',degree=3))
	predicted_ages = np.linspace(np.min(ages),np.max(ages),len(range(np.min(ages),np.max(ages)))+1).reshape(-1,1)
	predicted_strength = np.zeros((400,len(predicted_ages)))

	for i in range(400):
		X,y = ages.reshape(-1,1),strength[:,i].reshape(-1,1)
		clf.fit(X, y)
		predicted_strength[i] = clf.predict(predicted_ages.reshape(-1,1)).flatten()

	int_ages = np.around(predicted_ages/12,0).flatten()
	take_subs = [8,11,14,17,20,23]

	for i_age,j_age in zip(take_subs[:-1],take_subs[1:]):
		idx_mask = (int_ages>=i_age) & (int_ages<=j_age)

		d_club = np.nanmean(predicted_pc[:,idx_mask],axis=1)
		print (pearsonr(adult_pc,d_club))
		r_club = np.nanmean(predicted_strength[:,idx_mask],axis=1)
		print (pearsonr(adult_degree,r_club))
		
		d_club = d_club > np.percentile(d_club,80)
		r_club = r_club > np.percentile(r_club,80)


		# colors = np.zeros((400,4))
		# for i in range(len(r_club)):
		# 	if d_club[i] and r_club[i]:
		# 		colors[i] = [1,1,1,1]
		# 		continue
		# 	if d_club[i]:
		# 		colors[i] = d_color
		# 	if r_club[i]:
		# 		colors[i] = r_color		
		# write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/clubs_age_%s_%s'%(homedir,i_age,j_age))


		colors = np.zeros((400,4))
		for i in range(len(r_club)):
			if d_club[i]: colors[i] = dd_color
		write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/diverse_clubs_age_%s_%s'%(homedir,i_age,j_age))

		colors = np.zeros((400,4))
		for i in range(len(r_club)):
			if r_club[i]: colors[i] = rr_color
		write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/rich_clubs_age_%s_%s'%(homedir,i_age,j_age))

def basic_figure():
	q = np.load('/%s/data/q.npy'%(homedir))
	pc = np.load('/%s/data/pc.npy'%(homedir))
	rich_clubness = np.load('/%s/data/rich.npy'%(homedir))
	diverse_clubness = np.load('/%s/data/diverse.npy'%(homedir))
	
	rich_other = np.load('/%s/data/rich_other.npy'%(homedir))
	diverse_other = np.load('/%s/data/diverse_other.npy'%(homedir))
	ages = sub_df['masteref.Age'].values
	motion = sub_df['masteref.Motion'].values
	ef = sub_df['masteref.F1_Exec_Comp_Cog_Accuracy'].values
	age_r_ef = remove(ages.reshape(-1,1),ef)
	
	rich_clubness = remove(motion.reshape(-1,1),rich_clubness)
	diverse_clubness = remove(motion.reshape(-1,1),diverse_clubness)
	rich_other = remove(motion.reshape(-1,1),rich_other)
	diverse_other = remove(motion.reshape(-1,1),diverse_other)
	pc = remove(motion.reshape(-1,1),pc.mean(axis=1))
	age_pc = remove(ages.reshape(-1,1),pc)

	# ### test if clubness by age is because template networks fit better
	# diverse_clubness_q = remove(q.mean(axis=1).reshape(-1,1),diverse_clubness)
	# print (pearsonr(ages,diverse_clubness_q.mean(axis=1)))

	plt.close()
	f, (ax1, ax2) = plt.subplots(1,2, sharex=True,sharey=True)
	f.set_figwidth((full_width/3)*2)
	f.set_figheight(3)
	sns.regplot(ages/12,diverse_clubness.mean(axis=-1),ax=ax1,color=d_color,scatter_kws={'edgecolors':dd_color})  
	r = np.around(pearsonr(ages,diverse_clubness.mean(axis=-1))[0],2)   
	plt.sca(ax1)
	plt.ylabel('clubness')
	plt.xlabel('age')
	plt.text(0.1,.9,'r=%s'%(r),transform=ax1.transAxes)
	plt.sca(ax2)
	sns.regplot(ages/12,rich_clubness.mean(axis=-1),ax=ax2,color=r_color,scatter_kws={'edgecolors':rr_color})    
	r = np.around(pearsonr(ages,rich_clubness.mean(axis=-1))[0],2)   
	plt.text(0.1,.9,'r=%s'%(r),transform=ax2.transAxes)
	plt.xlabel('age')
	plt.tight_layout()
	sns.despine()
	# plt.show()
	plt.savefig('clubness.pdf')  
	
def interactions():

	q = np.load('/%s/data/q.npy'%(homedir))
	pc = np.load('/%s/data/pc.npy'%(homedir))
	rich_clubness = np.load('/%s/data/rich.npy'%(homedir))
	diverse_clubness = np.load('/%s/data/diverse.npy'%(homedir))
	ages = sub_df['masteref.Age'].values
	motion = sub_df['masteref.Motion'].values
	ef = sub_df['masteref.F1_Exec_Comp_Cog_Accuracy'].values
	age_r_ef = remove(ages.reshape(-1,1),ef)
	
	rich_clubness = remove(motion.reshape(-1,1),rich_clubness)
	diverse_clubness = remove(motion.reshape(-1,1),diverse_clubness)
	pc = remove(motion.reshape(-1,1),pc.mean(axis=1))
	age_pc = remove(ages.reshape(-1,1),pc)

	df = pd.DataFrame()
	for idx,s in enumerate(sub_df['masteref.bblid'].values):
		tdf = pd.DataFrame(columns=['sub','node','clubness','pc','age','ef_adj_adj','q'])
		tdf['node'] = np.arange(400)
		tdf['sub'] = str(s)
		tdf['pc'] = pc[idx]
		tdf['q'] = q[idx].mean(axis=0)
		tdf['clubness'] = diverse_clubness[idx].mean(axis=0)
		tdf['age'] = ages[idx]
		tdf['ef_adj_adj'] = age_r_ef[idx]
		df = df.append(tdf,ignore_index=True)


	club_pc_coefs = np.zeros((400))
	club_pc_q_coefs = np.zeros((400))
	ef_pc_coefs = np.zeros((400))
	for node in range(400):
		node_df = df[df['node'] == node]
		node_df = node_df.drop('node',axis=1)
		node_df = node_df.drop('sub',axis=1)
		
		#pc by age interaction, does clubness increase as pc and age increase together?
		model = sm.OLS.from_formula(formula='clubness ~ age + pc + age:pc', data=node_df).fit()
		result_df = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
		club_pc_coefs[node] = result_df['coef'][-1]

		#add q to make sure this is not driven by template networks fitting better
		model = sm.OLS.from_formula(formula='clubness ~ age + pc + q + age:pc', data=node_df).fit()
		result_df = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
		club_pc_q_coefs[node] = result_df['coef'][-1]

		#coefficients of PC for each node: does an increase in PC, holding clubness constant, increase EF?
		model = sm.OLS.from_formula(formula='ef_adj_adj ~ clubness + pc', data=node_df).fit()
		result_df = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
		ef_pc_coefs[node] = result_df['coef'][-1]

	print (pearsonr(club_pc_coefs,adult_pc))
	print (pearsonr(ef_pc_coefs,adult_pc))

	colors = np.array(write_brains.make_heatmap(write_brains.cut_data(club_pc_coefs,1),sns.diverging_palette(220, 10,n=1001)))
	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc:age-pc-coefs_clubness~age+pc+pc-age'%(homedir))

	# colors = np.array(write_brains.make_heatmap(write_brains.cut_data(ef_pc_coefs,1),sns.diverging_palette(220, 10,n=1001)))
	# write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_coefs_ef~clubness+pc'%(homedir))

	colors = np.array(write_brains.make_heatmap(write_brains.cut_data(adult_pc,1),sns.diverging_palette(220, 10,n=1001)))
	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/adult_pc'%(homedir))


	age_pc_corr = np.zeros((400))
	ef_pc_corr = np.zeros((400))
	q_pc_corr = np.zeros((400))
	club_pc_corr = np.zeros((400))

	for i in range(400):
		age_pc_corr[i] = pearsonr(ages,pc[:,i])[0]
		ef_pc_corr[i] = pearsonr(age_r_ef,pc[:,i])[0]
		q_pc_corr[i] = pearsonr(q.mean(axis=-1),pc[:,i])[0]
		club_pc_corr[i] = pearsonr(diverse_clubness.mean(axis=-1),pc[:,i])[0]

	colors = np.array(write_brains.make_heatmap(write_brains.cut_data(ef_pc_corr,1),sns.diverging_palette(220, 10,n=1001)))
	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/ef_pc_corr'%(homedir))


	sns.regplot(club_pc_corr,adult_pc)
	plt.ylabel('adult participation coef')
	plt.xlabel('pearson r(nodal pc,diverse clubness)')
	r = np.around(pearsonr(club_pc_corr,adult_pc)[0],2)   
	plt.text(0.1,.9,'r=%s'%(r),transform=plt.gca().transAxes)
	plt.savefig('club_pc_corr.pdf')  

	sns.regplot(age_pc_corr,adult_pc)
	plt.ylabel('adult participation coef')
	plt.xlabel('pearson r(nodal pc,age)')
	r = np.around(pearsonr(age_pc_corr,adult_pc)[0],2)   
	plt.text(0.1,.9,'r=%s'%(r),transform=plt.gca().transAxes)
	plt.savefig('age_pc_corr.pdf')  

	plt.close()
	sns.regplot(ef_pc_corr,adult_pc)
	plt.ylabel('adult participation coef')
	plt.xlabel('pearson r(nodal pc,ef)')
	r = np.around(pearsonr(ef_pc_corr,adult_pc)[0],3)   
	plt.text(0.1,.9,'r=%s'%(r),transform=plt.gca().transAxes)
	plt.savefig('ef_pc_corr.pdf')  

def full_correlations():
	matrix = []
	for subject in sub_df['masteref.bblid']:
		matrix.append(np.loadtxt('%s/%s_Schaefer400_network.txt'%(matrix_path,subject)))
	matrix = np.array(matrix)

	pc_corr_matrix = np.zeros((400,400,400))
	pc_corr_matrix[:] = np.nan
	s_corr_matrix = np.zeros((400,400,400))
	s_corr_matrix[:] = np.nan

	motion = sub_df['masteref.Motion'].values

	pc = np.load('/%s/data/pc.npy'%(homedir))
	pc = remove(motion.reshape(-1,1),pc.mean(axis=1))

	strength = np.load('/%s/data/strength.npy'%(homedir))
	strength = remove(motion.reshape(-1,1),strength.mean(axis=1))

	for i in range(400):
		pc_corr_matrix[i] = vcorrcoef(matrix.reshape(693,-1).swapaxes(0,1),pc[:,i]).reshape(400,400)
		s_corr_matrix[i] = vcorrcoef(matrix.reshape(693,-1).swapaxes(0,1),strength[:,i]).reshape(400,400)

	sns.heatmap(np.nanmean(pc_corr_matrix,axis=0),vmin=-0.04,vmax=0.04)  
	plt.xticks(np.arange(400)[::10],membership[::10]) 
	plt.yticks(np.arange(400)[::10],membership[::10]) 
		
def performance():

# qsub -l h_vmem=40G,s_vmem=40G -pe threaded 40 -N clubs -V -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/ //cbica/home/bertolem/diverse_dev/diverse_dev.py
# hcp_metrics()
# run_metrics()




























"""
GRAVEYARD OF CODE BEWARE 
"""
# sns.lineplot(x="age", y="clubness",hue="club",data=plot_df[(plot_df.club=='rich club') | (plot_df.club=='diverse club')]),palette=[sns.color_palette("Paired")[2],sns.color_palette("Paired")[1]])








	# mean_e = e.mean(axis=1).mean(axis=-1) 


	# non_diverse_mask = np.ones((400,400))
	# non_diverse_mask[np.ix_(adult_pc>np.percentile(adult_pc,20),adult_pc>np.percentile(adult_pc,20))] = 0

	# diverse_corr_community = np.zeros((np.unique(membership_ints).shape[0],np.unique(membership_ints).shape[0]))
	
	# for com in np.unique(membership_ints):
	# 	community_mask = np.zeros((400,400))
	# 	community_mask[np.ix_(np.where(membership_ints==com)[0],np.where(membership_ints==com)[0])] = 1
	# 	try:diverse_corr_community[com,com] = pearsonr(diverse_clubness.mean(axis=-1),matrix[:,community_mask.astype(bool) & non_diverse_mask.astype(bool)].mean(axis=-1))[0]
	# 	except: diverse_corr_community[com,com] = np.nan
	# for com1,com2 in itertools.combinations(np.unique(membership_ints),2):
	# 	community_mask = np.zeros((400,400))
	# 	community_mask[np.ix_(np.where(membership_ints==com1)[0],np.where(membership_ints==com2)[0])] = 1
		
	# 	try: r = pearsonr(diverse_clubness.mean(axis=-1),matrix[:,community_mask.astype(bool) & non_diverse_mask.astype(bool)].mean(axis=-1))[0]
	# 	except: r = np.nan
	# 	diverse_corr_community[com1,com2] = r 
	# 	diverse_corr_community[com2,com1] = r 

	# plt.close()
	# sns.heatmap(diverse_corr_community,annot=True,fmt=".2f",vmin=-.1,vmax=.1,cmap=sns.diverging_palette(220, 10,n=1001))
	# plt.yticks(range(16),names,rotation=360)
	# plt.xticks(range(16),names,rotation=90)
	# plt.tight_layout()



	# e_community = np.zeros((len(ages),np.unique(membership_ints).shape[0],np.unique(membership_ints).shape[0]))
	# e_age_community = np.zeros((np.unique(membership_ints).shape[0],np.unique(membership_ints).shape[0]))
	# e_mean = e.mean(axis=1)
	# for com in np.unique(membership_ints):
	# 	e_community[:,com,com] = e_mean[:,np.where(membership_ints==com)[0]][:,:,np.where(membership_ints==com)[0]].mean(axis=-1).mean(axis=-1)
	# 	r = pearsonr(ages,e_community[:,com,com])[0]
	# 	e_age_community[com,com] = r
	# 	e_age_community[com,com] = r
	# for com1,com2 in itertools.combinations(np.unique(membership_ints),2):
	# 	e_community[:,com1,com2] = e_mean[:,np.where(membership_ints==com1)[0]][:,:,np.where(membership_ints==com2)[0]].mean(axis=-1).mean(axis=-1)
	# 	r = pearsonr(ages,e_community[:,com1,com2])[0]
	# 	e_age_community[com1,com2] = r
	# 	e_age_community[com2,com1] = r

	# plt.close()
	# sns.heatmap(e_age_community,annot=True,fmt=".2f",vmin=-.1,vmax=.1,cmap=sns.diverging_palette(220, 10,n=1001))
	# plt.yticks(range(16),names,rotation=360)
	# plt.xticks(range(16),names,rotation=90)
	# plt.tight_layout()





	# eff_age_r = np.zeros(400)
	# for n in range(400):
	# 	eff_age_r[n] = pearsonr(mean_e[:,n],ages)[0]


	# sns.regplot(y=e.mean(axis=1)[:,membership_ints<8][:,:,membership_ints<8].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1),order=3)      
	# plt.xlabel('diverse strength')
	# plt.ylabel('closeness')
	# plt.savefig('diverse_close_non_fp_df.pdf')

	# plt.close()
	# sns.regplot(y=e.mean(axis=1)[:,membership_ints>=8][:,:,membership_ints>=8].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1),order=3)          
	# plt.xlabel('diverse strength')
	# plt.ylabel('closeness')
	# plt.savefig('diverse_close_fp_df.pdf')



	# matrix = remove_motion(motion.reshape(-1,1),matrix.reshape(matrix.shape[0],-1)).reshape(len(ages),400,400)

	# plt.close()
	# sns.regplot(y=matrix[:,membership_ints<8][:,:,membership_ints<8].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1),order=3)      
	# plt.xlabel('diverse strength')
	# plt.ylabel('fc strength')
	# plt.savefig('diverse_fc_non_fp_df.pdf')

	# plt.close()
	# sns.regplot(y=matrix[:,membership_ints>=8][:,:,membership_ints>=8].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1),order=3)          
	# plt.xlabel('diverse strength')
	# plt.ylabel('fc strength')
	# plt.savefig('diverse_fc_fp_df.pdf')


	# age_r_diverse_clubness = remove_motion(ages.reshape(-1,1),diverse_clubness)

	# pearsonr(y=matrix[:,membership_ints<5][:,:,membership_ints<5].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1))
	# pearsonr(y=matrix[:,membership_ints<8][:,:,membership_ints<8].mean(axis=-1).mean(axis=-1),x=ages)
	# pearsonr(y=matrix[:,membership_ints>=8][:,:,membership_ints>=8].mean(axis=-1).mean(axis=-1),x=diverse_clubness.mean(axis=-1))



	# colors = np.array(write_brains.make_heatmap(write_brains.cut_data(eff_age_r,1),sns.diverging_palette(220, 10,n=1001)))
	# write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/e_age_pc_corr'%(homedir))


	# df_cols = ['age','iteration','club','clubness']
	# plot_df = pd.DataFrame(columns=df_cols)
	# for i in range(1000):

	# 	subs = random.choices(range(0,q.shape[0]),k=q.shape[0])
	# 	q_low = lowess(q[subs].mean(axis=1),ages[subs],is_sorted=False,return_sorted=False)
	# 	d_low = lowess(diverse_clubness[subs].mean(axis=1),ages[subs],is_sorted=False,return_sorted=False)
	# 	r_low = lowess(rich_clubness[subs].mean(axis=1),ages[subs],is_sorted=False,return_sorted=False)
	# 	ages = ages[subs]

	# 	this_df =pd.DataFrame(columns=df_cols)
	# 	this_df['age'] = ages[subs]
	# 	this_df['clubness'] = q_low 
	# 	this_df['club'] = 'modularity'
	# 	this_df['iteration'] = i
	# 	plot_df = plot_df.append(this_df,ignore_index=True)

	# 	# this_df =pd.DataFrame(columns=df_cols)
	# 	# this_df['age'] = ages[subs]
	# 	# this_df['clubness'] = e[:,i].mean(axis=-1).mean(axis=-1)
	# 	# this_df['club'] = 'efficiency'
	# 	# this_df['iteration'] = i
	# 	# plot_df = plot_df.append(this_df,ignore_index=True)

	# 	this_df =pd.DataFrame(columns=df_cols)
	# 	this_df['age'] = ages[subs]
	# 	this_df['clubness'] = r_low
	# 	this_df['club'] = 'rich club'
	# 	this_df['iteration'] = i
	# 	plot_df = plot_df.append(this_df,ignore_index=True)

	# 	this_df =pd.DataFrame(columns=df_cols)
	# 	this_df['age'] = ages[subs]
	# 	this_df['clubness'] = d_low
	# 	this_df['club'] = 'diverse_club'
	# 	this_df['iteration'] = i
	# 	plot_df = plot_df.append(this_df,ignore_index=True)






	# sns.set(style='white',font='Palatino')
	# fig, ax1 = plt.subplots()
	# sns.lineplot(x=np.array(age_perm).mean(axis=0), y=np.array(q_perm).mean(axis=0),ax=ax1)
	# smooth_Q = lowess(q.mean(axis=1),ages,is_sorted=False)  
	# sns.lineplot(smooth_Q[:,0],smooth_Q[:,1],color='grey',ax=ax1)
	# plt.ylabel('modularity (Q)')

	# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# sns.lineplot(x="age", y="clubness",hue="club",data=plot_df[(plot_df.club=='rich club') | (plot_df.club=='diverse club')],ax=ax2,palette=[sns.color_palette("Paired")[2],sns.color_palette("Paired")[1]])
	# smooth_diverse = lowess(diverse_clubness.mean(axis=1),ages,is_sorted=False)
	# sns.lineplot(smooth_diverse[:,0],smooth_diverse[:,1],color=sns.color_palette("Paired")[1])
	# smooth_rich = lowess(rich_clubness.mean(axis=1),ages,is_sorted=False)  
	# sns.lineplot(smooth_rich[:,0],smooth_rich[:,1],color=sns.color_palette("Paired")[2])

	# fig.tight_layout()  # otherwise the right y-label is slightly clipped
	# plt.savefig('/%s/figures/fig1.pdf'%(homedir))




		# outcome_model = sm.OLS.from_formula(formula='clubness ~ age + pc', data=node_df)
		# mediator_model = sm.OLS.from_formula(formula='pc ~ age', data=node_df)
		# med = Mediation(outcome_model, mediator_model, 'age', 'pc').fit(n_rep=10)
		# club_pc_coefs[node] = med.total_effect.mean() / med.ACME_avg.mean() 



# def clubness(graph):
# 	rich_club_graph = graph.copy()
# 	rich_club_graph.vs.select(rich_club=False).delete()
	
# 	diverse_club_graph = graph.copy()
# 	diverse_club_graph.vs.select(diverse_club=False).delete()

# 	return [sum(rich_club_graph.es['weight']),sum(diverse_club_graph.es['weight'])]

# def run_clubness(graph):
# 	graph.vs['diverse_club'] = adult_pc_club
# 	graph.vs['rich_club'] = adult_degree_club
# 	degree_emperical_phi,pc_emperical_phi = clubness(graph)
# 	return [degree_emperical_phi,pc_emperical_phi]
# 	# degree_randomized_phis = np.zeros((niters))
# 	# pc_randomized_phis = np.zeros((niters))
# 	# for i in range(niters):
# 	# 	random_graph = club.preserve_strength(graph,randomize_topology=True,permute_strength=False)
# 	# 	random_graph.vs['diverse_club'] = adult_pc_club
# 	# 	random_graph.vs['rich_club'] = adult_degree_club
# 	# 	degree_randomized_phis[i],pc_randomized_phis[i] = clubness(random_graph)
	
# 	# pc_normalized_phis = pc_emperical_phi / pc_randomized_phis.astype(float)
# 	# degree_normalized_phis = degree_emperical_phi / degree_randomized_phis.astype(float)

# 	# return [degree_normalized_phis,pc_normalized_phis]





	# """
	# fancier thing, where we only input the data from the missing ages, then do a rolling mean
	# """
	# #we are going to make a dataframe so we can do rolling mean
	# rolling_df = pd.DataFrame(columns=['region','pc','age'])
	# for i in range(400):
	# 	rolling_df = rolling_df.append(pd.DataFrame(np.array([np.zeros((ages.shape[0]))+i,pc[:,i],ages]).transpose(),columns=['region','pc','age']),ignore_index=True)

	# #some ages are missing, use linear regression to predict them
	# all_ages=np.arange(np.min(ages),np.max(ages))
	# missing_ages = []
	# for a in all_ages:
	# 	if a not in ages:missing_ages.append(a)

	# missing_pc = np.zeros((400,len(missing_ages)))
	# for i in range(400):
	# 	pc_predict = LinearRegression()
	# 	pc_predict = pc_predict.fit(ages.reshape(-1,1),pc[:,i].reshape(-1,1))
	# 	missing_pc[i] = pc_predict.predict(np.array(missing_ages).reshape(-1,1)).flatten()
	# #add in the missing pc
	# for i in range(400):
	# 	rolling_df = rolling_df.append(pd.DataFrame(np.array([np.zeros((len(missing_ages)))+i,missing_pc[i],np.array(missing_ages)]).transpose(),columns=['region','pc','age']),ignore_index=True)

	# #sort by age
	# rolling_df = rolling_df.sort_values(by=['age'])
	
	# #roll it out
	# mean_age_pc = []
	# for i in range(400):
	# 	mean_age_pc.append(rolling_df[rolling_df.region==i].rolling(5*12).mean().pc.values)
	# mean_ages =  np.around(rolling_df[rolling_df.region==i].rolling(5*12).mean().age.values/12.,0)

	# mean_ages[np.isnan(mean_ages)] = 8
	# mean_ages = mean_ages.astype(int).astype(str)
	# mean_age_pc = np.array(mean_age_pc)

	# take_subs = np.linspace(0,710,4).astype(int)
	# for i,j in zip(take_subs[:-1],take_subs[1:]):
	# 	i_age = mean_ages[i]
	# 	j_age = mean_ages[j]
	# 	print (i_age,j_age)
	# 	this_pc = np.nanmean(mean_age_pc[:,i:j],axis=1)
	# 	print (pearsonr(adult_pc.mean(axis=0),this_pc))
	# 	colors = np.array(write_brains.make_heatmap(write_brains.cut_data(this_pc,1),sns.diverging_palette(220, 10,n=1001)))
	# 	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_age_%s_%s'%(homedir,i_age,j_age))

	# #save files
	# plotted_ages = np.around(rolling_df[rolling_df.region==i].rolling(24).mean().age.values[take_ages]/12.,0)
	# for i in range(5):
	# 	print (pearsonr(adult_pc.mean(axis=0),mean_age_pc.transpose()[i]))
	# 	colors = np.array(write_brains.make_heatmap(write_brains.cut_data(mean_age_pc.transpose()[i],1),sns.diverging_palette(220, 10,n=1001)))
	# 	write_brains.write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_age_%s'%(homedir,plotted_ages[i]))
