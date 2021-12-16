#!/cbica/home/bertolem/anaconda3/bin/python
# our tools
import pennlinckit
from pennlinckit import utils,brain,data,plotting
#basic python modules
import sys
import os
#
assert sys.executable == '/cbica/home/bertolem/anaconda3/bin/python' #make sure you are using my python
import numpy as np
import pandas as pd
import itertools
from multiprocessing import Pool
import random

global homedir
import matplotlib.pylab as plt
import seaborn as sns
import time

#statsmodels / stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.mediation import Mediation
import statsmodels.genmod.families.links as links
from scipy.stats import pearsonr,spearmanr

#sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.cluster import KMeans

global homedir
homedir = '/cbica/home/bertolem/diverse_development/'
global atlas_path
atlas_path = '/{0}/Schaefer2018_400Parcels_17Networks_order.dlabel.nii'.format(homedir)
global rank_threshold
rank_threshold = 80

global d_color
global r_color
global dd_color
global rr_color
d_color = np.array([119,158,203,255])/255.
r_color = np.array([120,162,123,255])/255.

def make_data(source,cores=20):
	"""
	Make the datasets and run network metrics
	"""
	data = pennlinckit.data.dataset(source,task='**', parcels='Schaefer417',fd_scrub=.2)
	data.load_matrices()
	data.filter(way='>',value=100,column='n_frames')
	score_bdp(data)
	data.network = pennlinckit.network.make_networks(data,yeo_partition=7,cores=cores-1)
	pennlinckit.utils.save_dataset(data,'/{0}/data/{1}.data'.format(homedir,source))

def submit_make_data(source):
	"""
	The above function makes the datasets (including the networks) we are going to use to generate uncomment out the code below and
	"""
	script_path = '/{0}/diverse_development.py make_data {1}'.format(homedir,source) #it me
	pennlinckit.utils.submit_job(script_path,'d_{0}'.format(source),RAM=40,threads=20)

def load_data(source,filters=['bpd_score','meanFD']):
	data = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,source))
	gender = np.zeros((data.subject_measures.shape[0]))
	if source == 'hcpya':
		gender[data.subject_measures.Gender=='F'] = 1
	if source == 'hcpd-dcan':
		gender[data.subject_measures.sex=='F'] = 1
	data.subject_measures['gender_dummy'] = gender
	for f in filters:
		data.filter(way='has_subject_measure',value=f)
	return data

def load_pnc():
	"""
	A function to load the PNC data and the network data I made above.
	This ensures I load the same filters everytime!
	"""
	data = pennlinckit.data.dataset('pnc')
	data.load_matrices('rest')
	data.filter('cognitive')
	data.filter('==',value=0,column='restRelMeanRMSMotionExclude')
	data.network = pennlinckit.utils.load_dataset('/%s/data/pnc.networks'%(homedir))
	data.measures['sex'] = data.measures['sex'].values - 1 #makes this easy to do regression
	return data

def remove_motion_sex(data,y):
	"""
	removes motion and sex from y in dataobject data
	"""
	motion = pnc.measures['restRelMeanRMSMotion'].values
	sex = pnc.measures.sex
	noise = np.array([motion,sex]).transpose()
	y_residuals = pennlinckit.utils.remove(noise,y)
	return y_residuals

def load_hcp_metrics():
	"""
	loads previously made network data object
	"""
	hcp = pennlinckit.utils.load_dataset('/%s/data/hcp.networks'%(homedir))
	hcp.pc = hcp.pc.mean(axis=0).mean(axis=0)
	hcp.strength = hcp.strength.mean(axis=0).mean(axis=0)
	hcp.diverse_club = hcp.pc > np.percentile(hcp.pc,rank_threshold)
	hcp.rich_club = hcp.strength > np.percentile(hcp.strength,rank_threshold)
	return hcp

def hcp_club_brains():
	"""
	makes the brains that have the HCP diverse and rich club
	"""
	hcp = load_hcp_metrics()
	colors = np.zeros((400,4))
	for i in range(400):
		if hcp.diverse_club[i] and hcp.rich_club[i]:
			colors[i] = [1,1,1,1]
			continue
		if hcp.diverse_club[i]:
			colors[i] = d_color
		if hcp.rich_club[i]:
			colors[i] = r_color
	out_path='/%s/brains/clubs_mask'%(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

	names = ['diverse','rich']
	for idx,club in enumerate([hcp.diverse_club,hcp.rich_club]):
		colors = np.zeros((400,4))
		for i in range(len(club)):
			if idx == 0:
				if club[i]:
					colors[i] = d_color
			if idx == 1:
				if club[i]:
					colors[i] = r_color
		out_path='/%s/brains/%s_clubs_mask'%(homedir,names[idx])
		pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

def age_by_club_brains():
	"""
	The most obvious figure, make a rich and diverse club for each age bracket
	simple regression way, predict value at each age, save a few ages
	"""

	pnc = load_pnc()
	hcp = load_hcp_metrics()
	pc = remove_motion_sex(pnc,pnc.network.pc.mean(axis=1))
	strength = remove_motion_sex(pnc,pnc.network.strength.mean(axis=1))
	ages = pnc.measures['ageAtScan1'].values

	clf = make_pipeline(StandardScaler(), SVR(kernel='poly',degree=3))
	predicted_ages = np.linspace(np.min(ages),np.max(ages),len(range(np.min(ages),np.max(ages)))+1).reshape(-1,1)
	predicted_pc = np.zeros((400,len(predicted_ages)))

	for i in range(400):

		X,y = ages.reshape(-1,1),pc[:,i].flatten()
		clf.fit(X, y)
		predicted_pc[i] = clf.predict(predicted_ages).flatten()

	clf = make_pipeline(StandardScaler(), SVR(kernel='poly',degree=3))
	predicted_ages = np.linspace(np.min(ages),np.max(ages),len(range(np.min(ages),np.max(ages)))+1).reshape(-1,1)
	predicted_strength = np.zeros((400,len(predicted_ages)))

	for i in range(400):
		X,y = ages.reshape(-1,1),strength[:,i].flatten()
		clf.fit(X, y)
		predicted_strength[i] = clf.predict(predicted_ages.reshape(-1,1)).flatten()

	int_ages = np.around(predicted_ages/12,0).flatten()
	take_subs = [8,11,14,17,20,23]

	for i_age,j_age in zip(take_subs[:-1],take_subs[1:]):
		idx_mask = (int_ages>=i_age) & (int_ages<=j_age)

		d_club = np.nanmean(predicted_pc[:,idx_mask],axis=1)
		print (pearsonr(hcp.pc,d_club))
		r_club = np.nanmean(predicted_strength[:,idx_mask],axis=1)
		print (pearsonr(hcp.strength,r_club))

		d_club = d_club > np.percentile(d_club,rank_threshold)
		r_club = r_club > np.percentile(r_club,rank_threshold)

		colors = np.zeros((400,4))
		for i in range(len(r_club)):
			if d_club[i]: colors[i] = d_color

		out_path='/%s/brains/diverse_clubs_age_%s_%s'%(homedir,i_age,j_age)
		pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')


		colors = np.zeros((400,4))
		for i in range(len(r_club)):
			if r_club[i]: colors[i] = r_color
		out_path='/%s/brains/rich_clubs_age_%s_%s'%(homedir,i_age,j_age)
		pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

def clubness(hcp,other):
	rich_clubness = other.matrix[np.ix_(np.arange(other.matrix.shape[0]),hcp.rich_club,hcp.rich_club)].sum(axis=-1).sum(axis=-1)
	diverse_clubness = other.matrix[np.ix_(np.arange(other.matrix.shape[0]),hcp.diverse_club,hcp.diverse_club)].sum(axis=-1).sum(axis=-1)
	return rich_clubness,diverse_clubness

def basic_figure():
	pnc = load_pnc()
	hcp = load_hcp_metrics()
	ages = pnc.measures['ageAtScan1'].values
	motion = pnc.measures['restRelMeanRMSMotion'].values
	rich_clubness,diverse_clubness = clubness(hcp,pnc)
	rich_clubness = remove_motion_sex(pnc,rich_clubness)
	diverse_clubness = remove_motion_sex(pnc,diverse_clubness)

	# %matplotlib inline
	plt.close()
	f, (ax1, ax2) = plt.subplots(1,2, sharex=True,sharey=True)
	f.set_figwidth(8/3.*2)
	f.set_figheight(3)
	sns.regplot(ages/12,diverse_clubness,ax=ax1,color=d_color,scatter_kws={'edgecolors':d_color,'alpha':.25})
	r = np.around(pearsonr(ages,diverse_clubness)[0],2)
	plt.sca(ax1)
	plt.ylabel('clubness')
	plt.xlabel('age')
	plt.text(0.1,.9,'r=%s'%(r),transform=ax1.transAxes)
	plt.sca(ax2)
	sns.regplot(ages/12,rich_clubness,ax=ax2,color=r_color,scatter_kws={'edgecolors':r_color,'alpha':.25})
	r = np.around(pearsonr(ages,rich_clubness)[0],2)
	plt.text(0.1,.9,'r=%s'%(r),transform=ax2.transAxes)
	plt.xlabel('age')
	plt.tight_layout()
	sns.despine()
	plt.savefig('/%s/figures/clubness.pdf'%(homedir))
	plt.show()

def pc_clubness_interactions():

	pnc = load_pnc()
	hcp = load_hcp_metrics()
	ages = pnc.measures['ageAtScan1'].values
	motion = pnc.measures['restRelMeanRMSMotion'].values
	rich_clubness,diverse_clubness = clubness(hcp,pnc)
	rich_clubness = remove_motion_sex(pnc,rich_clubness)
	diverse_clubness = remove_motion_sex(pnc,diverse_clubness)
	pc = remove_motion_sex(pnc,pnc.network.pc.mean(axis=1))
	strength = remove_motion_sex(pnc,pnc.network.strength.mean(axis=1))
	q = remove_motion_sex(pnc,pnc.network.modularity.mean(axis=1))

	df = pd.DataFrame()
	for idx,s in enumerate(pnc.measures.subject.values):
		tdf = pd.DataFrame(columns=['sub','node','clubness','pc','age','q'])
		tdf['node'] = np.arange(400)
		tdf['sub'] = str(s)
		tdf['pc'] = pc[idx]
		tdf['q'] = q[idx]
		tdf['clubness'] = diverse_clubness[idx]
		tdf['age'] = ages[idx]
		df = df.append(tdf,ignore_index=True)


	club_pc_coefs = np.zeros((400))
	club_pc_q_coefs = np.zeros((400))
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

	age_pc_corr = np.zeros((400)) #regions get stronger pc with age
	club_pc_corr = np.zeros((400)) #regions get stronger pc with clubness

	for i in range(400):
		age_pc_corr[i] = pearsonr(ages,pc[:,i])[0]
		club_pc_corr[i] = pearsonr(diverse_clubness,pc[:,i])[0]

	print (pearsonr(club_pc_coefs,hcp.pc))
	print (pearsonr(club_pc_corr,hcp.pc))

	spincorrs = pennlinckit.brain.spin_test(hcp.pc,club_pc_coefs,n=1000)
	spin_stat = pennlinckit.brain.spin_stat(hcp.pc,club_pc_coefs,spincorrs)


	# %matplotlib inline
	plt.close()
	f,axes = plt.subplots(1,2,figsize=(5.5,3))
	sns.regplot(x=club_pc_coefs,y=hcp.pc,ax=axes[0],truncate=False,x_jitter=.2,scatter_kws={"s": 50,'alpha':0.35})
	plt.sca(axes[0])
	r,p = pearsonr(hcp.pc,club_pc_coefs)
	plt.text(.25,.035,'r={0},p={1}'.format(np.around(r,2),np.around(p,4)))
	plt.ylabel('participation coef')
	plt.xlabel('clubness~age:pc interaction')
	sns.histplot(spincorrs,ax=axes[1])
	plt.sca(axes[1])
	plt.vlines(r,0,100,colors='black')
	plt.tight_layout()
	sns.despine()
	plt.savefig('/{0}/figures/pc_age_interaction.pdf'.format(homedir))
	plt.show()



	colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(club_pc_coefs,1),sns.diverging_palette(220, 10,n=1001)))
	out_path='/%s/brains/clubness~age+pc+age:pc'%(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

	colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(hcp.pc,1),sns.diverging_palette(220, 10,n=1001)))
	out_path='/%s/brains/hcp_pc'%(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

def region_predict(data,region,factor,**model_args):
	data.targets = data.measures[factor].values
	data.features = data.matrix[:,region]
	model_args['self'] = data
	pennlinckit.utils.predict(**model_args)

def predict(factor):
	data = load_pnc()
	prediction = np.zeros((400,len(data.measures.subject)))
	for node in range(400):
		region_predict(data,node,factor,**{'model':'deep','cv':'KFold','folds':5,'neurons':400,'layers':10,'remove_linear_vars':['restRelMeanRMSMotion','sex']})
		# region_predict(data,node,factor,**{'model':'ridge','cv':'KFold','folds':5,'remove_linear_vars':['restRelMeanRMSMotion','sex']})
		prediction[node] = data.prediction
	prediction_acc = np.zeros(400)
	for node in range(data.matrix.shape[-1]):
		prediction_acc[node] = pearsonr(prediction[node],data.corrected_targets)[0]
	np.save('{0}/data/deep/prediction_{1}.npy'.format(homedir,factor),prediction)
	np.save('{0}/data/deep/prediction_acc_{1}.npy'.format(homedir,factor),prediction_acc)
	np.save('{0}/data/deep/prediction_regressed_targets_{1}.npy'.format(homedir,factor),data.corrected_targets)

def submit_predict():
	"""
	The above function makes the predictions for each factor
	this submit it
	"""
	factors = ['mood_4factorv2','psychosis_4factorv2', 'externalizing_4factorv2', 'phobias_4factorv2','overall_psychopathology_4factorv2'] #clincal factors
	for f in factors:
		script_path = '/cbica/home/bertolem/diverse_development/diverse_development.py predict {0}'.format(f) #it me
		pennlinckit.utils.submit_job(script_path,f,RAM=10,threads=1)
	factors = ['F1_Exec_Comp_Res_Accuracy_RESIDUALIZED','F2_Social_Cog_Accuracy_RESIDUALIZED','F3_Memory_Accuracy_RESIDUALIZED'] #cogi factors
	for f in factors:
		script_path = '/cbica/home/bertolem/diverse_development/diverse_development.py predict {0}'.format(f) #it me
		pennlinckit.utils.submit_job(script_path,f,RAM=10,threads=1)

def flexibility(factors='clinical'):
	data = load_pnc()
	hcp = load_hcp_metrics()
	binary = False
	# factorset = 'clinical'
	factorset = 'cognitive'
	if factorset == 'cognitive':
		factors = ['F1_Exec_Comp_Res_Accuracy_RESIDUALIZED','F2_Social_Cog_Accuracy_RESIDUALIZED','F3_Memory_Accuracy_RESIDUALIZED'] #cogi factors
	if factorset == 'clinical':
		factors = ['mood_4factorv2','psychosis_4factorv2', 'externalizing_4factorv2', 'phobias_4factorv2','overall_psychopathology_4factorv2'] #clincal factors
	all_factor_predictions = np.zeros((len(factors),data.matrix.shape[-1],data.matrix.shape[0]))
	prediction_acc = np.zeros((len(factors),data.matrix.shape[-1]))
	for fidx, factor in enumerate(factors):
		prediction_acc[fidx] = np.load('/{0}/data/linear/prediction_acc_{1}.npy'.format(homedir,factor))
		all_factor_predictions[fidx] = np.load('/{0}/data/linear/prediction_{1}.npy'.format(homedir,factor))

	min = prediction_acc.mean()
	max = prediction_acc.mean() + prediction_acc.std() *2

	if binary == True:
		flexible_nodes = np.zeros((400))
		high_predict = prediction_acc.mean() + prediction_acc.std()
		flexible_nodes =flexible_nodes+ (prediction_acc>high_predict).sum(axis=0)
	if binary == False:
		flexible_nodes = np.zeros((400))
		for high_predict in np.linspace(min,max,8):
			flexible_nodes = flexible_nodes + (prediction_acc>high_predict).sum(axis=0)

	from pennlinckit import plotting
	spincorrs = pennlinckit.brain.spin_test(hcp.pc,flexible_nodes,n=1000)
	spin_stat = pennlinckit.brain.spin_stat(hcp.pc,flexible_nodes,spincorrs)

	# %matplotlib inline
	plt.close()
	f,axes = plt.subplots(1,2,figsize=(5.5,3))
	sns.regplot(x=flexible_nodes,y=hcp.pc,ax=axes[0],truncate=False,x_jitter=.2,scatter_kws={"s": 50,'alpha':0.45})
	plt.sca(axes[0])
	r,l,h,p = pennlinckit.utils.bootstrap_corr(flexible_nodes,hcp.pc,pearsonr,10000)
	plt.text(1,.01,'r={0},p={1}\n95%CI:{2},{3}'.format(np.around(r,2),np.around(p,4),np.around(l,2),np.around(h,2)),transform=axes[0].transAxes,horizontalalignment='right',verticalalignment='bottom')
	plt.ylabel('participation coef')
	plt.xlabel('predict flex index')
	sns.histplot(spincorrs,ax=axes[1])
	plt.sca(axes[1])
	plt.vlines(r,0,100,colors='black')
	plt.text(r-.01,100,'p={0}'.format(np.around(spin_stat,4)),horizontalalignment='right',verticalalignment='top',rotation=90)
	plt.tight_layout()
	sns.despine()
	plt.savefig('/{0}/figures/flex_{1}_{2}.pdf'.format(homedir,factorset,binary))
	plt.show()

	if binary == False:
		colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(flexible_nodes,1.5),sns.diverging_palette(220, 10,n=1001)))
		out_path='/{0}/brains/flexible_nodes_{1}'.format(homedir,factorset)
		pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')


if len(sys.argv) > 1:
	if sys.argv[1] == 'make_data': make_data(sys.argv[2])
	if sys.argv[1] == 'predict': predict(sys.argv[2])


"""
order to run this!

submit_make_data()

hcp_club_brains()

age_by_club_brains()

basic_figure()

"""





# def cluster_club_changes(n_components=5):
# 	n_components= 2
# 	pnc = load_pnc()
# 	pnc.pc = remove_motion_sex(pnc,pnc.network.pc.mean(axis=1))
# 	age_pc_corr = np.zeros(400)
# 	for i in range(400):
# 		age_pc_corr[i] = pearsonr(pnc.pc[:,i],pnc.measures.ageAtScan1.values)[0]
#
# 	km = KMeans(n_clusters=n_components)
# 	km.fit(age_pc_corr.reshape(-1,1))
# 	colors = np.array(pennlinckit.utils.make_heatmap(km.labels_,sns.diverging_palette(220, 10,n=1001)))
# 	out_path='/{0}/brains/cluster_pc'.format(homedir)
# 	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')
#
# 	colors = np.array(pennlinckit.utils.make_heatmap(age_pc_corr,sns.diverging_palette(220, 10,n=1001)))
# 	out_path='/{0}/brains/age_pc_corr'.format(homedir)
# 	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')


# data = pennlinckit.data.dataset('hcp')
# data.measures.columns.values
# neo_plus = [14,39,54,9,59,24,29,45,40,5,55,30,50]
# neo_neg = [22,36,31,21,26,41,51,11,8,28,33]
#
# neo_plus_scores = []
# neo_neg_scores = []
#
# for n in neo_plus:
# 	s = df['nffi_%s'%(n)][1:].values
# 	s[s=='nan'] = np.nan
# 	s = s.astype(float)
# 	neo_plus_scores.append(s)
#
# for n in neo_neg:
# 	s = df['nffi_%s'%(n)][1:].values
# 	s[s=='nan'] = np.nan
# 	s = s.astype(float)
# 	neo_neg_scores.append(s)
