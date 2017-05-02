import sys
sys.path.append('/home/chasenberg/repos/')
from dopy.dolearn.sklearn_utils import plot_roc_curve, plot_classifier_output, plot_correlations


import os, sys, time, random

from ROOT import TTree, TFile

# from root_numpy import root2array, rec2array, array2root

import pandas as pd
import numpy as np
import scipy
import root_pandas as rp
import root_numpy as ry

import pandas.core.common as com
from pandas.core.index import Index
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix

from tqdm import tqdm_notebook

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score

sys.path.append('/home/chasenberg/repos/')
sys.path.append('/home/chasenberg/repos/dopy')
from dopy import *
from dopy.dolearn.sklearn_utils import plot_roc_curve, plot_classifier_output, plot_correlations
from dopy.dolearn.sklearn_utils import plot_feature_importances, plot_classifier_output, classify_unseen_data
#from dopy.sklearn_utils import plot_bdt_vars
from dopy.doplot.plotting import Plotter, Plot
from dopy.doanalysis.df_utils import add_min_max, add_eta


#Declare triggerlines
trigger_lines = '(B0_L0MuonDecision_Dec==1)|(B0_Hlt1TrackMuonDecision_Dec==1)|(B0_Hlt2DiMuonJPsiDecision_Dec==1)'
#Path to ROOT files: directories and files
data_dir_2015 = '/fhgfs/users/chasenberg/data_trigger_incomplete/2015/jpsiks/flattened/'
data_file_2015 ='Bd2JpsiKS_data_2015_flattened.root'
data_dir_2016 = '/fhgfs/users/chasenberg/data_trigger_incomplete/2016/jpsiks/flattened/'
data_file_2016 ='Bd2JpsiKS_data_2016_flattened.root'

mc_dir = '/fhgfs/users/chasenberg/mc_trigger_incomplete/2016/jpsiks/dimuon/flattened/'
mc_file = 'Bd2JpsiKS_dimuon_mc_2016_flat.root'

data_dir_2015 = os.path.join(data_dir_2015, data_file_2015)
data_dir_2016 = os.path.join(data_dir_2016, data_file_2016)
mc_dir = os.path.join(mc_dir, mc_file)

#Treename and cut for data
tree_data = 'Bd2JpsiKs'
cut_string_data = 'B0_FitDaughtersConst_status==0&B0_FitPVConst_status==0&idxPV==0&B0_FitDaughtersConst_M<5450&B0_FitDaughtersConst_M>5220&(B0_L0MuonDecision_Dec==1|B0_Hlt1TrackMuonDecision_Dec==1|B0_Hlt2DiMuonJPsiDecision_Dec==1)'
#criteria and information for read in mc
cut_string_mc = 'B0_FitDaughtersConst_status==0&B0_FitPVConst_status==0&idxPV==0&B0_FitDaughtersConst_M<5450&B0_FitDaughtersConst_M>5220&B0_BKGCAT==0&(B0_L0MuonDecision_Dec==1|B0_Hlt1TrackMuonDecision_Dec==1|B0_Hlt2DiMuonJPsiDecision_Dec==1)'
tree_mc = 'Bd2JpsiKs'

#Listing all variables

variables = [
'B0_M',
'B0_TAGDECISION_OS',
'B0_TAGOMEGA_OS',
'B0_TAU',
'B0_TAUERR',
'B0_FitDaughtersConst_M',
'B0_FitDaughtersConst_chi2',
'B0_FitDaughtersConst_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_IP',
'B0_FitDaughtersConst_KS0_P1_PT',
'B0_FitDaughtersConst_KS0_P0_PT',
'B0_FitDaughtersConst_KS0_decayLength',
'B0_FitDaughtersConst_KS0_IP',
'B0_FitDaughtersConst_KS0_P0_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_IPCHI2',
'B0_L0MuonDecision_Dec',
'B0_Hlt1TrackMuonDecision_Dec',
'B0_Hlt2DiMuonJPsiDecision_Dec',
'B0_M',
'idxPV',
'piminus_TRACK_Type',
'B0_FitPVConst_status',
'B0_FitDaughtersConst_KS0_P1_PT',
'B0_FitDaughtersConst_KS0_P0_PT',
'B0_FitDaughtersConst_KS0_P0_IPCHI2',
'B0_FitDaughtersConst_KS0_P1_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_P0_PT',
'B0_FitDaughtersConst_J_psi_1S_P1_PT',
'B0_FitPVConst_KS0_tau',
'B0_FitPVConst_KS0_tauErr',
#'SigYield_sw',
#'BkgYield_sw'
]
variables_mc = [
'B0_M',
'B0_FitDaughtersConst_M',
'B0_FitDaughtersConst_chi2',
'B0_FitDaughtersConst_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_IP',
'B0_FitDaughtersConst_KS0_P1_PT',
'B0_FitDaughtersConst_KS0_P0_PT',
'B0_FitDaughtersConst_KS0_decayLength',
'B0_FitDaughtersConst_KS0_IP',
'B0_FitDaughtersConst_KS0_P0_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_IPCHI2',
'B0_L0MuonDecision_Dec',
'B0_Hlt1TrackMuonDecision_Dec',
'B0_Hlt2DiMuonJPsiDecision_Dec',
'B0_M',
'idxPV',
'piminus_TRACK_Type',
'B0_FitPVConst_status',
'B0_FitDaughtersConst_KS0_P1_PT',
'B0_FitDaughtersConst_KS0_P0_PT',
'B0_FitDaughtersConst_KS0_P0_IPCHI2',
'B0_FitDaughtersConst_KS0_P1_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_P0_PT',
'B0_FitDaughtersConst_J_psi_1S_P1_PT',
'B0_FitPVConst_KS0_tau',
'B0_FitPVConst_KS0_tauErr',
'B0_BKGCAT'
]

##################################################################################################

#read root files
#2015 sideband
real_dataframe_2015 = rp.read_root(data_dir_2015,key=tree_data, columns=variables,where=cut_string_data, flatten=True)
real_dataframe_2015 = real_dataframe_2015.replace([np.inf, -np.inf], np.nan)
real_dataframe_2015 = real_dataframe_2015.dropna()
#2016 sideband
real_dataframe_2016 = rp.read_root(data_dir_2016,key=tree_data ,columns=variables,where=cut_string_data, flatten=True)
real_dataframe_2016 = real_dataframe_2016.replace([np.inf, -np.inf], np.nan)
real_dataframe_2016 = real_dataframe_2016.dropna()

signal_dataframe = rp.read_root(mc_dir,key=tree_mc,columns=variables_mc,where=cut_string_mc, flatten=True)
signal_dataframe = signal_dataframe.replace([np.inf, -np.inf], np.nan)
signal_dataframe = signal_dataframe.dropna()
#Print dataframe size
print("Size of 2015 data",real_dataframe_2015.shape)
print("Size of 2016 data",real_dataframe_2016.shape)
#Merge 2015/2016 data
real_dataframe = pd.concat([real_dataframe_2015,real_dataframe_2016])

##################################################################################################

#add symmetrical variables to dataframe
from dopy.doplot.selection import add_log_to_dataframe, add_max_to_dataframe, add_min_to_dataframe
#choose minimal value of both myon transversal momentums
add_min_to_dataframe(real_dataframe, 'B0_FitDaughtersConst_KS0_min_PT', ['B0_FitDaughtersConst_KS0_P1_PT', 'B0_FitDaughtersConst_KS0_P0_PT'])
add_min_to_dataframe(signal_dataframe, 'B0_FitDaughtersConst_KS0_min_PT', ['B0_FitDaughtersConst_KS0_P1_PT', 'B0_FitDaughtersConst_KS0_P0_PT'])

#choose minimal value of both pions ipchi2
add_min_to_dataframe(real_dataframe, 'test_IP', ['B0_FitDaughtersConst_KS0_P0_IPCHI2', 'B0_FitDaughtersConst_KS0_P1_IPCHI2'])
add_min_to_dataframe(signal_dataframe, 'test_IP', ['B0_FitDaughtersConst_KS0_P0_IPCHI2', 'B0_FitDaughtersConst_KS0_P1_IPCHI2'])

#choose minimal value of both myons transversal momentum
add_min_to_dataframe(real_dataframe, 'B0_FitDaughtersConst_J_psi_1S_min_PT', ['B0_FitDaughtersConst_J_psi_1S_P0_PT', 'B0_FitDaughtersConst_J_psi_1S_P1_PT'])
add_min_to_dataframe(signal_dataframe, 'B0_FitDaughtersConst_J_psi_1S_min_PT', ['B0_FitDaughtersConst_J_psi_1S_P0_PT', 'B0_FitDaughtersConst_J_psi_1S_P1_PT'])

#Generate dimensionless decay-time distribution
real_dataframe['B0_FitPVConst_KS0_tau_dimless'] = real_dataframe['B0_FitPVConst_KS0_tau']/real_dataframe['B0_FitPVConst_KS0_tauErr']
signal_dataframe['B0_FitPVConst_KS0_tau_dimless'] = signal_dataframe['B0_FitPVConst_KS0_tau']/signal_dataframe['B0_FitPVConst_KS0_tauErr']

#del signal_dataframe['B0_BKGCAT']
del signal_dataframe['__array_index']

##################################################################################################
########################Import data sideband B0_M>5450############################################
##################################################################################################

cutstring = 'B0_FitDaughtersConst_status==0&B0_FitPVConst_status==0&idxPV==0&B0_FitDaughtersConst_M>5450&((B0_L0Global_TOS==1)|(B0_Hlt1DiMuonHighMassDecision_TOS==1)|B0_Hlt2DiMuonDetachedJPsiDecision_TOS==1)'
bkg_dataframe_2015 = rp.read_root('/fhgfs/users/chasenberg/data_trigger_incomplete/2015/jpsiks/flattened/Bd2JpsiKS_data_2015_flattened.root',columns=variables,key=tree_data,where=cutstring, flatten=True)
bkg_dataframe_2016 = rp.read_root('/fhgfs/users/chasenberg/data_trigger_incomplete/2016/jpsiks/flattened/Bd2JpsiKS_data_2016_flattened.root',columns=variables,key=tree_data,where=cutstring, flatten=True)

#Merge both years
bkg_dataframe = pd.concat([bkg_dataframe_2015,bkg_dataframe_2016])
bkg_dataframe = bkg_dataframe.replace([np.inf, -np.inf], np.nan)
bkg_dataframe = bkg_dataframe.dropna()

del bkg_dataframe['__array_index']

##################################################################################################
###############################Add symmetrical variables to dataframes############################
##################################################################################################

from dopy.doplot.selection import add_log_to_dataframe, add_max_to_dataframe, add_min_to_dataframe

###############################Signal MC and data ################################################

#choose minimal value of both myon transversal momentums
add_min_to_dataframe(real_dataframe, 'B0_FitDaughtersConst_KS0_min_PT', ['B0_FitDaughtersConst_KS0_P1_PT', 'B0_FitDaughtersConst_KS0_P0_PT'])
add_min_to_dataframe(signal_dataframe, 'B0_FitDaughtersConst_KS0_min_PT', ['B0_FitDaughtersConst_KS0_P1_PT', 'B0_FitDaughtersConst_KS0_P0_PT'])
#choose minimal value of both pions ipchi2
add_min_to_dataframe(real_dataframe, 'test_IP', ['B0_FitDaughtersConst_KS0_P0_IPCHI2', 'B0_FitDaughtersConst_KS0_P1_IPCHI2'])
add_min_to_dataframe(signal_dataframe, 'test_IP', ['B0_FitDaughtersConst_KS0_P0_IPCHI2', 'B0_FitDaughtersConst_KS0_P1_IPCHI2'])
#choose minimal value of both myons transversal momentum
add_min_to_dataframe(real_dataframe, 'B0_FitDaughtersConst_J_psi_1S_min_PT', ['B0_FitDaughtersConst_J_psi_1S_P0_PT', 'B0_FitDaughtersConst_J_psi_1S_P1_PT'])
add_min_to_dataframe(signal_dataframe, 'B0_FitDaughtersConst_J_psi_1S_min_PT', ['B0_FitDaughtersConst_J_psi_1S_P0_PT', 'B0_FitDaughtersConst_J_psi_1S_P1_PT'])
#Generate dimensionless decay-time distribution
real_dataframe['B0_FitPVConst_KS0_tau_dimless'] = real_dataframe['B0_FitPVConst_KS0_tau']/real_dataframe['B0_FitPVConst_KS0_tauErr']
signal_dataframe['B0_FitPVConst_KS0_tau_dimless'] = signal_dataframe['B0_FitPVConst_KS0_tau']/signal_dataframe['B0_FitPVConst_KS0_tauErr']

###############################Combinatorical background ##########################################

#choose minimal value of both myon transversal momentums
add_min_to_dataframe(bkg_dataframe, 'B0_FitDaughtersConst_KS0_min_PT', ['B0_FitDaughtersConst_KS0_P1_PT', 'B0_FitDaughtersConst_KS0_P0_PT'])
#choose minimal value of both pions ipchi2
add_min_to_dataframe(bkg_dataframe, 'test_IP', ['B0_FitDaughtersConst_KS0_P0_IPCHI2', 'B0_FitDaughtersConst_KS0_P1_IPCHI2'])
#choose minimal value of both myons transversal momentum
add_min_to_dataframe(bkg_dataframe, 'B0_FitDaughtersConst_J_psi_1S_min_PT', ['B0_FitDaughtersConst_J_psi_1S_P0_PT', 'B0_FitDaughtersConst_J_psi_1S_P1_PT'])
#Generate bkg_dataframedimensionless decay-time distribution
bkg_dataframe['B0_FitPVConst_KS0_tau_dimless'] = bkg_dataframe['B0_FitPVConst_KS0_tau']/bkg_dataframe['B0_FitPVConst_KS0_tauErr']

###############################Check signal and Combinatorical background #########################

dataframe_names = ['Signal', 'Background']
plot_vars = ['B0_FitDaughtersConst_M']
plot_names = ['B0_M_sideband']
dataframe_list = [signal_dataframe, bkg_dataframe]
plotter = Plotter()
plotter.create_plots(dataframe_list, plot_vars, None, plot_names , dataframe_names)
#plotter['B0_FitDaughtersConst_M'].set_range(5200, 6000)
p = Plotter('/home/chasenberg/plots/selection/')
plotter.plot()
print('Just checked upper sideband and signal mc')
###############################Check BDT features#########################

bdt_features = [
'B0_FitDaughtersConst_chi2',
'B0_FitDaughtersConst_IPCHI2',
#'B0_FitDaughtersConst_J_psi_1S_IP',
#'B0_FitDaughtersConst_KS0_P1_PT',
#'B0_FitDaughtersConst_KS0_P0_PT',
'B0_FitDaughtersConst_KS0_min_PT',
'B0_FitDaughtersConst_KS0_decayLength',
'B0_FitPVConst_KS0_tau_dimless',
#'test_IP',
'B0_FitDaughtersConst_KS0_IP',
'B0_FitDaughtersConst_KS0_P0_IPCHI2',
#'B0_FitDaughtersConst_J_psi_1S_IPCHI2',
'B0_FitDaughtersConst_J_psi_1S_min_PT'
]

plt.figure(figsize=(12,8))
plot_correlations(bkg_dataframe[bdt_features], annot=True, fmt='.2f')
plt.savefig('/home/chasenberg/plots/selection/correlation_data.png')
plt.show()
plt.figure(figsize=(12,8))
plot_correlations(signal_dataframe[bdt_features], annot=True, fmt='.2f')
plt.savefig('/home/chasenberg/plots/selection/correlation_mc.png')
plt.show()
print('Just printed correlation tables')


##################################################################################################
###############################Develop GradientBoostingClassifier################################
##################################################################################################
#Set flags on signal mc and data
flags = np.array([1]*len(signal_dataframe)+[0]*len(bkg_dataframe))
dataframe = pd.concat([signal_dataframe, bkg_dataframe])
#Train classifier
train_dataframe, test_dataframe, train_flags, test_flags = train_test_split(
                                                            dataframe[bdt_features], flags, test_size=0.5, random_state=42)
classifier = GradientBoostingClassifier(max_depth=3, verbose=1,n_estimators=200,learning_rate=0.1)
classifier.fit(train_dataframe, train_flags)
print("The performance of the classifier is:")
print(classifier.score(test_dataframe, test_flags))
#Feature importances
importances = classifier.feature_importances_
print("Features sorted by their score:" )
importances_sorted = sorted(zip(importances, dataframe.columns), reverse=True)
for val,name in importances_sorted:
    print('{}: {:.4f}'.format(name, val))

#Plot feature importances
plot_feature_importances(classifier,dataframe[bdt_features])
plt.savefig('/home/chasenberg/plots/selection/feature_importance.png', bbox_inches='tight')
print("Just saved feature_importance plot")
plt.figure(figsize=(12,10))
#Overtraining
plot_classifier_output(classifier, train_dataframe, train_flags, test_dataframe, test_flags, title='Ausgabe des BDT',bins=50)
plt.savefig('/home/chasenberg/repos/b2cc_sin2beta/notebooks/selection/plots/overtraining.png')
print("Just saved overtraining plot")
#ROC curve
plot_roc_curve(classifier, test_dataframe, test_flags)
plt.savefig('/home/chasenberg/repos/b2cc_sin2beta/notebooks/selection/plots/roc_curve.png')
print("Just saved ROC curve plot")

##################################################################################################
###############################Classify unseen data###############################################
##################################################################################################

unseen_df.query('B0_FitDaughtersConst_M<5230').shape
print("The size of the unseen data is:")
print(unseen_df.query('B0_FitDaughtersConst_M<5230').shape)
#apply classifier to data and MC
classify_unseen_data([classifier], real_dataframe, bdt_features, 'BDTresponse')
classify_unseen_data([classifier], signal_dataframe, bdt_features, 'BDTresponse')
classify_unseen_data([classifier], bkg_dataframe, bdt_features, 'BDTresponse')


##################################################################################################
###############################Write BDT output to ROOT-file######################################
##################################################################################################

#BDT output to arrays
bdt_output = real_dataframe['BDTresponse']

from ROOT import TTreeFormula
# writing interim file and tree to have same number of events in ttree and dataset (restricted mass range)

print('Creatin output file')
interim_file = TFile("/tmp/interim.root","recreate")
interim_tree = tree_data
cut_string = ""
formula = TTreeFormula("formula",cut_string,interim_tree)
interim_tree = tree_data.CopyTree(cut_string)
interim_tree.Write()
data.Close()

# now writing final File
new_file = TFile("/fhgfs/users/chasenberg/data_trigger_incomplete/2015/jpsiks/Bd2JpsiKS_data_2015_bdtoutput.root","recreate")
new_tree = interim_tree.CloneTree()
interim_file.Close()

interim_entries = new_tree.GetEntries()

bdt_response = np.zeros(1, dtype=float)
bdt_output_branch = new_tree.Branch('bdt_output',bdt_output,'bdt_output')

for i in range(0,interim_entries):
  bdt_response[0] = bdt_output[i]
  bdt_output_branch.Fill()

new_tree.Write()
new_file.Close()

# removing interim file
os.remove("/tmp/interim.root")
