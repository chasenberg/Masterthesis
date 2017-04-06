import ROOT
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
from ROOT import gROOT, TCanvas, TF1, TFile, TTree, gRandom, TH1F

from ROOT import RooRealVar, RooFormulaVar, RooVoigtian, RooChebychev, RooArgList, \
                 RooArgSet, RooAddPdf, RooDataSet, RooCategory, RooSimultaneous, \
                 RooBreitWigner, RooCBShape, RooFFTConvPdf, RooGaussian,RooExponential, \
                 RooBinning, kRed, kBlue, kDotted,TString,RooAbsData, RooPlot, TCut, RooAbsData, RooFit

import os, sys, time, random

from ROOT import TTree, TFile

# from root_numpy import root2array, rec2array, array2root

import pandas as pd
import numpy as np
import scipy
import root_pandas as rp


sys.path.append('/home/chasenberg/repos/root_utils/')
sys.path.append('/home/chasenberg/repos/root_numpy/')
import root_numpy as ry
from ROOT import TColor
#import rootnotes  # For displaying TCanvas in notebooks
from lhcb_style import set_lhcb_style # Set basic LHCb plot style
set_lhcb_style()
from root_utils import plot_pulls

#directories and files
data_dir = '/fhgfs/users/chasenberg/data/2015/jpsiks/flattened/'
data_file ='Bd2JpsiKS_data_2015_flattened.root'
data_dir = os.path.join(data_dir, data_file)

tree_name = 'test'
data = ROOT.TFile('/fhgfs/users/chasenberg/data_trigger_incomplete/2015/jpsiks/test.root',"READ")
tree_data = data.Get(tree_name)

print("The tuple has",tree_data.GetEntries(),"entries.")

B0_M = RooRealVar("B0_FitDaughtersConst_M", "B0_M", 5280, "MeV")
upper_limit_mass = 5350
lower_limit_mass = 5220
mass = RooRealVar("B0_FitDaughtersConst_M", "Mass(J/psi K_{S})", 5280,lower_limit_mass, upper_limit_mass, "MeV")
# Construct signal from two Gaussian pdf's with different means and sigmas
mean = RooRealVar("mean", "mean", 5280,  5210,5360)
sigma_1 = RooRealVar("sigma_1", "sigma_1", 10, 0, 20)
sigma_2 = RooRealVar("sigma_2", "sigma_2", 15, 0, 20)
sig1frac = RooRealVar("sig1frac","fraction of component 1 in signal",0.8,0.,1.)

signal_1 = RooGaussian("signal_1", "signal_1", mass, mean, sigma_1)
signal_2 = RooGaussian("signal_2", "signal_2", mass, mean, sigma_2)
signal = RooAddPdf("DoubleGaussian","DoubleGaussian",signal_1, signal_2,sig1frac)
# Construct background pdf
lambda_1= RooRealVar("lambda","lambda",0.0,-0.1,0.0)
background = RooExponential("background","background",mass,lambda_1)
# Construct composite pdf
nsig = RooRealVar("nsig", "nsig", 35000,0,45000)
nbkg = RooRealVar("nbkg", "nbkg", 320000, 0, 400000)
model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

#Create dataset
ntupleVarSet =  RooArgSet(mass)
dataset = RooDataSet('data','data',tree_data,ntupleVarSet)

from ROOT import RooStats
print("Calculating sWeights")
data_sweight = ROOT.RooStats.SPlot("sData","An SPlot", dataset,model, RooArgList(nsig, nbkg))
sWeight_sig =ROOT.RooDataSet("dataset_nsig","dataset_nsig", dataset, dataset.get(),"","nsig_sw")
sWeight_bkg = ROOT.RooDataSet("dataset_nbkg","dataset_nbkg", dataset, dataset.get(),"","nbkg_sw")

#Fill SWeights to numpy array
Weight_sig_array = []
Weight_bkg_array = []
num = sWeight_sig.numEntries()
for i in range(num):
    Weight_sig_array.append(data_sweight.GetSWeight(i,"nsig"))
    Weight_bkg_array.append(data_sweight.GetSWeight(i,"nbkg"))


from ROOT import TTreeFormula
# writing interim file and tree to have same number of events in ttree and dataset (restricted mass range)
entries = tree_data.GetEntries()

print('Creatin output file')
interim_file = TFile("/tmp/interim.root","recreate")
interim_tree = tree_data
cut_string = ""
formula = TTreeFormula("formula",cut_string,interim_tree)
interim_tree = tree_data.CopyTree(cut_string)
interim_tree.Write()
data.Close()

# now writing final File
new_file = TFile("/fhgfs/users/chasenberg/data/2015/jpsiks/sweights/Bd2JpsiKS_data_2015_flattened_sw.root","recreate")
new_tree = interim_tree.CloneTree()
interim_file.Close()

interim_entries = new_tree.GetEntries()

sig_weight = np.zeros(1, dtype=float)
bkg_weight = np.zeros(1, dtype=float)
sigweight_branch = new_tree.Branch('SigYield_sw',sig_weight,'SigYield_sw/D')
bkgweight_branch = new_tree.Branch('BkgYield_sw',bkg_weight,'BkgYield_sw/D')

for i in range(0,interim_entries):
  sig_weight[0] = Weight_sig_array[i]
  bkg_weight[0] = Weight_bkg_array[i]
  sigweight_branch.Fill()
  bkgweight_branch.Fill()

new_tree.Write()
new_file.Close()

# removing interim file
os.remove("/tmp/interim.root")
