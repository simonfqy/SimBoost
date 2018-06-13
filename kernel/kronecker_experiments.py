import sys
import random
random.seed(1)
import subprocess

import numpy as np
import pdb

from kron_rls import KronRLS
from cindex_measure import cindex


def neg_rmse(y_true, y_pred):
  """Computes RMS error."""
  return -np.sqrt(np.mean((y_true - y_pred)**2))

def r_square(y_true, y_pred):
  return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)

def get_folds(fold_split_file_nm, foldcount, tsize):
  splitted_data = np.loadtxt(fold_split_file_nm, delimiter=",", skiprows=1)
  assert foldcount == max(splitted_data[:, 3])
  if tsize != len(splitted_data):
    pdb.set_trace()
  folds = [[] for j in range(foldcount)]
  for i in range(len(splitted_data)):
    foldind = int(splitted_data[i, 3]) - 1
    folds[foldind].append(i)
  
  #assert stuff
  foldunion = set([])
  for find in range(len(folds)):
    fold = set(folds[find])
    assert len(fold & foldunion) == 0, str(find)
    foldunion = foldunion | fold
  assert len(foldunion & set(range(tsize))) == tsize    
  return folds

def get_warm_folds(tsize, foldcount, data_dir="../data/davis_data", additional_suffix = "",
  featurizer_suffix = ""):
  additional_suffix = featurizer_suffix + "_warm_filtered" + additional_suffix
  fold_split_file = data_dir + "triplet_split" + additional_suffix + ".csv"
  return get_folds(fold_split_file, foldcount, tsize)  

def get_drugwise_folds(label_row_inds, foldcount, data_dir="../data/davis_data", 
  additional_suffix = "", featurizer_suffix = ""):
  assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
  additional_suffix = featurizer_suffix + "_cold_drug_filtered" + additional_suffix
  fold_split_file = data_dir + "triplet_split" + additional_suffix + ".csv"
  return get_folds(fold_split_file, foldcount, len(label_row_inds))  


def get_targetwise_folds(label_col_inds, foldcount, data_dir="../data/davis_data", 
  additional_suffix = "", featurizer_suffix = ""):
  assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
  additional_suffix = featurizer_suffix + "_cold_target_filtered" + additional_suffix
  fold_split_file = data_dir + "triplet_split" + additional_suffix + ".csv"
  return get_folds(fold_split_file, foldcount, len(label_col_inds))  


def nested_nfold_1_2_3_setting_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, folds):
  foldcount = len(folds)
  allindices = range(len(label_row_inds))
  outer_train_sets = []
  bestparaminds = []
  for test_foldind in range(foldcount):
    test_fold = folds[test_foldind]
    otherfoldinds = range(foldcount)
    otherfoldinds.pop(test_foldind)
    val_sets = []
    train_sets = []
    otherdatainds = set(allindices) - set(test_fold)
    for val_foldind in otherfoldinds:
      val_fold = folds[val_foldind]
      val_sets.append(val_fold)
      train_sets.append(sorted(list(otherdatainds - set(val_fold))))
    bestparamind, _, _ = general_nfold_cv_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, train_sets, val_sets)
    bestparaminds.append(bestparamind)
    outer_train_sets.append(sorted(list(otherdatainds)))
    print
    print 'Outer fold', test_foldind, 'done'
    print
  _, _, all_predictions = general_nfold_cv_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, outer_train_sets, folds)
  finalpreds = np.zeros(Y[label_row_inds, label_col_inds].shape)
  avgperf = 0.
  for test_foldind in range(len(folds)):
    bestparamind = bestparaminds[test_foldind]
    test_fold = folds[test_foldind]
    finalpreds[test_fold] = all_predictions[bestparamind][test_fold]
    label_arr = Y[label_row_inds, label_col_inds].ravel().T[test_fold]
    pred_arr = finalpreds.ravel().T[test_fold]
    foldperf = measure(label_arr, pred_arr)
    avgperf += foldperf
  avgperf = avgperf / foldcount
  #perf = measure(Y[label_row_inds, label_col_inds].ravel().T, finalpreds.ravel().T)
  #poolaupr = get_aupr(Y[label_row_inds, label_col_inds].ravel().T, finalpreds.ravel().T)
  print bestparaminds
  #print 'pooled ci', perf
  print 'averaged performance', avgperf
  #print 'pooled aupr', poolaupr
  return avgperf
    
# def nested_nfold_setting_4_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure):

#   dfcount, tfcount = 3, 3
#   print 'nested_nfold_setting_4_with_imputation', dfcount, 'and', tfcount, 'folds'
#   drugfolds = get_drugwise_folds(label_row_inds, label_col_inds, XD.shape[0], dfcount)
#   targetfolds = get_targetwise_folds(label_row_inds, label_col_inds, XT.shape[0], tfcount)
#   allindices = range(len(label_row_inds))
#   outer_train_sets = []
#   outer_test_sets = []
#   bestparams = []

#   for test_dfoldind in range(dfcount):
#     test_dfold = drugfolds[test_dfoldind]
#     other_drugfoldinds = range(dfcount)
#     other_drugfoldinds.pop(test_dfoldind)
#     for test_tfoldind in range(tfcount):
#       test_tfold = targetfolds[test_tfoldind]
#       other_targetfoldinds = range(tfcount)
#       other_targetfoldinds.pop(test_tfoldind)
#       val_sets = []
#       labeled_sets = []
#       testset = sorted(list(set(test_dfold) & set(test_tfold)))
#       outer_test_sets.append(testset)
#       for val_dfoldind in other_drugfoldinds:
#         val_dfold = drugfolds[val_dfoldind]
#         for val_tfoldind in other_targetfoldinds:
#           val_tfold = targetfolds[val_tfoldind]
#           fold = sorted(list(set(val_dfold) & set(val_tfold)))
#           val_sets.append(fold)
#           trainset = sorted(list(set(allindices) - set(test_dfold) - set(test_tfold) - set(val_dfold) - set(val_tfold)))
#           labeled_sets.append(trainset)
#       otherdatainds = sorted(list(set(allindices) - set(test_dfold) - set(test_tfold)))
#       bestparam, _, _ = general_nfold_cv_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, labeled_sets, val_sets)
#       bestparams.append(bestparam)
#       outer_train_sets.append(sorted(list(otherdatainds)))

#   _, _, all_predictions = general_nfold_cv_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, outer_train_sets, outer_test_sets)
#   finalpreds = np.mat(np.zeros(Y[label_row_inds, label_col_inds].shape))
#   avgperf = 0.
#   for test_foldind in range(len(outer_test_sets)):
#     bestparam = bestparams[test_foldind]
#     test_fold = outer_test_sets[test_foldind]
#     finalpreds[:, test_fold] = all_predictions[bestparam][test_fold]
#     foldperf = measure(Y[label_row_inds, label_col_inds].ravel().T[test_fold], np.squeeze(np.asarray(
#         finalpreds.ravel().T[test_fold])))
#     avgperf += foldperf
#   #pdb.set_trace()
#   perf = measure(Y[label_row_inds, label_col_inds].ravel().T, np.squeeze(np.asarray(finalpreds.ravel().T)))
#   print bestparams
#   #print 'pooled perf', perf
#   #print 'averaged perf', avgperf / len(outer_test_sets)
#   #return finalpreds, outer_test_sets
#   return avgperf / len(outer_test_sets)


def general_nfold_cv_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, labeled_sets, val_sets):
  
  logrpstart = -10
  logrpend = 41
  
  print 'param grid: ', logrpstart, logrpend
  all_predictions = []
  logrprange = range(logrpstart, logrpend)
  for logrpind in range(len(logrprange)):
    all_predictions.append(np.zeros(np.array(Y)[label_row_inds, label_col_inds].shape))
  for foldind in range(len(val_sets)):
    valinds = val_sets[foldind]
    labeledinds = labeled_sets[foldind]
    impvalue = np.mean(Y[label_row_inds[valinds], label_col_inds[valinds]])
    not_labeledinds = sorted(list(set(range(len(label_row_inds))) - set(labeledinds)))
    Y_train = np.mat(np.copy(Y))
    Y_train[label_row_inds[not_labeledinds], label_col_inds[not_labeledinds]] = impvalue
    nan_row_inds, nan_col_inds = np.where(np.isnan(Y_train)==True)
    Y_train[nan_row_inds, nan_col_inds] = impvalue 
    params = {}
    XD_train = XD
    XT_train = XT
    rows = label_row_inds[labeledinds]
    cols = label_col_inds[labeledinds]
    rows = np.array(list(set(rows)))
    cols = np.array(list(set(cols)))
    XD_train = XD[rows]
    XT_train = XT[cols]
    Y_train = Y_train[np.ix_(rows, cols)]
    #pdb.set_trace()
    params["xmatrix1"] = XD_train
    params["xmatrix2"] = XT_train
    params["train_labels"] = Y_train
    learner = KronRLS.createLearner(**params)
    for logrpind in range(len(logrprange)):
      logrp = logrprange[logrpind]
      regparam = 2. ** logrp
      learner.solve_linear(regparam)
      hopred = learner.getModel().predictWithDataMatrices(XD, XT)
      all_predictions[logrpind][valinds] = hopred[label_row_inds[valinds], label_col_inds[valinds]]
    #print foldind, 'done'
  #print
  bestperf = -float('Inf')
  bestparamind = None
  
  for logrpind in range(len(logrprange)):
    avgperf = 0.
    for foldind in range(len(val_sets)):
      valinds = val_sets[foldind]
      labels = Y[label_row_inds[valinds], label_col_inds[valinds]].ravel().T
      preds = all_predictions[logrpind].ravel().T[valinds]        
      foldperf = measure(labels, preds)
      avgperf += foldperf
    avgperf /= len(val_sets)
    logrp = logrprange[logrpind]
    print logrp, avgperf
    if avgperf > bestperf:
      bestperf = avgperf
      bestparamind = logrpind
  return bestparamind, bestperf, all_predictions


def get_aupr(Y, P):
  if hasattr(Y, 'A'): Y = Y.A
  if hasattr(P, 'A'): P = P.A
  Y = np.where(Y>0, 1, 0)
  Y = Y.ravel()
  P = P.ravel()
  f = open("temp.txt", 'w')
  for i in range(Y.shape[0]):
    f.write("%f %d\n" %(P[i], Y[i]))
  f.close()
  f = open("foo.txt", 'w')
  subprocess.call(["java", "-jar", "auc.jar", "temp.txt", "list"], stdout=f)
  f.close()
  f = open("foo.txt")
  lines = f.readlines()
  aucpr = float(lines[-2].split()[-1])
  f.close()
  return aucpr

  
def experiment(XD, XT, Y, perfmeasure, foldcount=5, data_dir="../data/davis_data", split_warm=False, 
  cold_drug=False, cold_target=False, additional_suffix = "", featurizer_suffix = ""):
  #Runs nested cross-validation in settings 1-4, automatically selecting regularization parameter
  #and computing the prediction performance using supplied performance measure.
  #Input
  #XD: [drugs, features] sized array (features may also be similarities with other drugs
  #XT: [targets, features] sized array (features may also be similarities with other targets
  #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
  #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
  #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
  #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation
  #
  #The regularization parameter is selected from the range 2**-10 ... 2**40
  
  label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
  #pdb.set_trace()
  #setting 1 folds
  Invalid_val = -10000
  S1_avgperf = Invalid_val
  S2_avgperf = Invalid_val
  S3_avgperf = Invalid_val
  if split_warm:
    S1_folds = get_warm_folds(len(label_row_inds), foldcount, data_dir=data_dir, 
      additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)
    print "setting 1"
    S1_avgperf = nested_nfold_1_2_3_setting_with_imputation(XD, XT, Y, label_row_inds, 
      label_col_inds, perfmeasure, S1_folds)

  if cold_drug:
    drugcount = XD.shape[0]
    #setting 2 folds
    S2_folds = get_drugwise_folds(label_row_inds, foldcount, data_dir=data_dir, 
      additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)
    print "setting 2"
    S2_avgperf = nested_nfold_1_2_3_setting_with_imputation(XD, XT, Y, label_row_inds, 
      label_col_inds, perfmeasure, S2_folds)

  if cold_target:
    targetcount = XT.shape[0]
    #setting 3 folds
    S3_folds = get_targetwise_folds(label_col_inds, foldcount, data_dir=data_dir, 
      additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)        
    print "setting 3"
    S3_avgperf = nested_nfold_1_2_3_setting_with_imputation(XD, XT, Y, label_row_inds, 
      label_col_inds, perfmeasure, S3_folds)
  
  #print "setting 4"
  #In setting 4 a 3x3 fold split is automatically generated inside the called function
  #S4_avgperf = nested_nfold_setting_4_with_imputation(XD, XT, Y, label_row_inds, label_col_inds, perfmeasure)
  
  print "Setting 1 Nested CV performance", S1_avgperf
  print "Setting 2 Nested CV performance", S2_avgperf
  print "Setting 3 Nested CV performance", S3_avgperf
  #print "Setting 4 Nested CV performance", S4_avgperf

#Some examples on how the program can be run   

def load_dataset(data_dir):
  fname = data_dir + "drug_sim_mat.csv"
  XD = np.loadtxt(fname, delimiter=',', skiprows=1)
  fname = data_dir + "target_sim_mat.csv"
  XT = np.loadtxt(fname, delimiter=',', skiprows=1)
  fname = data_dir + "interaction_mat.csv"
  Y = np.loadtxt(fname, delimiter=',', skiprows=0)

  return XD, XT, Y
  
# def load_metz(data_dir):
#   # Y = np.loadtxt("known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt")
#   # XD = np.loadtxt("drug-drug_similarities_2D__Metz_et_al.2011.txt")
#   # XT = np.loadtxt("target-target_similarities_WS_normalized__Metz_et_al.2011.txt")
#   fname = data_dir + "drug_sim_mat.csv"
#   XD = np.loadtxt(fname, delimiter=',', skiprows=1)
#   fname = data_dir + "target_sim_mat.csv"
#   XT = np.loadtxt(fname, delimiter=',', skiprows=1)
#   fname = data_dir + "interaction_mat.csv"
#   Y = np.loadtxt(fname, delimiter=',', skiprows=0)
#   return XD, XT, Y 

def davis_regression(additional_suffix="", featurizer_suffix="", perfmeasure=cindex):
  #Run the experiment on davis data with real-valued outputs
  #and concordance index as performance measure
  #concordance index is a value between 0 and 1, 0.5 random baseline
  #generalization of area under ROC curve (AUC) to regression, returns
  #the same value for binary Y-values (e.g. +1,-1)
  #perfmeasure = cindex
  #perfmeasure = neg_rmse
  #perfmeasure = r_square
  data_dir = "../data/davis_data/"
  XD, XT, Y = load_dataset(data_dir)
  experiment(XD, XT, Y, perfmeasure, data_dir=data_dir, split_warm=True, cold_target=True, 
    cold_drug=True, additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)
  
def davis_classification_cindex():
  #Run the experiment on davis data with binarized outputs
  #and concordance index as performance measure
  perfmeasure = cindex    
  XD, XT, Y = load_davis()
  #Same as before, but now we convert this to a binary classification task
  #Now cindex is same as AUC
  thval = 30
  Y_binary = np.where(Y<=thval, 1., -1.)
  experiment(XD, XT, Y_binary, perfmeasure)
  
def davis_classification_aupr():
  #Run the experiment on davis data with binarized outputs
  #and AUPR as performance measure
  perfmeasure = get_aupr
  #Classification with area under precision recall curve (AUPR)
  XD, XT, Y = load_davis()
  thval = 30
  Y_binary = np.where(Y<=thval, 1., -1.)
  experiment(XD, XT, Y_binary, perfmeasure)
  
def metz_regression(additional_suffix="", featurizer_suffix="", perfmeasure=cindex):
  #Run the experiment on metz data with real-valued outputs
  #and concordance index as performance measure
  #perfmeasure = cindex
  #perfmeasure = neg_rmse
  #perfmeasure = r_square
  data_dir = "../data/metz_data/"
  #Metz data is more complicated, since it has some Y-values missing
  #regression with real values
  XD, XT, Y = load_dataset(data_dir)
  experiment(XD, XT, Y, perfmeasure, data_dir=data_dir, split_warm=True, cold_target=True, 
    cold_drug=True, additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)
  
def metz_classification_cindex():
  #Run the experiment on metz data with binarized outputs
  #and concordance index as performance measure
  perfmeasure = cindex
  #Metz data is more complicated, since it has some Y-values missing
  #classification with AUC
  XD, XT, Y = load_metz()
  thval = 7.6
  #find indices for values that are not "nan" in the data
  label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
  #binarize values other than nan
  Y[label_row_inds, label_col_inds] = np.where(Y[label_row_inds, label_col_inds]>=thval, 1., -1.)
  experiment(XD, XT, Y, perfmeasure)
  
  
def metz_classification_aupr():
  #Run the experiment on metz data with binarized outputs
  #and AUPR index as performance measure
  perfmeasure = get_aupr
  #Metz data is more complicated, since it has some Y-values missing
  #classification with AUC
  XD, XT, Y = load_metz()
  thval = 7.6
  #find indices for values that are not "nan" in the data
  label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
  #binarize values other than nan
  Y[label_row_inds, label_col_inds] = np.where(Y[label_row_inds, label_col_inds]>=thval, 1., -1.)
  experiment(XD, XT, Y, perfmeasure)

def kiba_regression(additional_suffix="", featurizer_suffix="", perfmeasure=cindex):
  #perfmeasure = cindex
  #perfmeasure = neg_rmse
  #perfmeasure = r_square
  data_dir = "../data/KIBA_data/"
  XD, XT, Y = load_dataset(data_dir)
  experiment(XD, XT, Y, perfmeasure, data_dir=data_dir, split_warm=True, cold_target=True, 
    cold_drug=True, additional_suffix=additional_suffix, featurizer_suffix=featurizer_suffix)

if __name__=="__main__":
  featurizer_suffix="_gc"
  #featurizer_suffix = ""
  additional_suffix = "_2"
  perfmeasure = r_square
  # davis_regression(featurizer_suffix=featurizer_suffix, additional_suffix=additional_suffix,
  #   perfmeasure=perfmeasure)
  #davis_classification_cindex()
  #davis_classification_aupr()
  metz_regression(featurizer_suffix=featurizer_suffix, additional_suffix=additional_suffix,
    perfmeasure=perfmeasure)
  #metz_classification_cindex()
  #metz_classification_aupr()
  #kiba_regression(featurizer_suffix=featurizer_suffix, additional_suffix=additional_suffix,
  #  perfmeasure=perfmeasure)

