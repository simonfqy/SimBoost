from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import os
import time
import sys
import pwd
import pdb
import csv
import re

def generate_interaction_mat(dataset_nm, predict_cold = False, cold_drug=False, 
  cold_target=False, split_warm=False, filter_threshold=0):

  assert (predict_cold + cold_drug + cold_target + split_warm) <= 1
  data_dir = ""
  if re.search('davis', dataset_nm, re.I):
    data_dir = "./davis_data/"
  elif re.search('metz', dataset_nm, re.I):
    data_dir = "./metz_data/"
  elif re.search('kiba', dataset_nm, re.I):
    data_dir = "./KIBA_data/"

  suffix = ""
  if filter_threshold > 0:
    suffix = "_filtered" + suffix
  if predict_cold:
    suffix = "_cold" + suffix
  elif split_warm:
    suffix = "_warm" + suffix
  elif cold_drug:
    suffix = "_cold_drug" + suffix
  elif cold_target:
    suffix = "_cold_target" + suffix

  triplet_name = "triplet_split" + suffix + ".csv"
  quadruplets = np.loadtxt(data_dir + triplet_name, delimiter=",", skiprows=1)
  pair_to_value = {}
  drugs = []
  targets = []
  drug_set = set(drugs)
  target_set = set(targets)
  for i in range(len(quadruplets)):
    drug_code = int(quadruplets[i, 0])
    if drug_code not in drug_set:
      drugs.append(drug_code)
      drug_set = set(drugs)
    target_code = int(quadruplets[i, 1])
    if target_code not in target_set:
      targets.append(target_code)
      target_set = set(targets)
    pair = (drug_code, target_code)
    if pair not in pair_to_value:
      pair_to_value[pair] = quadruplets[i, 2]

  interaction_matrix = np.empty([len(drug_set), len(target_set)])
  interaction_matrix[:] = np.nan
  for d in drugs:
    for t in targets:
      pair = (d, t)
      if pair not in pair_to_value:
        continue
      # Because d and t start from 1, we need to deduct 1.
      interaction_matrix[d-1, t-1] = pair_to_value[pair]

  int_mat_name = "interaction_mat" + suffix + ".csv"
  np.savetxt(data_dir + int_mat_name, interaction_matrix, delimiter=',')

if __name__ == '__main__':
  generate_interaction_mat('davis', split_warm=True, filter_threshold=1)
  generate_interaction_mat('metz', split_warm=True, filter_threshold=1)
  generate_interaction_mat('kiba', split_warm=True, filter_threshold=6)