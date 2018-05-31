davis_dir = "./davis_data/"

davis_quadruplet = read.csv(paste(davis_dir, "triplet_split.csv", sep=""))
compound_info = read.csv(paste(davis_dir, "drug_info.csv", sep=""))
target_info = read.csv(paste(davis_dir, "prot_info.csv", sep=""))

## number of drugs and targets and density now:
dim(compound_info)[1]
dim(target_info)[1]
dim(davis_quadruplet)[1]/(dim(compound_info)[1]*dim(target_info)[1])

compound_sim = read.csv(paste(davis_dir,"tanimoto_cluster.csv", sep=""))
compound_ids_sim = compound_sim[,1]
compound_sim = compound_sim[,c(-1,-ncol(compound_sim))]

temp = match(compound_info[,1], compound_ids_sim)
## this is our drug sim mat. It is basically a rearrangement of the original matrix, namely compound_sim.
# It is rearranged according to the vector cids. Say, drug_sim_mat[1,1] correponds to cids[1]. The cids
# are PubChem CIDs.
drug_sim_mat = as.matrix(compound_sim[temp, temp])

save(drug_sim_mat, file="davis_drug_sim.rda")

## now we need to get the target similarity mat
require("Biostrings")
require("seqinr")
require("Rcpp")
## now compte the normalized Smith Waterman similarity
self_aligned_score = rep(NA, dim(target_info)[1])

for (i in 1:dim(target_info)[1]){
  cat('aligning ',i,'/', dim(target_info)[1],' with itself..\n')
  seq = target_info[i, 2]
  align_score = pairwiseAlignment(seq, seq, scoreOnly=TRUE, gapExtension=0.5, type="local", 
  	substitutionMatrix = "BLOSUM50")
  self_aligned_score[i] = align_score
}
# now compute the alignments for all pairs of targets.. note that I choose alignment parameters 
# type="local" to perform SW alignment and gapExtension = 0.5, type="local", substitutionMatrix = "BLOSUM50" 
# to get the same results as the target-target similarities here http://staff.cs.utu.fi/~aatapa/data/DrugTarget/

all_combinations = combn(dim(target_info)[1],2)
sim_mat = matrix(NA, nrow = dim(target_info)[1], ncol = dim(target_info)[1])

for (i in 1:ncol(all_combinations)){
  target_a = all_combinations[1, i]
  target_b = all_combinations[2, i]
  seq_1 = target_info[target_a, 2]
  seq_2 = target_info[target_b, 2]
  align_score_ab = pairwiseAlignment(seq_1, seq_2, scoreOnly=TRUE, gapExtension = 0.5, 
  	type="local", substitutionMatrix = "BLOSUM50")
  similarity = align_score_ab/(sqrt(self_aligned_score[target_a]) * sqrt(self_aligned_score[target_b]))
  sim_mat[target_a, target_b] = similarity
  if (i%%300 == 0){
    cat('computing similarity for ',target_a,', ',target_b,'(',i,'/',ncol(all_combinations),')\n')
    cat('similarity: ',similarity,'\n')
  }
}

for (i in 1:nrow(sim_mat)){
  for (k in 1:ncol(sim_mat)){
    if (!is.na(sim_mat[i,k])){
      sim_mat[k,i] = sim_mat[i,k]
    }
    if (i==k){
      sim_mat[i,k] = 1
    }
  }
}

target_sim_mat = sim_mat
save(target_sim_mat, file='davis_target_sim.rda')

## save everything in one file
davis_target_sim = target_sim_mat
davis_drug_sim = drug_sim_mat
save(davis_quadruplet, davis_drug_sim, davis_target_sim, file='../data/davis_data.Rda')

############################################################
# Now starts metz dataset.
metz_dir = "./metz_data/"

metz_quadruplet = read.csv(paste(metz_dir, "triplet_split.csv", sep=""))
compound_info = read.csv(paste(metz_dir, "drug_info.csv", sep=""))
target_info = read.csv(paste(metz_dir, "prot_info.csv", sep=""))

## number of drugs and targets and density now:
dim(compound_info)[1]
dim(target_info)[1]
dim(metz_quadruplet)[1]/(dim(compound_info)[1]*dim(target_info)[1])

compound_sim = read.csv(paste(metz_dir,"tanimoto_cluster.csv", sep=""))
compound_ids_sim = compound_sim[,1]
compound_sim = compound_sim[,c(-1,-ncol(compound_sim))]

temp = match(compound_info[,1], compound_ids_sim)
## this is our drug sim mat. It is basically a rearrangement of the original matrix, namely compound_sim.
# It is rearranged according to the vector cids. Say, drug_sim_mat[1,1] correponds to cids[1]. The cids
# are PubChem CIDs.
drug_sim_mat = as.matrix(compound_sim[temp, temp])

save(drug_sim_mat, file="metz_drug_sim.rda")

## now we need to get the target similarity mat
## now compte the normalized Smith Waterman similarity
self_aligned_score = rep(NA, dim(target_info)[1])

for (i in 1:dim(target_info)[1]){
  cat('aligning ',i,'/', dim(target_info)[1],' with itself..\n')
  seq = target_info[i, 2]
  align_score = pairwiseAlignment(seq, seq, scoreOnly=TRUE, gapExtension=0.5, type="local", 
    substitutionMatrix = "BLOSUM50")
  self_aligned_score[i] = align_score
}
# now compute the alignments for all pairs of targets.. note that I choose alignment parameters 
# type="local" to perform SW alignment and gapExtension = 0.5, type="local", substitutionMatrix = "BLOSUM50" 
# to get the same results as the target-target similarities here http://staff.cs.utu.fi/~aatapa/data/DrugTarget/

all_combinations = combn(dim(target_info)[1],2)
sim_mat = matrix(NA, nrow = dim(target_info)[1], ncol = dim(target_info)[1])

for (i in 1:ncol(all_combinations)){
  target_a = all_combinations[1, i]
  target_b = all_combinations[2, i]
  seq_1 = target_info[target_a, 2]
  seq_2 = target_info[target_b, 2]
  align_score_ab = pairwiseAlignment(seq_1, seq_2, scoreOnly=TRUE, gapExtension = 0.5, 
    type="local", substitutionMatrix = "BLOSUM50")
  similarity = align_score_ab/(sqrt(self_aligned_score[target_a]) * sqrt(self_aligned_score[target_b]))
  sim_mat[target_a, target_b] = similarity
  if (i%%300 == 0){
    cat('computing similarity for ',target_a,', ',target_b,'(',i,'/',ncol(all_combinations),')\n')
    cat('similarity: ',similarity,'\n')
  }
}

for (i in 1:nrow(sim_mat)){
  for (k in 1:ncol(sim_mat)){
    if (!is.na(sim_mat[i,k])){
      sim_mat[k,i] = sim_mat[i,k]
    }
    if (i==k){
      sim_mat[i,k] = 1
    }
  }
}

target_sim_mat = sim_mat
save(target_sim_mat, file='metz_target_sim.rda')

## save everything in one file
metz_target_sim = target_sim_mat
metz_drug_sim = drug_sim_mat
save(metz_quadruplet, metz_drug_sim, metz_target_sim, file='../data/metz_data.Rda')
