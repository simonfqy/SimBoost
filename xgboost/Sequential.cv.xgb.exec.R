require(xgboost)
require(methods)
require(stringr)
source('utils.R')
source('Sequential.cv.xgb.R')

#Looptime
L = 1

parse.mid = function(res) {
  as.numeric(strsplit(gsub('\\s+','',res),',')[[1]][2])
}

rmse.fun = function(preds, label) {
  sqrt(mean((preds - label)^2))
}

# Davis

# Load original data
source('Sequential.cv.R')
load('../data/davis_data.Rda')
quadruplet = davis_quadruplet
quadruplet = quadruplet[order(quadruplet[,1]),]
d.sim = davis_drug_sim
t.sim = davis_target_sim
# res = clear.cold.start(quadruplet, d.sim, t.sim, 5)
# quadruplet = res[[1]]
# d.sim = res[[2]]
# t.sim = res[[3]]
nfold = 5
k = 4
latent.dim = 10
threshold = 0.3
threshold.identity = (0:5)/10

davis = list()
rmse = rep(0,L)
params = list(nthread = 8, objective = "reg:linear", eta = 0.05, 
              subsample = 0.6, colsample_bytree = 0.6, #sqrt(feature.p)/feature.p,
              num_parallel_tree = 1, max_depth = 6, min_child_weight = 10)

for (i in 1:L) {
  cat('Round ',i,'\n')
  set.seed(i)
  file_name = '../data/sequential.cv.feature.mf.davis.'
  file_name = paste0(file_name, i, '.rda')
  if (!file.exists(file_name)){
    #sink("temporary.txt", append=T, type="output")
    seq.feature = sequential.cv.feature(quadruplet, d.sim, t.sim, latent.dim,
                                        nfold, k, threshold, threshold.identity)
    #sink()
    cv.data = seq.feature[[1]]
    cv.folds = seq.feature[[2]]
    triplet = quadruplet[,1:3]
    save(triplet, d.sim, t.sim, cv.data, cv.folds, file=file_name)
  }
  out_file_name = '../data/sequential.cv.xgb.mf.davis.'
  out_file_name = paste0(out_file_name, i, '.rda')
  davis[[i]] = sequential.mean.cv(file_name, out_file_name,
                                  cutoff = 7, params = params, nrounds = 500, seed = i)
  res = davis[[i]]
  res = cbind(res[,1],res[,2])
  auc(res[,1],res[,2],7)
  write.table(res, file=paste0('davis_',i,'_cont.txt'), col.names = FALSE,quote = FALSE,
              row.names = FALSE)
  # rmse[i] = rmse.fun(res[,1], res[,2])
  # res[,2] = as.numeric(res[,2] > 7)
  # write.table(res, file=paste0('davis_',i,'_bin.txt'), col.names = FALSE,quote = FALSE,
  #             row.names = FALSE)
}

save(davis, file='../data/xgb.davis.rda')

sapply(1:L, function(i) auc(davis[[i]][,1],davis[[i]][,2], 7))

auc = rep(0,L)
aupr = rep(0,L)
ci = rep(0,L)
rmse = rep(0,L)
for (i in 1:L) {
  cat(i,'\n')
  res = read.table(paste0('davis_',i,'_cont.txt'))
  rmse[i] = rmse.fun(res[,1],res[,2])
  # res_auc = system(paste0('python ../evaluation/evaluate_AUC_complete.py davis_',i,'_bin.txt'), intern = TRUE)
  # auc[i] = parse.mid(res_auc)
  # res_aupr = system(paste0('python ../evaluation/evaluate_AUPR_complete.py davis_',i,'_bin.txt'), intern = TRUE)
  # aupr[i] = parse.mid(res_aupr)
  res_ci = system(paste0('python ../evaluation/evaluate_CI_complete.py davis_',i,'_cont.txt'), intern = TRUE)
  ci[i] = as.numeric(str_extract(res_ci, "\\-*\\d+\\.*\\d*"))
  #ci[i] = parse.mid(res_ci)
}

davis.mean = c(mean(rmse), mean(auc), mean(aupr), mean(ci))
davis.sd = c(sd(rmse), sd(auc), sd(aupr), sd(ci))
save(davis.mean, davis.sd, file='davis.table.rda')

# Metz
load('../data/metz_data.Rda')
quadruplet = metz_quadruplet
quadruplet = quadruplet[order(quadruplet[,1]),]
d.sim = metz_drug_sim
t.sim = metz_target_sim
# res = clear.cold.start(quadruplet, d.sim, t.sim, 5)
# quadruplet = res[[1]]
# d.sim = res[[2]]
# t.sim = res[[3]]
nfold = 5
k = 4
latent.dim = 10
threshold = 0.3
threshold.identity = (0:5)/10

metz = list()
rmse = rep(0,L)
params = list(nthread = 8, objective = "reg:linear", eta = 0.1, 
              subsample = 0.6, colsample_bytree = 0.6, #sqrt(feature.p)/feature.p,
              num_parallel_tree = 1, max_depth = 6, min_child_weight = 10)

for (i in 1:L) {
  cat('Round ',i,'\n')
  set.seed(i)
  file_name = '../data/sequential.cv.feature.mf.metz.'
  file_name = paste0(file_name, i, '.rda')
  if (!(file.exists(file_name))){
    seq.feature = sequential.cv.feature(quadruplet, d.sim, t.sim, latent.dim,
                                        nfold, k, threshold, threshold.identity)
    cv.data = seq.feature[[1]]
    cv.folds = seq.feature[[2]]
    triplet = quadruplet[,1:3]
    save(triplet, d.sim, t.sim, cv.data, cv.folds, file=file_name)
  }
  out_file_name = '../data/sequential.cv.xgb.mf.metz.'
  out_file_name = paste0(out_file_name, i, '.rda')
  metz[[i]] = sequential.mean.cv(file_name, out_file_name,cutoff = 7.6, 
                                 params = params, nrounds = 500, seed = i)
  res = metz[[i]]
  res = cbind(res[,1],res[,2])
  auc(res[,1],res[,2],7.6)
  write.table(res, file=paste0('metz_',i,'_cont.txt'), col.names = FALSE,quote = FALSE,
              row.names = FALSE)
  rmse[i] = rmse.fun(res[,1], res[,2])
  # res[,2] = as.numeric(res[,2] > 7.6)
  # write.table(res, file=paste0('metz_',i,'_bin.txt'), col.names = FALSE,quote = FALSE,
  #             row.names = FALSE)
}

save(metz, file='../data/xgb.metz.rda')

auc = rep(0,L)
aupr = rep(0,L)
rmse = rep(0,L)
ci = rep(0,L)
for (i in 1:L) {
  cat(i,'\n')
  res = read.table(paste0('metz_',i,'_cont.txt'))
  rmse[i] = rmse.fun(res[,1],res[,2])
  # res_auc = system(paste0('python ../evaluation/evaluate_AUC_complete.py metz_',i,'_bin.txt'), intern = TRUE)
  # auc[i] = parse.mid(res_auc)
  # res_aupr = system(paste0('python ../evaluation/evaluate_AUPR_complete.py metz_',i,'_bin.txt'), intern = TRUE)
  # aupr[i] = parse.mid(res_aupr)
  res_ci = system(paste0('python ../evaluation/evaluate_CI_complete.py metz_',i,'_cont.txt'), intern = TRUE)
  ci[i] = as.numeric(str_extract(res_ci, "\\-*\\d+\\.*\\d*"))
  #ci[i] = parse.mid(res_ci)
}

metz.mean = c(mean(rmse), mean(auc), mean(aupr), mean(ci))
metz.sd = c(sd(rmse), sd(auc), sd(aupr), sd(ci))
save(metz.mean, metz.sd, file='metz.table.rda')

# Kiba

load('../data/kiba_data.Rda')
quadruplet = kiba_quadruplet
quadruplet = quadruplet[order(quadruplet[,1]),]
d.sim = kiba_drug_sim
t.sim = kiba_target_sim
# res = clear.cold.start(quadruplet, d.sim, t.sim, 2)
# quadruplet = res[[1]]
# d.sim = res[[2]]
# t.sim = res[[3]]
nfold = 5
k = 4
latent.dim = 10
threshold = 0.3
threshold.identity = (0:5)/10

kiba = list()
rmse = rep(0,L)
params = list(nthread = 8, objective = "reg:linear", eta = 0.2, 
              subsample = 0.8, colsample_bytree = 0.8,
              num_parallel_tree = 1, max_depth = 6, min_child_weight = 10)

for (i in 1:L) {
  cat('Round ',i,'\n')
  set.seed(i)
  file_name = '../data/sequential.cv.feature.mf.kiba.'
  file_name = paste0(file_name, i, '.rda')
  if (!file.exists(file_name)){
    
    seq.feature = sequential.cv.feature(quadruplet, d.sim, t.sim, latent.dim,
                                        nfold, k, threshold, threshold.identity)
    
    cv.data = seq.feature[[1]]
    cv.folds = seq.feature[[2]]
    triplet = quadruplet[,1:3]
    save(triplet, d.sim, t.sim, cv.data, cv.folds, file=file_name)
  }
  out_file_name = '../data/sequential.cv.xgb.mf.kiba.'
  out_file_name = paste0(out_file_name, i, '.rda')
  kiba[[i]] = sequential.mean.cv(file_name, out_file_name, cutoff = 3.0, 
                                 params = params, nrounds = 1000, seed = i)
  res = kiba[[i]]
  res = cbind(res[,1],res[,2])
  auc(res[,1],res[,2],4.5)
  write.table(res, file=paste0('kiba_',i,'_cont.txt'), col.names = FALSE,quote = FALSE,
              row.names = FALSE)
  rmse[i] = rmse.fun(res[,1], res[,2])
  # res[,2] = as.numeric(res[,2] > 4.5)
  # write.table(res, file=paste0('kiba_',i,'_bin.txt'), col.names = FALSE,quote = FALSE,
  #             row.names = FALSE)
}

save(kiba, file='../data/xgb.kiba.rda')

auc = rep(0,L)
aupr = rep(0,L)
rmse = rep(0,L)
ci = rep(0,L)
for (i in 1:L) {
  cat(i,'\n')
  res = read.table(paste0('kiba_',i,'_cont.txt'))
  rmse[i] = rmse.fun(res[,1],res[,2])
  # res_auc = system(paste0('python ../evaluation/evaluate_AUC_complete.py kiba_',i,'_bin.txt'), intern = TRUE)
  # auc[i] = parse.mid(res_auc)
  # res_aupr = system(paste0('python ../evaluation/evaluate_AUPR_complete.py kiba_',i,'_bin.txt'), intern = TRUE)
  # aupr[i] = parse.mid(res_aupr)
  res_ci = system(paste0('python ../evaluation/evaluate_CI_complete.py kiba_',i,'_cont.txt'), intern = TRUE)
  ci[i] = as.numeric(str_extract(res_ci, "\\-*\\d+\\.*\\d*"))
  #ci[i] = parse.mid(res_ci)
}

kiba.mean = c(mean(rmse), mean(auc), mean(aupr), mean(ci))
kiba.sd = c(sd(rmse), sd(auc), sd(aupr), sd(ci))
save(kiba.mean, kiba.sd, file='kiba.table.rda')


