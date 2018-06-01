require(xgboost)
source('Feature.heterous.R')
library(foreach)
library(doParallel)

similarity.knn = function(sim, k) {
  n = nrow(sim)
  matind = matrix(0, n, n)
  for (i in 1:n) {
    ind = order(sim[i,], decreasing = TRUE)[1:k]
    matind[i, ind] = 1
  }
  matind = 1-(1-matind)*(1-t(matind))
  sim = sim*matind
  diag(sim) = 0
  return(sim)
}

clear.cold.start = function(triplet, d.sim, t.sim, min.num) {
  flag = FALSE
  while (!flag) {
    flag = TRUE
    d.dict = sort(unique(triplet[,1]))
    for (dd in d.dict) {
      ind = which(triplet[,1] == dd)
      if (length(ind) < min.num) {
        flag = FALSE
        triplet = triplet[-ind,]
        # d.sim = d.sim[-dd, -dd]
      }
    }
    t.dict = sort(unique(triplet[,2]))
    for (tt in t.dict) {
      ind = which(triplet[,2] == tt)
      if (length(ind) < min.num) {
        flag = FALSE
        triplet = triplet[-ind,]
        # t.sim = t.sim[-tt, -tt]
      }
    }
  }
  d.dict = sort(unique(triplet[,1]))
  t.dict = sort(unique(triplet[,2]))
  d.sim = d.sim[d.dict, d.dict]
  t.sim = t.sim[t.dict, t.dict]
  triplet[,1] = match(triplet[,1], d.dict)
  triplet[,2] = match(triplet[,2], t.dict)
  return(list(triplet, d.sim, t.sim))
}

feature.construction = function(train.triplet, test.triplet, 
                                d.sim, t.sim, k, latent.dim,
                                threshold, threshold.identity) {
  #d.sim = similarity.knn(d.sim, 4)
  #t.sim = similarity.knn(t.sim, 5)
  #browser()
  time_id_start = Sys.time()
  d.identity = get.identity(train.triplet[,1], train.triplet[,3], 
                            d.sim, threshold.identity)
  colnames(d.identity) = paste0('d.', colnames(d.identity))
  #browser()
  t.identity = get.identity(train.triplet[,2], train.triplet[,3], 
                            t.sim, threshold.identity)

  colnames(t.identity) = paste0('t.', colnames(t.identity))
  time_id_end = Sys.time()
  time_id_diff = as.double(difftime(time_id_end, time_id_start, units="secs"))
  cat(paste0("Time took to get the identity: ", time_id_diff, ' seconds', '\n'))
  
  time_homo_start = Sys.time()
  d.homo = get.homogeneous.feature(train.triplet[,1], train.triplet[,3],
                                   d.sim, 4, threshold, threshold.identity)
  colnames(d.homo) = paste0('d.', colnames(d.homo))
  t.homo = get.homogeneous.feature(train.triplet[,2], train.triplet[,3],
                                   t.sim, 4, threshold, threshold.identity)
  colnames(t.homo) = paste0('t.', colnames(t.homo))
  time_homo_end = Sys.time()
  time_homo_diff = as.double(difftime(time_homo_end, time_homo_start, units="secs"))
  cat(paste0("Time took to get the homogeneous features: ", time_homo_diff,
    ' seconds', '\n'))
  
  time_hetero_start = Sys.time()
  f.hetero = get.heterogenous.feature(train.triplet[,1], train.triplet[,2],
                                      train.triplet[,3], 4,
                                      test.triplet[,1], test.triplet[,2],
                                      d.sim, t.sim, 0.6, 0.6, latent.dim)
  time_hetero_end = Sys.time()
  time_hetero_diff = as.double(difftime(time_hetero_end, time_hetero_start, units="secs"))
  cat(paste0("Time took to get the heterogenous features: ", time_hetero_diff,
    ' seconds', '\n'))

  d.hetero = f.hetero[[2]]
  t.hetero = f.hetero[[3]]
  d.t.hetero = f.hetero[[1]]
  
  d.feature = cbind(d.identity, d.homo, d.hetero)
  # colnames(d.feature) = paste0('d.', colnames(d.feature))
  t.feature = cbind(t.identity, t.homo, t.hetero)
  # colnames(t.feature) = paste0('t.', colnames(t.feature))
  
  train.feature = cbind(d.feature[train.triplet[,1],],
                        t.feature[train.triplet[,2],],
                        d.t.hetero[1:nrow(train.triplet),])
  test.feature = cbind(d.feature[test.triplet[,1],],
                       t.feature[test.triplet[,2],],
                       d.t.hetero[1:nrow(test.triplet),])
  train.y = train.triplet[,3]
  test.y = test.triplet[,3]
  return(list(train.feature, test.feature, train.y, test.y))
}

get.cv.fold = function(triplet, nfold) {
  n = nrow(triplet)
  train.fold = vector(nfold, mode='list')
  test.fold = vector(nfold, mode='list')
  
  # label = sample(1:nfold, n, replace = TRUE)
  label = rep(sample(1:nfold), n%/%nfold)
  m = n-length(label)
  label = c(label, sample(1:nfold, m))
  t.dict = sort(unique(triplet[,2]))
  for (tt in t.dict) {
    ind = which(triplet[,2] == tt)
    if (length(ind) <= 1){
      stop("Not enough observations to do cv folding!")
    }
    if (length(unique(label[ind])) == 1) {
      # If the label for the triplets corresponding to the target tt are identical.
      cat(tt,'\n')
      cat(label[ind],'\n')
      flag = FALSE
      l = 1
      counter = 0
      while (!flag && l <= length(ind)) {
        ind.x = ind[l]
        dd = triplet[ind.x, 1]
        ind.d = which(triplet[,1] == dd)
        if (length(ind.d) <= 1){
          stop("Not enough observations to do cv folding!")
        }
        d.label = label[ind.d]
        new.lb = (label[ind.x] + 1) %% nfold
        new.label = label
        new.label[ind.x] = new.lb
        new.d.label = label[ind.d]
        if (length(unique(d.label)) == 1){
          # The next line would necessarily make flag=TRUE.          
          label[ind.x] = new.lb
          d.label = label[ind.d]
        }    
        else if (length(unique(new.d.label)) > 1) {
          label[ind.x] = new.lb
          d.label = label[ind.d]
        }
        else{
          l = l+1
        }
        flag = ((length(unique(d.label)) > 1) && (length(unique(label[ind])) > 1))
        counter = counter + 1
        if (counter >= 80000){
          browser()
          stop("Dead loop!")
        }       
      }
      if (!flag) {
        browser()
        stop("Not able to do cv folding!")
      }
    }
  }
  # for (tt in t.dict) {
  #   ind = which(triplet[,2] == tt)
  #   if (length(unique(label[ind])) == 1) {
  #     # If the label for the triplets corresponding to the target tt are identical.
  #     cat(tt,'\n')
  #     cat(label[ind],'\n')
  #     flag = FALSE
  #     l = 1
  #     while (!flag && l <= length(ind)) {
  #       ind.x = ind[l]
  #       dd = triplet[ind.x, 1]
  #       ind.d = which(triplet[,1] == dd)
  #       new.lb = nfold + 1 - label[ind.x]
  #       new.d.label = label[ind.d]
  #       new.d.label[l] = new.lb
  #       if (length(unique(new.d.label)) > 1) {
  #         flag = TRUE
  #         label[ind.x] = new.lb          
  #       } else {
  #         l = l+1
  #       }
  #     }
  #     if (!flag) {
  #       stop("Not able to do cv folding!")
  #     }
  #   }
  # }
  for (i in 1:nfold) {
    test.fold[[i]] = which(label == i)
  }
  # for (i in 1:n) {
  #   test.fold[[(i %% nfold) + 1]]
  # }
  for (i in 1:nfold) {
    train.fold[[i]] = setdiff(1:n, test.fold[[i]])
  }
  cv.folds = vector(nfold, mode='list')
  for (i in 1:nfold) {
    cv.folds[[i]] = list(train.fold[[i]], test.fold[[i]])
  }
  return(cv.folds)
}

get.cv.fold.quad = function(quadruplet, nfold) {
  n = nrow(quadruplet)
  train.fold = vector(nfold, mode='list')
  test.fold = vector(nfold, mode='list')
  label = quadruplet[, 4]
  
  for (i in 1:nfold) {
    test.fold[[i]] = which(label == i)
  }
  
  for (i in 1:nfold) {
    train.fold[[i]] = setdiff(1:n, test.fold[[i]])
  }
  cv.folds = vector(nfold, mode='list')
  for (i in 1:nfold) {
    cv.folds[[i]] = list(train.fold[[i]], test.fold[[i]])
  }
  return(cv.folds)
}

# The original implementation.
sequential.cv.feature.old = function(triplet, d.sim, t.sim, latent.dim,
                                 nfold, k, threshold, threshold.identity) {
  n = nrow(triplet)
  cv.folds = get.cv.fold(triplet, nfold)
  cv.models = vector(nfold, mode='list')
  cv.preds = rep(0, n)
  cv.data = vector(nfold, mode='list')
  # no_cores = 5
  # registerDoParallel(makeCluster(no_cores))
  # cv.data = foreach(i = 1:nfold) %dopar% {
  #   source('Sequential.cv.R') 
  #   #sink("temporary.txt", append=T, type = "output")
  #   cat(i,'\n')
  #   train.fold = cv.folds[[i]][[1]] # $train.fold
  #   test.fold = cv.folds[[i]][[2]] # $test.fold
  #   train.triplet = triplet[train.fold,]
  #   test.triplet = triplet[test.fold,]
  #   time_feature_start = Sys.time()
  #   features = feature.construction(train.triplet, test.triplet, 
  #                                   d.sim, t.sim, k, latent.dim,
  #                                   threshold, threshold.identity)
  #   time_feature_end = Sys.time()
  #   time_diff = as.double(difftime(time_feature_end, time_feature_start, units="secs"))
  #   cat(paste0("Time used in feature.construction(): ", time_diff,
  #     ' seconds', '\n'))
  #   #sink()
  #   train.x = features[[1]]
  #   train.y = features[[3]]
    
  #   test.x = features[[2]]
  #   test.y = features[[4]]
  #   #cv.data[[i]] = list(train.x, test.x, train.y, test.y)
  #   list(train.x, test.x, train.y, test.y)
  # }
  # stopImplicitCluster()
  # registerDoSEQ()
  for (i in 1:nfold) {
    #sink("temporary.txt", append=T, type = "output")
    cat(i,'\n')
    train.fold = cv.folds[[i]][[1]] # $train.fold
    test.fold = cv.folds[[i]][[2]] # $test.fold
    train.triplet = triplet[train.fold,]
    test.triplet = triplet[test.fold,]
    time_feature_start = Sys.time()
    features = feature.construction(train.triplet, test.triplet, 
                                    d.sim, t.sim, k, latent.dim,
                                    threshold, threshold.identity)
    time_feature_end = Sys.time()
    time_diff = as.double(difftime(time_feature_end, time_feature_start, units="secs"))
    cat(paste0("Time used in feature.construction(): ", time_diff,
      ' seconds', '\n'))
    #sink()
    train.x = features[[1]]
    train.y = features[[3]]
    
    test.x = features[[2]]
    test.y = features[[4]]
    cv.data[[i]] = list(train.x, test.x, train.y, test.y)
  }
  return(list(cv.data, cv.folds))
}

sequential.cv.feature = function(quadruplet, d.sim, t.sim, latent.dim,
                                 nfold, k, threshold, threshold.identity) {
  n = nrow(quadruplet)
  cv.folds = get.cv.fold.quad(quadruplet, nfold)
  cv.models = vector(nfold, mode='list')
  cv.preds = rep(0, n)
  cv.data = vector(nfold, mode='list')
  triplet = quadruplet[,1:3]
  no_cores = 5
  registerDoParallel(makeCluster(no_cores))
  cv.data = foreach(i = 1:nfold) %dopar% {
    source('Sequential.cv.R') 
    #sink("temporary.txt", append=T, type = "output")
    cat(i,'\n')
    train.fold = cv.folds[[i]][[1]] # $train.fold
    test.fold = cv.folds[[i]][[2]] # $test.fold
    train.triplet = triplet[train.fold,]
    test.triplet = triplet[test.fold,]
    time_feature_start = Sys.time()
    features = feature.construction(train.triplet, test.triplet, 
                                    d.sim, t.sim, k, latent.dim,
                                    threshold, threshold.identity)
    time_feature_end = Sys.time()
    time_diff = as.double(difftime(time_feature_end, time_feature_start, units="secs"))
    cat(paste0("Time used in feature.construction(): ", time_diff,
      ' seconds', '\n'))
    #sink()
    train.x = features[[1]]
    train.y = features[[3]]
    
    test.x = features[[2]]
    test.y = features[[4]]
    #cv.data[[i]] = list(train.x, test.x, train.y, test.y)
    list(train.x, test.x, train.y, test.y)
  }
  stopImplicitCluster()
  registerDoSEQ()
  # for (i in 1:nfold) {
  #   #sink("temporary.txt", append=T, type = "output")
  #   cat(i,'\n')
  #   train.fold = cv.folds[[i]][[1]] # $train.fold
  #   test.fold = cv.folds[[i]][[2]] # $test.fold
  #   train.triplet = triplet[train.fold,]
  #   test.triplet = triplet[test.fold,]
  #   time_feature_start = Sys.time()
  #   features = feature.construction(train.triplet, test.triplet, 
  #                                   d.sim, t.sim, k, latent.dim,
  #                                   threshold, threshold.identity)
  #   time_feature_end = Sys.time()
  #   time_diff = as.double(difftime(time_feature_end, time_feature_start, units="secs"))
  #   cat(paste0("Time used in feature.construction(): ", time_diff,
  #     ' seconds', '\n'))
  #   #sink()
  #   train.x = features[[1]]
  #   train.y = features[[3]]
    
  #   test.x = features[[2]]
  #   test.y = features[[4]]
  #   cv.data[[i]] = list(train.x, test.x, train.y, test.y)
  # }
  return(list(cv.data, cv.folds))
}

 