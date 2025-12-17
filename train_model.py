import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import optuna
import pyper

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import Lasso, ElasticNet


# make test dataset & training dataset using X-fold cross validation
def make_data(trait_name, genotype, phenotype, CV=5, seed=1024, plot=True, log=True):
    # read dataset
    phenotype = pd.read_csv(phenotype, index_col=0)
    genotype = pd.read_csv(genotype)
    # genotype = genotype.loc[:, genotype.isna().sum() < genotype.shape[0]/2] # remove many NA columns
    genotype = genotype.loc[:, genotype.isna().sum() < genotype.shape[0]] # remove many NA columns
    # extract ids
    ids = list(set(phenotype["Line"]) & set(genotype.columns[2:]))
    ids.sort()
    # extract phenotype & genotype data of RILs
    phenotype = phenotype[phenotype.Line.isin(ids)]
    genotype_columns = ["Chrom", "Position"]
    genotype_columns.extend(ids)
    genotype = genotype.loc[:, genotype_columns]
    # make phenotype & genotype data
    phenotype_data = phenotype.loc[:, ["Line", trait_name]]
    phenotype_data = phenotype_data.set_index("Line")
    position_data = genotype.loc[:, ["Chrom", "Position"]]
    genotype = genotype.drop(genotype.columns[0:2], axis=1)
    genotype = genotype.T
    if log:
        print("The phenotype data:", phenotype_data[trait_name].notna().sum(), "The genotype data:", genotype.shape[0])
    # merge phenotype & genotype data
    merge_data = pd.concat([phenotype_data, genotype], axis=1, join="inner")
    merge_data_index = merge_data.index.values
    # remove lines without phenotype values & common genotype SNP across all lines
    merge_data = np.array(merge_data)
    merge_data_index = merge_data_index[~np.isnan(merge_data)[:, 0]]
    merge_data = merge_data[~np.isnan(merge_data)[:, 0]]
    merge_data = pd.DataFrame(merge_data)
    merge_data.index = merge_data_index
    if log:
        print("The merge data:", merge_data.shape)

    del genotype, phenotype, phenotype_data

    # separate dataset to train, test.
    train_data = []
    test_data = []
    kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
    for train, test in kf.split(range(merge_data.shape[0])):
        train_data.append([merge_data.iloc[train, 1:], merge_data.iloc[train, 0]])
        test_data.append([merge_data.iloc[test, 1:], merge_data.iloc[test, 0]])
    
    del merge_data
    
    if plot:
        fig = plt.figure(figsize=(15,3))
        for i in range(CV):
            ax1 = fig.add_subplot(1, 5, i+1)
            sns.histplot(train_data[i][1], stat="density", color="r", label="train", ax=ax1)
            sns.histplot(test_data[i][1], stat="density", color="b", label="test", ax=ax1)
        plt.legend()
        plt.show()

    return test_data, train_data, position_data


# calculate accuracy like r2
def calc_acc(y_test_preds, test_data, metrics, plot=True):
    if isinstance(y_test_preds, list):
        y_test_pred = pd.DataFrame(y_test_preds).mean().values
    else:
        y_test_pred = y_test_preds
    y_test = test_data[1].values
    if plot:
        yyplot(y_test, y_test_pred)
    if metrics == "r2":
        acc = np.corrcoef(y_test, y_test_pred)[0][1]
    elif metrics == "rmse":
        acc = np.sqrt(mean_squared_error(y_test, y_test_pred))
    elif metrics == "mae":
        acc = mean_absolute_error(y_test, y_test_pred)
    return acc


# parameter tuning for ElasticNet using optuna
def objective_variable_data_Enet(X, y, optuna_seed):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0001, 10)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        regr = ElasticNet(alpha = alpha,
                          l1_ratio = l1_ratio,
                          random_state=optuna_seed)
        original_sk = KFold(n_splits=5, shuffle=True, random_state=1024)
        score = cross_val_score(regr, X, y, cv=original_sk, scoring="neg_mean_squared_error")
        return np.mean(score)
    return objective 

def tuning_ElasticNet(X, y, optuna_seed=1024):
    # ElasticNet Optuna Tuning
    X = X.fillna(0)
    y = y
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_variable_data_Enet(X, y, optuna_seed), n_trials=100)
    print('ElasticNet Best params:{}'.format(study.best_params))
    return study.best_params


# train ElasticNet model
def train_Enet(test_data, train_data, params):
    
    alpha = params["alpha"]
    l1_ratio = params["l1_ratio"]      
    X_train = train_data[0].fillna(0)
    y_train = train_data[1]
    X_test = test_data[0].fillna(0)
    # train model
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
    clf.fit(X_train, y_train)
    # get coefs
    coefs = clf.coef_
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, coefs


# train GBLUP model by rrBLUP
def train_GBLUP(test_data, train_data, genotype):
    X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
    X_column = test_data[0].columns.values
    y_train = train_data[1].sort_index()
    y_train = pd.DataFrame(y_train).reset_index()
    y_train.columns = ["line", "y"]
    
    r = pyper.R(use_numpy='True', use_pandas='True')
    r.assign("genotype", genotype)
    r.assign("X_index", X_index)
    r.assign("X_column", X_column)
    r.assign("y_train", y_train)
    code = """
    library(rrBLUP)
    X <- read.csv(genotype)
    X <- X[X_column, X_index]
    M <- t(X-1)
    A <- A.mat(M)
    ans <- kin.blup(data=y_train,geno='line',pheno='y',K=A)
    pred <- ans$pred
    VarG <- ans$Vg
    VarE <- ans$Ve
    """
    r(code)
    all_pred = r.get("pred")
    all_pred = pd.DataFrame({"y":all_pred})
    all_pred.index = X_index
    
    # calc estimated heritaility
    VarG = r.get("VarG")
    VarE = r.get("VarE")
    h2 = VarG / (VarG + VarE)
    
    # predicted values of test phenotype
    test_pred = all_pred.loc[test_data[0].index.values, :]
    
    # calc prediction accuracy
    r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
    
    return test_pred.y.values, h2, r2


# get coefficient using rrBLUP
def GBLUP_coef(train_data, genotype):
    X_train = train_data[0]
    X_index = X_train.sort_index().index.values
    X_column = X_train.columns.values
    y_train = train_data[1].sort_index()
    r = pyper.R(use_numpy='True', use_pandas='True')
    r.assign("genotype", genotype)
    r.assign("X_index", X_index)
    r.assign("X_column", X_column)
    r.assign("y", y_train.values)
    code = """
    library(rrBLUP)
    X <- read.csv(genotype)
    X <- X[X_column, X_index]
    M <- t(X-1)
    M[is.na(M)] <- 0
    ans <- mixed.solve(y,Z=M)
    coef <- ans$u
    """
    r(code)
    coef = r.get("coef")
    return coef

