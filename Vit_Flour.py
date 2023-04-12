#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:56:44 2020

@author: macbookpro
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from scipy import io
import scipy as sp
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut,LeaveOneOut, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV, LeaveOneGroupOut,LeaveOneOut,train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



dirname = '/Users/aykuteken/Documents/MATLAB'
fname = os.path.join(dirname, 'Feat_Vit_Flour.mat')
FVs = io.loadmat(fname,mat_dtype=True,matlab_compatible=True)


corr_dxy = FVs['Feat_Vec_corr_dxy']
dtw_dxy = FVs['Feat_Vec_dtw_dxy']
corr_oxy = FVs['Feat_Vec_corr_oxy']
dtw_oxy = FVs['Feat_Vec_dtw_oxy']

class_flour = np.ravel(FVs['class_flour'])
class_vit = np.ravel(FVs['class_vit'])

Reg_Info = FVs['Reg_name_comb'][0]

## Feature selection

## Corr deoxy flour
corr_dxy_new_flour= LinearSVC(C=0.1, penalty="l1", dual=False).fit(corr_dxy, class_flour)
corr_dxy_new_flour = SelectFromModel(corr_dxy_new_flour, prefit=True)
corr_dxy_new_flour = corr_dxy_new_flour.transform(corr_dxy)

## Corr oxy flour

corr_oxy_new_flour = LinearSVC(C=0.1, penalty="l1", dual=False).fit(corr_oxy, class_flour)
corr_oxy_new_flour = SelectFromModel(corr_oxy_new_flour, prefit=True)
corr_oxy_new_flour = corr_oxy_new_flour.transform(corr_oxy)

## DTW deoxy flour
dtw_dxy_new_flour = LinearSVC(C=0.06, penalty="l1", dual=False).fit(dtw_dxy, class_flour)
dtw_dxy_new_flour = SelectFromModel(dtw_dxy_new_flour, prefit=True)
dtw_dxy_new_flour = dtw_dxy_new_flour.transform(dtw_dxy)

## DTW oxy flour
dtw_oxy_new_flour = LinearSVC(C=0.06, penalty="l1", dual=False).fit(dtw_oxy, class_flour)
dtw_oxy_new_flour = SelectFromModel(dtw_oxy_new_flour, prefit=True)
dtw_oxy_new_flour = dtw_oxy_new_flour.transform(dtw_oxy)

## corr deoxy vital

corr_dxy_new_vit = LinearSVC(C=0.1, penalty="l1", dual=False).fit(corr_dxy, class_vit)
corr_dxy_new_vit = SelectFromModel(corr_dxy_new_vit, prefit=True)
corr_dxy_new_vit = corr_dxy_new_vit.transform(corr_dxy)

## corr oxy vital

corr_oxy_new_vit = LinearSVC(C=0.1, penalty="l1", dual=False).fit(corr_oxy, class_vit)
corr_oxy_new_vit = SelectFromModel(corr_oxy_new_vit, prefit=True)
corr_oxy_new_vit = corr_oxy_new_vit.transform(corr_oxy)

## DTW deoxy vital

dtw_dxy_new_vit = LinearSVC(C=0.06, penalty="l1", dual=False).fit(dtw_dxy, class_vit)
dtw_dxy_new_vit = SelectFromModel(dtw_dxy_new_vit, prefit=True)
dtw_dxy_new_vit = dtw_dxy_new_vit.transform(dtw_dxy)

## DTW oxy vital

dtw_oxy_new_vit = LinearSVC(C=0.06, penalty="l1", dual=False).fit(dtw_oxy, class_vit)
dtw_oxy_new_vit = SelectFromModel(dtw_oxy_new_vit, prefit=True)
dtw_oxy_new_vit = dtw_oxy_new_vit.transform(dtw_oxy)


#vit_dtw= np.concatenate((dtw_oxy_new_vit,dtw_dxy_new_vit),axis=1)
#flour_dtw= np.concatenate((dtw_oxy_new_flour, dtw_dxy_new_flour),axis=1)
#vit_corr = np.concatenate((corr_oxy_new_vit, corr_dxy_new_vit),axis=1)
#flour_corr = np.concatenate((corr_oxy_new_flour, corr_dxy_new_flour),axis=1)

vit_dtw_oxy = dtw_oxy_new_vit
flour_dtw_oxy = dtw_oxy_new_flour
vit_corr_oxy = corr_oxy_new_vit
flour_corr_oxy = corr_oxy_new_flour
 

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Poly SVM",
    "Gradient Boosting",
    "AdaBoost",
    "Naive Bayes",
    "Linear Discriminant Analysis",
    "Quadratic Discriminant Analysis",
    "Logistic Regression",
    "Perceptron",
    ]
params_classifiers = {
    "Nearest Neighbors": {
        'n_neighbors': range(1, 11, 2),
        'weights': ('uniform', 'distance', ), 
        },
    "Linear SVM": {
        'kernel': ('linear', ),
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [.001,.01, .1, 1],
        'class_weight': ('balanced', None, ),},
    "RBF SVM": {
        'kernel': ('rbf', ),
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [.001,.01, .1, 1, 10, 100],
        'class_weight': ('balanced', None, ), },
    # "Poly SVM": {
    #     'kernel': ('poly', ),
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'gamma': [.001,.01, .1, 1, 10, 100],
    #     'degree': range(2,6),
    #     'class_weight': ('balanced', None, ),
    #     },
    "Gradient Boosting": {
        'n_estimators': [50, 100, ],
        'learning_rate': [.1, 0.05, 0.01, 0.005, 0.001, ], },
    "AdaBoost": {
        'n_estimators': [50, 100, ],
        'learning_rate': [.1, 0.05, 0.01, 0.005, 0.001, ], },
    "Naive Bayes": {},
    "Linear Discriminant Analysis": {},
    "Quadratic Discriminant Analysis": {},
    "Logistic Regression": {
        'penalty': ('l1', 'l2', ),
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, ], },
    }

classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True),
    SVC(probability=True),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    ]


all_vit_corr_acc_mean_oxy=[]
all_vit_corr_acc_std_oxy=[]
all_vit_dtw_acc_mean_oxy=[]
all_vit_dtw_acc_std_oxy=[]
all_flour_corr_acc_mean_oxy=[]
all_flour_corr_acc_std_oxy=[]
all_flour_dtw_acc_mean_oxy=[]
all_flour_dtw_acc_std_oxy=[]

all_nested_scores_flour_corr_oxy =[]
all_nested_scores_flour_corr_dxy =[]
all_nested_scores_flour_corr_fuse =[]
all_nested_scores_flour_dtw_oxy =[]
all_nested_scores_flour_dtw_dxy =[]
all_nested_scores_flour_dtw_fuse =[]

all_nested_scores_vit_corr_oxy =[]
all_nested_scores_vit_corr_dxy =[]
all_nested_scores_vit_corr_fuse =[]
all_nested_scores_vit_dtw_oxy =[]
all_nested_scores_vit_dtw_dxy =[]
all_nested_scores_vit_dtw_fuse =[]

all_fprs_vit_corr_oxy =[]
all_fprs_vit_dtw_oxy =[]
all_fprs_flour_corr_oxy =[]
all_fprs_flour_dtw_oxy =[]

all_tprs_vit_corr_oxy =[]
all_tprs_vit_dtw_oxy =[]
all_tprs_flour_corr_oxy =[]
all_tprs_flour_dtw_oxy =[]

all_fprs_vit_corr_dxy =[]
all_fprs_vit_dtw_dxy =[]
all_fprs_flour_corr_dxy =[]
all_fprs_flour_dtw_dxy =[]

all_tprs_vit_corr_dxy =[]
all_tprs_vit_dtw_dxy =[]
all_tprs_flour_corr_dxy =[]
all_tprs_flour_dtw_dxy =[]

all_fprs_vit_corr_fuse =[]
all_fprs_vit_dtw_fuse =[]
all_fprs_flour_corr_fuse =[]
all_fprs_flour_dtw_fuse =[]

all_tprs_vit_corr_fuse =[]
all_tprs_vit_dtw_fuse =[]
all_tprs_flour_corr_fuse =[]
all_tprs_flour_dtw_fuse =[]

mean_tpr_vit_corr_oxy =[]
std_tpr_vit_corr_oxy =[]
mean_fpr_vit_corr_oxy =[]
mean_auc_vit_corr_oxy =[]

mean_tpr_vit_dtw_oxy =[]
std_tpr_vit_dtw_oxy =[]
mean_fpr_vit_dtw_oxy =[]
mean_auc_vit_dtw_oxy =[]

mean_tpr_flour_corr_oxy =[]
std_tpr_flour_corr_oxy =[]
mean_fpr_flour_corr_oxy =[]
mean_auc_flour_corr_oxy =[]

mean_tpr_flour_dtw_oxy =[]
std_tpr_flour_dtw_oxy =[]
mean_fpr_flour_dtw_oxy =[]
mean_auc_flour_dtw_oxy =[]

mean_tpr_vit_corr_dxy =[]
std_tpr_vit_corr_dxy =[]
mean_fpr_vit_corr_dxy =[]
mean_auc_vit_corr_dxy =[]

mean_tpr_vit_dtw_dxy =[]
std_tpr_vit_dtw_dxy =[]
mean_fpr_vit_dtw_dxy =[]
mean_auc_vit_dtw_dxy =[]

mean_tpr_flour_corr_dxy =[]
std_tpr_flour_corr_dxy =[]
mean_fpr_flour_corr_dxy =[]
mean_auc_flour_corr_dxy =[]

mean_tpr_flour_dtw_dxy =[]
std_tpr_flour_dtw_dxy =[]
mean_fpr_flour_dtw_dxy =[]
mean_auc_flour_dtw_dxy =[]

mean_tpr_vit_corr_fuse =[]
std_tpr_vit_corr_fuse =[]
mean_fpr_vit_corr_fuse =[]
mean_auc_vit_corr_fuse =[]

mean_tpr_vit_dtw_fuse =[]
std_tpr_vit_dtw_fuse =[]
mean_fpr_vit_dtw_fuse =[]
mean_auc_vit_dtw_fuse =[]

mean_tpr_flour_corr_fuse =[]
std_tpr_flour_corr_fuse =[]
mean_fpr_flour_corr_fuse =[]
mean_auc_flour_corr_fuse =[]

mean_tpr_flour_dtw_fuse =[]
std_tpr_flour_dtw_fuse =[]
mean_fpr_flour_dtw_fuse =[]
mean_auc_flour_dtw_fuse =[]


mean_fpr = np.linspace(0, 1, 100)


print('---------%%% For Oxy Hb Features %%%-----------')
for name, classifier in zip(names, classifiers):
    if name in ('Linear SVM', 'RBF SVM',
                'Gradient Boosting', 'AdaBoost', 'Logistic Regression'):
        verbose = 0
    else:
        verbose = 0
    print(name)
    time.sleep(0.1)
    NUM_TRIALS = 30
    nested_scores_vit_corr_oxy = np.zeros(NUM_TRIALS)
    nested_scores_vit_dtw_oxy = np.zeros(NUM_TRIALS)
    nested_scores_flour_corr_oxy = np.zeros(NUM_TRIALS)
    nested_scores_flour_dtw_oxy= np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        param_search = GridSearchCV(
            estimator=classifier, param_grid=params_classifiers[name],
            verbose=verbose, cv =inner_cv)

        # Nested CV with parameter optimization
        nested_score_vit_corr_oxy = cross_val_score(param_search, X=vit_corr_oxy, y=class_vit, cv=outer_cv)
        nested_scores_vit_corr_oxy[i] = nested_score_vit_corr_oxy.mean()
        
        nested_score_vit_dtw_oxy = cross_val_score(param_search, X=vit_dtw_oxy, y=class_vit, cv=outer_cv)
        nested_scores_vit_dtw_oxy[i] = nested_score_vit_dtw_oxy.mean()
        

        nested_score_flour_corr_oxy = cross_val_score(param_search, X=flour_corr_oxy, y=class_flour, cv=outer_cv)
        nested_scores_flour_corr_oxy[i] = nested_score_flour_corr_oxy.mean()
        
        
        nested_score_flour_dtw_oxy = cross_val_score(param_search, X=flour_dtw_oxy, y=class_flour, cv=outer_cv)
        nested_scores_flour_dtw_oxy[i] = nested_score_flour_dtw_oxy.mean()
        
        
        
        # Nested CV Prediction
        prob_vit_corr_oxy = cross_val_predict(param_search, X=vit_corr_oxy, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_vit_dtw_oxy = cross_val_predict(param_search, X=vit_dtw_oxy, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_flour_corr_oxy = cross_val_predict(param_search, X=flour_corr_oxy, y=class_flour, cv=outer_cv,method='predict_proba')
        
        prob_flour_dtw_oxy = cross_val_predict(param_search, X=flour_dtw_oxy, y=class_flour, cv=outer_cv,method='predict_proba')
        
        
        fprs_vit_corr_oxy,tprs_vit_corr_oxy,_=roc_curve(class_vit,prob_vit_corr_oxy[:,1])
        fprs_vit_dtw_oxy,tprs_vit_dtw_oxy,_=roc_curve(class_vit,prob_vit_dtw_oxy[:,1])
        fprs_flour_corr_oxy,tprs_flour_corr_oxy,_=roc_curve(class_flour,prob_flour_corr_oxy[:,1])
        fprs_flour_dtw_oxy,tprs_flour_dtw_oxy,_=roc_curve(class_flour,prob_flour_dtw_oxy[:,1])
        
        
        tpr_vit_corr_oxy=np.interp(mean_fpr,fprs_vit_corr_oxy,tprs_vit_corr_oxy)
        tpr_vit_corr_oxy[0] =0.0

        tpr_vit_dtw_oxy=np.interp(mean_fpr,fprs_vit_dtw_oxy,tprs_vit_dtw_oxy)
        tpr_vit_dtw_oxy[0] =0.0
        
        tpr_flour_corr_oxy=np.interp(mean_fpr,fprs_flour_corr_oxy,tprs_flour_corr_oxy)
        tpr_flour_corr_oxy[0] =0.0       

        tpr_flour_dtw_oxy=np.interp(mean_fpr,fprs_flour_dtw_oxy,tprs_flour_dtw_oxy)
        tpr_flour_dtw_oxy[0] =0.0              
        
        all_fprs_vit_corr_oxy.append(fprs_vit_corr_oxy)
        all_fprs_vit_dtw_oxy.append(fprs_vit_dtw_oxy)
        all_fprs_flour_corr_oxy.append(fprs_flour_corr_oxy)
        all_fprs_flour_dtw_oxy.append(fprs_flour_dtw_oxy)
        
        all_tprs_vit_corr_oxy.append(tpr_vit_corr_oxy)
        all_tprs_vit_dtw_oxy.append(tpr_vit_dtw_oxy)
        all_tprs_flour_corr_oxy.append(tpr_flour_corr_oxy)
        all_tprs_flour_dtw_oxy.append(tpr_flour_dtw_oxy)
        
    
    m_tpr = np.mean(all_tprs_vit_corr_oxy,axis=0)
    std_tpr =np.std(all_tprs_vit_corr_oxy,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_corr_oxy.append(m_tpr)
    std_tpr_vit_corr_oxy.append(std_tpr)
    mean_fpr_vit_corr_oxy.append(mean_fpr)
    mean_auc_vit_corr_oxy.append(auc(mean_fpr,m_tpr))
    
    m_tpr = np.mean(all_tprs_vit_dtw_oxy,axis=0)
    std_tpr =np.std(all_tprs_vit_dtw_oxy,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_dtw_oxy.append(m_tpr)
    std_tpr_vit_dtw_oxy.append(std_tpr)
    mean_fpr_vit_dtw_oxy.append(mean_fpr)
    mean_auc_vit_dtw_oxy.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_corr_oxy,axis=0)
    std_tpr = np.std(all_tprs_flour_corr_oxy,axis=0)
    m_tpr [-1] =1.0   
    
    mean_tpr_flour_corr_oxy.append(m_tpr)
    std_tpr_flour_corr_oxy.append(std_tpr)
    mean_fpr_flour_corr_oxy.append(mean_fpr)
    mean_auc_flour_corr_oxy.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_dtw_oxy,axis=0)
    std_tpr = np.std(all_tprs_flour_dtw_oxy,axis=0)
    m_tpr [-1] =1.0   
    
    all_fprs_vit_corr_oxy =[]
    all_fprs_vit_dtw_oxy =[]
    all_fprs_flour_corr_oxy =[]
    all_fprs_flour_dtw_oxy =[]

    all_tprs_vit_corr_oxy =[]
    all_tprs_vit_dtw_oxy =[]
    all_tprs_flour_corr_oxy =[]
    all_tprs_flour_dtw_oxy =[]
    
    
    mean_tpr_flour_dtw_oxy.append(m_tpr)
    std_tpr_flour_dtw_oxy.append(std_tpr)
    mean_fpr_flour_dtw_oxy.append(mean_fpr)
    mean_auc_flour_dtw_oxy.append(auc(mean_fpr,m_tpr))
    
    
    all_nested_scores_flour_corr_oxy.append(nested_scores_flour_corr_oxy)
    all_nested_scores_vit_corr_oxy.append(nested_scores_vit_corr_oxy)
    all_nested_scores_flour_dtw_oxy.append(nested_scores_flour_dtw_oxy)
    all_nested_scores_vit_dtw_oxy.append(nested_scores_vit_dtw_oxy)
    
    
    
    print("Vital_corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_corr_oxy.mean(), nested_scores_vit_corr_oxy.std()))
    all_vit_corr_acc_mean_oxy.append(nested_scores_vit_corr_oxy.mean())
    all_vit_corr_acc_std_oxy.append(nested_scores_vit_corr_oxy.std())
    
    print("Vital_DTW: Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_dtw_oxy.mean(), nested_scores_vit_dtw_oxy.std()))
    all_vit_dtw_acc_mean_oxy.append(nested_scores_vit_dtw_oxy.mean())
    all_vit_dtw_acc_std_oxy.append(nested_scores_vit_dtw_oxy.std())

    print("Flour_Corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_corr_oxy.mean(), nested_scores_flour_corr_oxy.std()))
    all_flour_corr_acc_mean_oxy.append(nested_scores_flour_corr_oxy.mean())
    all_flour_corr_acc_std_oxy.append(nested_scores_flour_corr_oxy.std())

    print("Flour_DTW : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_dtw_oxy.mean(), nested_scores_flour_dtw_oxy.std()))
    all_flour_dtw_acc_mean_oxy.append(nested_scores_flour_dtw_oxy.mean())
    all_flour_dtw_acc_std_oxy.append(nested_scores_flour_dtw_oxy.std())



print('---------%%% For Deoxy Hb Features %%%-----------')

vit_dtw_dxy = dtw_dxy_new_vit
flour_dtw_dxy = dtw_dxy_new_flour
vit_corr_dxy = corr_dxy_new_vit
flour_corr_dxy = corr_dxy_new_flour



all_vit_corr_acc_mean_dxy=[]
all_vit_corr_acc_std_dxy=[]
all_vit_dtw_acc_mean_dxy=[]
all_vit_dtw_acc_std_dxy=[]
all_flour_corr_acc_mean_dxy=[]
all_flour_corr_acc_std_dxy=[]
all_flour_dtw_acc_mean_dxy=[]
all_flour_dtw_acc_std_dxy=[]


for name, classifier in zip(names, classifiers):
    if name in ('Linear SVM', 'RBF SVM',
                'Gradient Boosting', 'AdaBoost', 'Logistic Regression',
                ):
        verbose = 0
    else:
        verbose = 0
    print(name)
    time.sleep(0.1)
    NUM_TRIALS = 30
    nested_scores_vit_corr_dxy = np.zeros(NUM_TRIALS)
    nested_scores_vit_dtw_dxy = np.zeros(NUM_TRIALS)
    nested_scores_flour_corr_dxy = np.zeros(NUM_TRIALS)
    nested_scores_flour_dtw_dxy= np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        param_search = GridSearchCV(
            estimator=classifier, param_grid=params_classifiers[name],
            verbose=verbose, cv =inner_cv)

        # Nested CV with parameter optimization
        nested_score_vit_corr_dxy = cross_val_score(param_search, X=vit_corr_dxy, y=class_vit, cv=outer_cv)
        nested_scores_vit_corr_dxy[i] = nested_score_vit_corr_dxy.mean()

        # Nested CV with parameter optimization
        nested_score_vit_dtw_dxy = cross_val_score(param_search, X=vit_dtw_dxy, y=class_vit, cv=outer_cv)
        nested_scores_vit_dtw_dxy[i] = nested_score_vit_dtw_dxy.mean()

        nested_score_flour_corr_dxy = cross_val_score(param_search, X=flour_corr_dxy, y=class_flour, cv=outer_cv)
        nested_scores_flour_corr_dxy[i] = nested_score_flour_corr_dxy.mean()
        
        nested_score_flour_dtw_dxy = cross_val_score(param_search, X=flour_dtw_dxy, y=class_flour, cv=outer_cv)
        nested_scores_flour_dtw_dxy[i] = nested_score_flour_dtw_dxy.mean()

        # Nested CV Prediction
        prob_vit_corr_dxy = cross_val_predict(param_search, X=vit_corr_dxy, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_vit_dtw_dxy = cross_val_predict(param_search, X=vit_dtw_dxy, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_flour_corr_dxy = cross_val_predict(param_search, X=flour_corr_dxy, y=class_flour, cv=outer_cv,method='predict_proba')
        
        prob_flour_dtw_dxy = cross_val_predict(param_search, X=flour_dtw_dxy, y=class_flour, cv=outer_cv,method='predict_proba')
        
        
        fprs_vit_corr_dxy,tprs_vit_corr_dxy,_=roc_curve(class_vit,prob_vit_corr_dxy[:,1])
        fprs_vit_dtw_dxy,tprs_vit_dtw_dxy,_=roc_curve(class_vit,prob_vit_dtw_dxy[:,1])
        fprs_flour_corr_dxy,tprs_flour_corr_dxy,_=roc_curve(class_flour,prob_flour_corr_dxy[:,1])
        fprs_flour_dtw_dxy,tprs_flour_dtw_dxy,_=roc_curve(class_flour,prob_flour_dtw_dxy[:,1])
        
        
        tpr_vit_corr_dxy=np.interp(mean_fpr,fprs_vit_corr_dxy,tprs_vit_corr_dxy)
        tpr_vit_corr_dxy[0] =0.0

        tpr_vit_dtw_dxy=np.interp(mean_fpr,fprs_vit_dtw_dxy,tprs_vit_dtw_dxy)
        tpr_vit_dtw_dxy[0] =0.0
        
        tpr_flour_corr_dxy=np.interp(mean_fpr,fprs_flour_corr_dxy,tprs_flour_corr_dxy)
        tpr_flour_corr_dxy[0] =0.0       

        tpr_flour_dtw_dxy=np.interp(mean_fpr,fprs_flour_dtw_dxy,tprs_flour_dtw_dxy)
        tpr_flour_dtw_dxy[0] =0.0              
        
        all_fprs_vit_corr_dxy.append(fprs_vit_corr_dxy)
        all_fprs_vit_dtw_dxy.append(fprs_vit_dtw_dxy)
        all_fprs_flour_corr_dxy.append(fprs_flour_corr_dxy)
        all_fprs_flour_dtw_dxy.append(fprs_flour_dtw_dxy)
        
        all_tprs_vit_corr_dxy.append(tpr_vit_corr_dxy)
        all_tprs_vit_dtw_dxy.append(tpr_vit_dtw_dxy)
        all_tprs_flour_corr_dxy.append(tpr_flour_corr_dxy)
        all_tprs_flour_dtw_dxy.append(tpr_flour_dtw_dxy)
        
    
    m_tpr = np.mean(all_tprs_vit_corr_dxy,axis=0)
    std_tpr = np.std(all_tprs_vit_corr_dxy,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_corr_dxy.append(m_tpr)
    std_tpr_vit_corr_dxy.append(std_tpr)
    mean_fpr_vit_corr_dxy.append(mean_fpr)
    mean_auc_vit_corr_dxy.append(auc(mean_fpr,m_tpr))
    
    m_tpr = np.mean(all_tprs_vit_dtw_dxy,axis=0)
    std_tpr = np.std(all_tprs_vit_dtw_dxy,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_dtw_dxy.append(m_tpr)
    std_tpr_vit_dtw_dxy.append(std_tpr)
    mean_fpr_vit_dtw_dxy.append(mean_fpr)
    mean_auc_vit_dtw_dxy.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_corr_dxy,axis=0)
    std_tpr = np.std(all_tprs_flour_corr_dxy,axis=0)
    m_tpr [-1] =1.0   
    
    mean_tpr_flour_corr_dxy.append(m_tpr)
    std_tpr_flour_corr_dxy.append(std_tpr)
    mean_fpr_flour_corr_dxy.append(mean_fpr)
    mean_auc_flour_corr_dxy.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_dtw_dxy,axis=0)
    std_tpr = np.std(all_tprs_flour_dtw_dxy,axis=0)
    m_tpr [-1] =1.0   
    
    mean_tpr_flour_dtw_dxy.append(m_tpr)
    std_tpr_flour_dtw_dxy.append(std_tpr)
    mean_fpr_flour_dtw_dxy.append(mean_fpr)
    mean_auc_flour_dtw_dxy.append(auc(mean_fpr,m_tpr))
        
    all_fprs_vit_corr_dxy =[]
    all_fprs_vit_dtw_dxy =[]
    all_fprs_flour_corr_dxy =[]
    all_fprs_flour_dtw_dxy =[]

    all_tprs_vit_corr_dxy =[]
    all_tprs_vit_dtw_dxy =[]
    all_tprs_flour_corr_dxy =[]
    all_tprs_flour_dtw_dxy =[]
        
        
        
    
    all_nested_scores_flour_corr_dxy.append(nested_scores_flour_corr_dxy)
    all_nested_scores_vit_corr_dxy.append(nested_scores_vit_corr_dxy)
    all_nested_scores_flour_dtw_dxy.append(nested_scores_flour_dtw_dxy)
    all_nested_scores_vit_dtw_dxy.append(nested_scores_vit_dtw_dxy)
    
    
    
    
    
    print("Vital_corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_corr_dxy.mean(), nested_scores_vit_corr_dxy.std()))
    all_vit_corr_acc_mean_dxy.append(nested_scores_vit_corr_dxy.mean())
    all_vit_corr_acc_std_dxy.append(nested_scores_vit_corr_dxy.std())
    
    print("Vital_DTW: Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_dtw_dxy.mean(), nested_scores_vit_dtw_dxy.std()))
    all_vit_dtw_acc_mean_dxy.append(nested_scores_vit_dtw_dxy.mean())
    all_vit_dtw_acc_std_dxy.append(nested_scores_vit_dtw_dxy.std())

    print("Flour_Corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_corr_dxy.mean(), nested_scores_flour_corr_dxy.std()))
    all_flour_corr_acc_mean_dxy.append(nested_scores_flour_corr_dxy.mean())
    all_flour_corr_acc_std_dxy.append(nested_scores_flour_corr_dxy.std())

    print("Flour_DTW : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_dtw_dxy.mean(), nested_scores_flour_dtw_dxy.std()))
    all_flour_dtw_acc_mean_dxy.append(nested_scores_flour_dtw_dxy.mean())
    all_flour_dtw_acc_std_dxy.append(nested_scores_flour_dtw_dxy.std())

print('---------%%% For Fused (Oxy + Deoxy) Features %%%-----------')
vit_dtw_fuse= np.concatenate((dtw_oxy_new_vit,dtw_dxy_new_vit),axis=1)
flour_dtw_fuse= np.concatenate((dtw_oxy_new_flour, dtw_dxy_new_flour),axis=1)
vit_corr_fuse = np.concatenate((corr_oxy_new_vit, corr_dxy_new_vit),axis=1)
flour_corr_fuse = np.concatenate((corr_oxy_new_flour, corr_dxy_new_flour),axis=1)


all_vit_corr_acc_mean_fuse=[]
all_vit_corr_acc_std_fuse=[]
all_vit_dtw_acc_mean_fuse=[]
all_vit_dtw_acc_std_fuse=[]
all_flour_corr_acc_mean_fuse=[]
all_flour_corr_acc_std_fuse=[]
all_flour_dtw_acc_mean_fuse=[]
all_flour_dtw_acc_std_fuse=[]
for name, classifier in zip(names, classifiers):
    if name in ('Linear SVM', 'RBF SVM',
                'Gradient Boosting', 'AdaBoost', 'Logistic Regression',
                ):
        verbose = 0
    else:
        verbose = 0
    print(name)
    time.sleep(0.1)
    NUM_TRIALS = 30
    nested_scores_vit_corr_fuse = np.zeros(NUM_TRIALS)
    nested_scores_vit_dtw_fuse = np.zeros(NUM_TRIALS)
    nested_scores_flour_corr_fuse = np.zeros(NUM_TRIALS)
    nested_scores_flour_dtw_fuse = np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        param_search = GridSearchCV(
            estimator=classifier, param_grid=params_classifiers[name],
            verbose=verbose, cv =inner_cv)

        # Nested CV with parameter optimization
        nested_score_vit_corr_fuse = cross_val_score(param_search, X=vit_corr_fuse, y=class_vit, cv=outer_cv)
        nested_scores_vit_corr_fuse[i] = nested_score_vit_corr_fuse.mean()

        # Nested CV with parameter optimization
        nested_score_vit_dtw_fuse = cross_val_score(param_search, X=vit_dtw_fuse, y=class_vit, cv=outer_cv)
        nested_scores_vit_dtw_fuse[i] = nested_score_vit_dtw_fuse.mean()

        nested_score_flour_corr_fuse = cross_val_score(param_search, X=flour_corr_fuse, y=class_flour, cv=outer_cv)
        nested_scores_flour_corr_fuse[i] = nested_score_flour_corr_fuse.mean()
        
        nested_score_flour_dtw_fuse= cross_val_score(param_search, X=flour_dtw_fuse, y=class_flour, cv=outer_cv)
        nested_scores_flour_dtw_fuse[i] = nested_score_flour_dtw_fuse.mean()

        # Nested CV Prediction
        prob_vit_corr_fuse = cross_val_predict(param_search, X=vit_corr_fuse, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_vit_dtw_fuse = cross_val_predict(param_search, X=vit_dtw_fuse, y=class_vit, cv=outer_cv,method='predict_proba')

        prob_flour_corr_fuse = cross_val_predict(param_search, X=flour_corr_fuse, y=class_flour, cv=outer_cv,method='predict_proba')
        
        prob_flour_dtw_fuse = cross_val_predict(param_search, X=flour_dtw_fuse, y=class_flour, cv=outer_cv,method='predict_proba')
        
        
        fprs_vit_corr_fuse,tprs_vit_corr_fuse,_=roc_curve(class_vit,prob_vit_corr_fuse[:,1])
        fprs_vit_dtw_fuse,tprs_vit_dtw_fuse,_=roc_curve(class_vit,prob_vit_dtw_fuse[:,1])
        fprs_flour_corr_fuse,tprs_flour_corr_fuse,_=roc_curve(class_flour,prob_flour_corr_fuse[:,1])
        fprs_flour_dtw_fuse,tprs_flour_dtw_fuse,_=roc_curve(class_flour,prob_flour_dtw_fuse[:,1])
        
        
        tpr_vit_corr_fuse=np.interp(mean_fpr,fprs_vit_corr_fuse,tprs_vit_corr_fuse)
        tpr_vit_corr_fuse[0] =0.0

        tpr_vit_dtw_fuse=np.interp(mean_fpr,fprs_vit_dtw_fuse,tprs_vit_dtw_fuse)
        tpr_vit_dtw_fuse[0] =0.0
        
        tpr_flour_corr_fuse=np.interp(mean_fpr,fprs_flour_corr_fuse,tprs_flour_corr_fuse)
        tpr_flour_corr_fuse[0] =0.0       

        tpr_flour_dtw_fuse=np.interp(mean_fpr,fprs_flour_dtw_fuse,tprs_flour_dtw_fuse)
        tpr_flour_dtw_fuse[0] =0.0              
        
        all_fprs_vit_corr_fuse.append(fprs_vit_corr_fuse)
        all_fprs_vit_dtw_fuse.append(fprs_vit_dtw_fuse)
        all_fprs_flour_corr_fuse.append(fprs_flour_corr_fuse)
        all_fprs_flour_dtw_fuse.append(fprs_flour_dtw_fuse)
        
        all_tprs_vit_corr_fuse.append(tpr_vit_corr_fuse)
        all_tprs_vit_dtw_fuse.append(tpr_vit_dtw_fuse)
        all_tprs_flour_corr_fuse.append(tpr_flour_corr_fuse)
        all_tprs_flour_dtw_fuse.append(tpr_flour_dtw_fuse)
        
    
    m_tpr = np.mean(all_tprs_vit_corr_fuse,axis=0)
    std_tpr = np.std(all_tprs_vit_corr_fuse,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_corr_fuse.append(m_tpr)
    std_tpr_vit_corr_fuse.append(std_tpr)
    mean_fpr_vit_corr_fuse.append(mean_fpr)
    mean_auc_vit_corr_fuse.append(auc(mean_fpr,m_tpr))
    
    m_tpr = np.mean(all_tprs_vit_dtw_fuse,axis=0)
    std_tpr = np.std(all_tprs_vit_dtw_fuse,axis=0)
    m_tpr [-1] =1.0
    
    mean_tpr_vit_dtw_fuse.append(m_tpr)
    std_tpr_vit_dtw_fuse.append(std_tpr)
    mean_fpr_vit_dtw_fuse.append(mean_fpr)
    mean_auc_vit_dtw_fuse.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_corr_fuse,axis=0)
    std_tpr = np.std(all_tprs_flour_corr_fuse,axis=0)
    m_tpr [-1] =1.0   
    
    mean_tpr_flour_corr_fuse.append(m_tpr)
    std_tpr_flour_corr_fuse.append(std_tpr)
    mean_fpr_flour_corr_fuse.append(mean_fpr)
    mean_auc_flour_corr_fuse.append(auc(mean_fpr,m_tpr))
    
    
    m_tpr = np.mean(all_tprs_flour_dtw_fuse,axis=0)
    std_tpr = np.std(all_tprs_flour_dtw_fuse,axis=0)
    m_tpr [-1] =1.0   
    
    mean_tpr_flour_dtw_fuse.append(m_tpr)
    std_tpr_flour_dtw_fuse.append(std_tpr)
    mean_fpr_flour_dtw_fuse.append(mean_fpr)
    mean_auc_flour_dtw_fuse.append(auc(mean_fpr,m_tpr))

    all_fprs_vit_corr_fuse =[]
    all_fprs_vit_dtw_fuse =[]
    all_fprs_flour_corr_fuse =[]
    all_fprs_flour_dtw_fuse =[]

    all_tprs_vit_corr_fuse =[]
    all_tprs_vit_dtw_fuse =[]
    all_tprs_flour_corr_fuse =[]
    all_tprs_flour_dtw_fuse =[]
    
        
    all_nested_scores_flour_corr_fuse.append(nested_scores_flour_corr_fuse)
    all_nested_scores_vit_corr_fuse.append(nested_scores_vit_corr_fuse)
    all_nested_scores_flour_dtw_fuse.append(nested_scores_flour_dtw_fuse)
    all_nested_scores_vit_dtw_fuse.append(nested_scores_vit_dtw_fuse)
    
    
    print("Vital_corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_corr_fuse.mean(), nested_scores_vit_corr_fuse.std()))
    all_vit_corr_acc_mean_fuse.append(nested_scores_vit_corr_fuse.mean())
    all_vit_corr_acc_std_fuse.append(nested_scores_vit_corr_fuse.std())
    
    print("Vital_DTW: Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_dtw_fuse.mean(), nested_scores_vit_dtw_fuse.std()))
    all_vit_dtw_acc_mean_fuse.append(nested_scores_vit_dtw_fuse.mean())
    all_vit_dtw_acc_std_fuse.append(nested_scores_vit_dtw_fuse.std())

    print("Flour_Corr : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_corr_fuse.mean(), nested_scores_flour_corr_fuse.std()))
    all_flour_corr_acc_mean_fuse.append(nested_scores_flour_corr_fuse.mean())
    all_flour_corr_acc_std_fuse.append(nested_scores_flour_corr_fuse.std())

    print("Flour_DTW : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_dtw_fuse.mean(), nested_scores_flour_dtw_fuse.std()))
    all_flour_dtw_acc_mean_fuse.append(nested_scores_flour_dtw_fuse.mean())
    all_flour_dtw_acc_std_fuse.append(nested_scores_flour_dtw_fuse.std())


# Flour DTW dxy 
All_Vec = dtw_dxy
Feat_Vec = flour_dtw_dxy
reg_flour_dtw_dxy=[]
ind_flour_dtw_dxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_flour_dtw_dxy.append(j)
            reg_flour_dtw_dxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))
       
# Flour DTW oxy 
All_Vec = dtw_oxy
Feat_Vec = flour_dtw_oxy
reg_flour_dtw_oxy=[]
ind_flour_dtw_oxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_flour_dtw_oxy.append(j)
            reg_flour_dtw_oxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))
            
# Flour corr dxy 
All_Vec = corr_dxy
Feat_Vec = flour_corr_dxy
reg_flour_corr_dxy=[]
ind_flour_corr_dxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_flour_corr_dxy.append(j)
            reg_flour_corr_dxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))

# Flour corr oxy 

All_Vec = corr_oxy
Feat_Vec = flour_corr_oxy
reg_flour_corr_oxy=[]
ind_flour_corr_oxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_flour_corr_oxy.append(j)
            reg_flour_corr_oxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))
                        
# Vital DTW dxy     
            
All_Vec = dtw_dxy
Feat_Vec = vit_dtw_dxy
reg_vit_dtw_dxy=[]
ind_vit_dtw_dxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_vit_dtw_dxy.append(j)
            reg_vit_dtw_dxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))
       

# Vital DTW oxy        

All_Vec = dtw_oxy
Feat_Vec = vit_dtw_oxy
reg_vit_dtw_oxy=[]
ind_vit_dtw_oxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_vit_dtw_oxy.append(j)
            reg_vit_dtw_oxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))
     
# Vital corr dxy

All_Vec = corr_dxy
Feat_Vec = vit_corr_dxy
reg_vit_corr_dxy=[]
ind_vit_corr_dxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_vit_corr_dxy.append(j)
            reg_vit_corr_dxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))

# Vital corr oxy

All_Vec = corr_oxy
Feat_Vec = vit_corr_oxy
reg_vit_corr_oxy=[]
ind_vit_corr_oxy=[]
_,dim_arr = np.shape(All_Vec)
_,dim_feat = np.shape(Feat_Vec)

for j in range(dim_arr):
    for i in range(0,dim_feat):
        if sum(Feat_Vec[:,i]-All_Vec[:,j])==0:
            ind_vit_corr_oxy.append(j)
            reg_vit_corr_oxy.append(''.join(Reg_Info[j][0]))
            print(''.join(Reg_Info[j][0]))    
            
all_t_flour_corr_ox=[]
all_p_flour_corr_ox=[]
all_t_flour_dtw_ox=[]
all_p_flour_dtw_ox=[]
all_t_vital_corr_ox=[]
all_p_vital_corr_ox=[]
all_t_vital_dtw_ox=[]
all_p_vital_dtw_ox=[]

all_t_flour_corr_dx=[]
all_p_flour_corr_dx=[]
all_t_flour_dtw_dx=[]
all_p_flour_dtw_dx=[]
all_t_vital_corr_dx=[]
all_p_vital_corr_dx=[]
all_t_vital_dtw_dx=[]
all_p_vital_dtw_dx=[]

for i in range(0,9):
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_flour_corr_oxy[i],all_nested_scores_flour_corr_fuse[i])
    all_t_flour_corr_ox.append(t)
    all_p_flour_corr_ox.append(p)
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_flour_dtw_oxy[i],all_nested_scores_flour_dtw_fuse[i])
    all_t_flour_dtw_ox.append(t)
    all_p_flour_dtw_ox.append(p)    
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_vit_corr_oxy[i],all_nested_scores_vit_corr_fuse[i])
    all_t_vital_corr_ox.append(t)
    all_p_vital_corr_ox.append(p)
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_vit_dtw_oxy[i],all_nested_scores_vit_dtw_fuse[i])
    all_t_vital_dtw_ox.append(t)
    all_p_vital_dtw_ox.append(p)        
    
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_flour_corr_dxy[i],all_nested_scores_flour_corr_fuse[i])
    all_t_flour_corr_dx.append(t)
    all_p_flour_corr_dx.append(p)
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_flour_dtw_dxy[i],all_nested_scores_flour_dtw_fuse[i])
    all_t_flour_dtw_dx.append(t)
    all_p_flour_dtw_dx.append(p)    
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_vit_corr_dxy[i],all_nested_scores_vit_corr_fuse[i])
    all_t_vital_corr_dx.append(t)
    all_p_vital_corr_dx.append(p)
    t,p=sp.stats.mstats.ttest_ind(all_nested_scores_vit_dtw_dxy[i],all_nested_scores_vit_dtw_fuse[i])
    all_t_vital_dtw_dx.append(t)
    all_p_vital_dtw_dx.append(p)         
    
chance = np.linspace(0,1,100)
fig, axs = plt.subplots(3, 3,figsize=(100, 50),sharex=False, sharey=False,constrained_layout=True)
fig.suptitle('ROC Curves of Classification of Highly Flourishing Individuals',fontsize=100,verticalalignment='bottom',fontweight='bold')
alph=['a)','b)','c)','d)','e)','f)','g)','h)','i)']
k=0
fs=50
fs2=70
lw=10
for i in range(0,3):
    for j in range(0,3):
        legends =['CC-ΔHbO','CC-ΔHb','CC-Fuse','DTW-ΔHbO','DTW-ΔHb','DTW-Fuse']
        axs[i,j].plot(mean_fpr_flour_corr_oxy[k],mean_tpr_flour_corr_oxy[k],linewidth=lw)#,yerr = std_tpr_flour_corr_oxy[k])
        axs[i,j].plot(mean_fpr_flour_corr_dxy[k],mean_tpr_flour_corr_dxy[k],linewidth=lw)#,yerr = std_tpr_flour_corr_dxy[k])
        axs[i,j].plot(mean_fpr_flour_corr_fuse[k],mean_tpr_flour_corr_fuse[k],linewidth=lw)#,yerr = std_tpr_flour_corr_fuse[k])
        axs[i,j].plot(mean_fpr_flour_dtw_oxy[k],mean_tpr_flour_dtw_oxy[k],linewidth=lw)#,yerr = std_tpr_flour_dtw_oxy[k])
        axs[i,j].plot(mean_fpr_flour_dtw_dxy[k],mean_tpr_flour_dtw_dxy[k],linewidth=lw)#,yerr = std_tpr_flour_dtw_dxy[k])
        axs[i,j].plot(mean_fpr_flour_dtw_fuse[k],mean_tpr_flour_dtw_fuse[k],linewidth=lw)#,yerr = std_tpr_flour_dtw_fuse[k])
        axs[i,j].plot(chance,chance,linewidth=lw)
        auc = [mean_auc_flour_corr_oxy[k],mean_auc_flour_corr_dxy[k],mean_auc_flour_corr_fuse[k],
               mean_auc_flour_dtw_oxy[k],mean_auc_flour_dtw_dxy[k],mean_auc_flour_dtw_fuse[k]]
        axs[i,j].set_xlabel('False Positive Rate (FPR)'  '\n' + alph[k] + '\n' ,fontsize=fs2,fontweight='bold')
        axs[i,j].set_ylabel('True Positive Rate (TPR)',fontsize=fs2,fontweight='bold')
        plt.setp(axs[i,j].get_xticklabels(), fontsize=fs2, fontweight="bold")
        plt.setp(axs[i,j].get_yticklabels(), fontsize=fs2, fontweight="bold")
        axs[i,j].set_title('ROC Curve -' + names[k],fontsize=fs2,fontweight='bold')
        for s in range(0,6):
            legends[s]=legends[s] +', AUC:'+ str("{:.2f}".format(auc[s]))
        legends.append('Chance Level')
        axs[i,j].legend(legends,fontsize=40)
        legends=[]
        k=k+1

fig, axs = plt.subplots(3, 3,figsize=(25, 25),sharex=False, sharey=False)
fig.suptitle('ROC Curves of Classification of Highly Vital Individuals',fontsize=30,verticalalignment='bottom',fontweight='bold')
k=0
for i in range(0,3):
    for j in range(0,3):
        legends =['CC-ΔHbO','CC-ΔHb','CC-Fuse','DTW-ΔHbO','DTW-ΔHb','DTW-Fuse']
        axs[i,j].plot(mean_fpr_vit_corr_oxy[k],mean_tpr_vit_corr_oxy[k])#, yerr=std_tpr_vit_corr_oxy[k])
        axs[i,j].plot(mean_fpr_vit_corr_dxy[k],mean_tpr_vit_corr_dxy[k])#, yerr=std_tpr_vit_corr_dxy[k])
        axs[i,j].plot(mean_fpr_vit_corr_fuse[k],mean_tpr_vit_corr_fuse[k])#, yerr =std_tpr_vit_corr_fuse[k] )
        axs[i,j].plot(mean_fpr_vit_dtw_oxy[k],mean_tpr_vit_dtw_oxy[k])#, yerr = std_tpr_vit_dtw_oxy[k])
        axs[i,j].plot(mean_fpr_vit_dtw_dxy[k],mean_tpr_vit_dtw_dxy[k])#, yerr = std_tpr_vit_dtw_dxy[k])
        axs[i,j].plot(mean_fpr_vit_dtw_fuse[k],mean_tpr_vit_dtw_fuse[k])#, yerr= std_tpr_vit_dtw_fuse[k])
        axs[i,j].plot(chance,chance)
        auc = [mean_auc_vit_corr_oxy[k],mean_auc_vit_corr_dxy[k],mean_auc_vit_corr_fuse[k],
               mean_auc_vit_dtw_oxy[k],mean_auc_vit_dtw_dxy[k],mean_auc_vit_dtw_fuse[k]]
        axs[i,j].set_xlabel('False Positive Rate (FPR)' '\n' + alph[k] + '\n' ,fontsize=fs,fontweight='bold')
        axs[i,j].set_ylabel('True Positive Rate (TPR)',fontsize=fs,fontweight='bold')
        axs[i,j].set_title('ROC Curve -' + names[k],fontsize=fs,fontweight='bold')
        plt.setp(axs[i,j].get_xticklabels(), fontsize=16, fontweight="bold")
        plt.setp(axs[i,j].get_yticklabels(), fontsize=16, fontweight="bold")
        for s in range(0,6):
            legends[s]=legends[s] +', AUC:'+ str("{:.2f}".format(auc[s]))
        legends.append('Chance Level')
        axs[i,j].legend(legends,fontsize=fs)
        legends=[]
        k=k+1

## ---- Violin Plots -------#####

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels,fontsize=70)
    ax.set_xlim(0.25, len(labels) + 0.75)
    #ax.set_xlabel('Feature Sets')


fig, axs = plt.subplots(3, 3,figsize=(100, 50),sharex=False, sharey=False,constrained_layout=True)
fig.suptitle('Flourishing Classification Accuracy Plots',fontsize=100,verticalalignment='center_baseline',fontweight='bold')
k=0
fs=50
fs2=80
lw=20
for i in range(0,3):
    for j in range(0,3):
        legends =['CC' '\n' 'ΔHbO','CC' '\n' 'ΔHb','CC' '\n' 'Fuse','DTW' '\n' 'ΔHbO','DTW' '\n' 'ΔHb','DTW' '\n' 'Fuse']
        
        data = [all_nested_scores_flour_corr_oxy[k],all_nested_scores_flour_corr_dxy[k],all_nested_scores_flour_corr_fuse[k],all_nested_scores_flour_dtw_oxy[k],
                all_nested_scores_flour_dtw_dxy[k],all_nested_scores_flour_dtw_fuse[k]]
        
        pp=axs[i,j].violinplot(data,showmeans=True,showextrema=True,vert=True, widths=0.8)
        for pc in pp['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('red')
            pc.set_linewidth(lw)
        pp['cmeans'].set_color('k')
        pp['cmeans'].set_linewidth(lw)
        pp['cmaxes'].set_color('k')
        pp['cmaxes'].set_linewidth(lw)
        pp['cmins'].set_color('k')
        pp['cmins'].set_linewidth(lw)
        pp['cbars'].set_color('k')
        pp['cbars'].set_linewidth(lw)
        plt.setp(axs[i,j].get_xticklabels(), fontsize=fs2, fontweight="bold")
        plt.setp(axs[i,j].get_yticklabels(), fontsize=fs2, fontweight="bold")
        #axs[i,j].set_xlabel('Feature Set' '\n' '\n' + alph[k] + '\n' ,fontsize=fs,fontweight='bold')
        axs[i,j].set_ylabel('Accuracy',fontsize=fs2,fontweight='bold')
        axs[i,j].set_title(names[k],fontsize=fs2,fontweight='bold')
        set_axis_style(axs[i,j], legends)
        legends=[]
        k=k+1


fig, axs = plt.subplots(3, 3,figsize=(25, 25),sharex=False, sharey=False)
fig.suptitle('Vital Classification Accuracy Plots',fontsize=24,verticalalignment='center_baseline',fontweight='bold')
k=0
for i in range(0,3):
    for j in range(0,3):
        legends =['CC' '\n' 'ΔHbO','CC' '\n' 'ΔHb','CC' '\n' 'Fuse','DTW' '\n' 'ΔHbO','DTW' '\n' 'ΔHb','DTW' '\n' 'Fuse']
        
        data = [all_nested_scores_vit_corr_oxy[k],all_nested_scores_vit_corr_dxy[k],all_nested_scores_vit_corr_fuse[k],all_nested_scores_vit_dtw_oxy[k],
                all_nested_scores_vit_dtw_dxy[k],all_nested_scores_vit_dtw_fuse[k]]
        
        axs[i,j].violinplot(data,showmeans=True,showextrema=True,vert=True)
        plt.setp(axs[i,j].get_xticklabels(), fontsize=12, fontweight="bold")
        plt.setp(axs[i,j].get_yticklabels(), fontsize=12, fontweight="bold")
        axs[i,j].set_xlabel('Feature Set' '\n' '\n' + alph[k] + '\n' ,fontsize=14,fontweight='bold')
        axs[i,j].set_ylabel('Accuracy',fontsize=14,fontweight='bold')
        axs[i,j].set_title(names[k],fontsize=14,fontweight='bold')
        set_axis_style(axs[i,j], legends)
        legends=[]
        k=k+1



df = pd.read_excel('resting_state.xlsx')

Vit_score = df['Vit_score'].tolist()
Flour_score = df['Flour_score'].tolist()
Trai_rumin_rss = df['Trai_rumin_RSS'].tolist()
Trait_mind_maas = df['Trait_mind_MAAS'].tolist()
State_Rumin = df['State_Rumination'].tolist()
State_Mind = df['State_MindWandering'].tolist()


print('#### -------- FLOURISHING -------####')

for i in range(0,np.shape(corr_oxy_new_flour)[1]):
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,i],Flour_score)
    if p<0.001:
        print('Flour_score & Corr_oxy:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Trai_rumin_rss & Corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,i],Trait_mind_maas)
    if p<0.001:
        print('Trait_mind_maas & Corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,i],State_Rumin)
    if p<0.001:
        print('State_rumin & Corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,i],State_Mind)
    if p<0.001:
        print('State_Mind & Corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        
for i in range(0,np.shape(corr_dxy_new_flour)[1]):
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_flour[:,i],Flour_score)
    if p<0.001:
        print('Flour_score & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_flour[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Trai_rumin_rss & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_flour[:,i],Trait_mind_maas)
    if p<0.001:
        print('Trait_mind_maas & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_flour[:,i],State_Rumin)
    if p<0.001:
        print('State_rumin & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_flour[:,i],State_Mind)
    if p<0.001:
        print('State_Mind & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        

for i in range(0,np.shape(dtw_oxy_new_flour)[1]):
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_flour[:,i],Flour_score)
    if p<0.001:
        print('Flour_score & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_flour[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Trai_rumin_rss & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_flour[:,i],Trait_mind_maas)
    if p<0.001:
        print('Trait_mind_maas & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_flour[:,i],State_Rumin)
    if p<0.001:
        print('State_rumin & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_flour[:,i],State_Mind)
    if p<0.001:
        print('State_Mind & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        
    
for i in range(0,np.shape(dtw_dxy_new_flour)[1]):
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,i],Flour_score)
    if p<0.001:
        print('Flour_score & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Trai_rumin_rss & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,i],Trait_mind_maas)
    if p<0.001:
        print('trait_mind_maas & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,i],State_Rumin)
    if p<0.001:
        print('State_Rumin & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,i],State_Mind)
    if p<0.001:
        print('State_Mind & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

fig,axs = plt.subplots(2,1,figsize =(20,15),sharex=False, sharey=False,constrained_layout=True)
fig.suptitle('Correlation Between Flourishing Score and Connections',fontsize=24,fontweight='bold')

[slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_flour[:,1],Flour_score)
sns.regplot(Flour_score,corr_oxy_new_flour[:,1],ax=axs[0],color='k',scatter_kws={"s": 200})
axs[0].set_xlabel('Flourishing Score \n a)',fontsize=25,fontweight='bold')
axs[0].set_ylabel('R V3 - R SAC, '+' CC - ΔHbO',fontsize=25,fontweight='bold')

[slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_flour[:,3],Flour_score)
sns.regplot(Flour_score,dtw_dxy_new_flour[:,3],ax=axs[1],color='k',scatter_kws={"s": 200})
axs[1].set_xlabel('Flourishing Score \n b)',fontsize=25,fontweight='bold')
axs[1].set_ylabel('L V3 - L SAC, '+'DTW - ΔHb',fontsize=25,fontweight='bold')

### ------------VITAL --------- #####
print('### ------------VITAL --------- #####')
for i in range(0,np.shape(corr_oxy_new_vit)[1]):
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_vit[:,i],Vit_score)
    if p<0.001:
        print('Vit_score & corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_vit[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Vit_Trai_rumin_rss & corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_vit[:,i],Trait_mind_maas)
    if p<0.001:
        print('Vit_Trait_mind_maas & corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_vit[:,i],State_Rumin)
    if p<0.001:
        print('Vit_State_Rumin & corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_oxy_new_vit[:,i],State_Mind)
    if p<0.001:
        print('Vit_State_Mind & Corr_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        
    
for i in range(0,np.shape(corr_dxy_new_vit)[1]):
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_vit[:,i],Vit_score)
    if p<0.001:
        print('Vit_score & corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_vit[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Vit_Trai_rumin_rss & corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_vit[:,i],Trait_mind_maas)
    if p<0.001:
        print('Vit_Trait_mind_maas & corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_vit[:,i],State_Rumin)
    if p<0.001:
        print('Vit_State_Rumin & corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(corr_dxy_new_vit[:,i],State_Mind)
    if p<0.001:
        print('Vit_State_Mind & Corr_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        

for i in range(0,np.shape(dtw_oxy_new_vit)[1]):
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_vit[:,i],Vit_score)
    if p<0.001:
        print('Vit_score & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_vit[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Vit_Trai_rumin_rss & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_vit[:,i],Trait_mind_maas)
    if p<0.001:
        print('Vit_Trait_mind_maas & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_vit[:,i],State_Rumin)
    if p<0.001:
        print('Vit_State_Rumin & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_oxy_new_vit[:,i],State_Mind)
    if p<0.001:
        print('Vit_State_Mind & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        
    
for i in range(0,np.shape(dtw_dxy_new_vit)[1]):
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_vit[:,i],Vit_score)
    if p<0.001:
        print('Vit_score & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_vit[:,i],Trai_rumin_rss)
    if p<0.001:
        print('Vit_Trai_rumin_rss & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_vit[:,i],Trait_mind_maas)
    if p<0.001:
        print('Vit_Trait_mind_maas & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_vit[:,i],State_Rumin)
    if p<0.001:
        print('Vit_State_Rumin & dtw_dxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
    [slope,intercept,r,p,err]=stats.linregress(dtw_dxy_new_vit[:,i],State_Mind)
    if p<0.001:
        print('Vit_State_Mind & dtw_oxy feat number:' +str(i))
        print('rr:' +str(r))
        print('pval:'+str(p))
        print('--------******---------')
        

fig,axs = plt.subplots(3,1,figsize =(10,20),sharex=False, sharey=False)
fig.suptitle('Correlation Between Vital Score and Connections',fontsize=24,fontweight='bold')

sns.regplot(Vit_score,dtw_oxy_new_vit[:,2],ax=axs[0])
sns.regplot(Vit_score,dtw_dxy_new_vit[:,2],ax=axs[1])
sns.regplot(Vit_score,dtw_dxy_new_vit[:,4],ax=axs[2])
axs[0].set_xlabel('Vital Score',fontsize=14,fontweight='bold')
axs[1].set_xlabel('Vital Score',fontsize=14,fontweight='bold')
axs[2].set_xlabel('Vital Score',fontsize=14,fontweight='bold')

axs[0].set_ylabel(reg_vit_dtw_oxy[2]+' conn. using DTW',fontsize=14,fontweight='bold')
axs[1].set_ylabel(reg_vit_dtw_dxy[2]+' conn. using DTW',fontsize=14,fontweight='bold')
axs[2].set_ylabel(reg_vit_dtw_dxy[4]+' conn. using DTW',fontsize=14,fontweight='bold')

print('---------%%% For Level 2-Fused (CC-Oxy + DTW-Deoxy for flourishing & CC-Deoxy + DTW-Deoxy for vital) Features %%%-----------')
vit_cc_dtw_fuse= np.concatenate((corr_oxy_new_vit,dtw_dxy_new_vit),axis=1)
flour_cc_dtw_fuse= np.concatenate((corr_oxy_new_flour, dtw_dxy_new_flour),axis=1)


# vit_ccoxy_dtwdeoxy_fuse = LinearSVC(C=0.1, penalty="l1", dual=False).fit(vit_cc_dtw_fuse, class_vit)
# vit_ccoxy_dtwdeoxy_fuse = SelectFromModel(vit_ccoxy_dtwdeoxy_fuse, prefit=True)
# vit_ccoxy_dtwdeoxy_fuse = vit_ccoxy_dtwdeoxy_fuse.transform(vit_cc_dtw_fuse)

# flour_ccoxy_dtwdeoxy_fuse = LinearSVC(C=0.1, penalty="l1", dual=False).fit(flour_cc_dtw_fuse, class_flour)
# flour_ccoxy_dtwdeoxy_fuse = SelectFromModel(flour_ccoxy_dtwdeoxy_fuse, prefit=True)
# flour_ccoxy_dtwdeoxy_fuse = flour_ccoxy_dtwdeoxy_fuse.transform(flour_cc_dtw_fuse)

vit_ccoxy_dtw_deoxy_fuse = vit_cc_dtw_fuse
flour_ccoxy_dtwdeoxy_fuse = flour_cc_dtw_fuse

all_vit_acc_mean_ccoxy_dtwdeoxy_fuse=[]
all_flour_acc_mean_ccoxy_dtwdeoxy_fuse=[]
all_vit_acc_std_ccoxy_dtwdeoxy_fuse=[]
all_flour_acc_std_ccoxy_dtwdeoxy_fuse=[]


all_fprs_vit_ccoxy_dtwdeoxy_fuse=[]
all_tprs_vit_ccoxy_dtwdeoxy_fuse=[]
all_fprs_flour_ccoxy_dtwdeoxy_fuse=[]
all_tprs_flour_ccoxy_dtwdeoxy_fuse=[]
all_vit_corr_acc_mean_ccoxy_dtwdeoxy=[]
all_vit_corr_acc_std_ccoxy_dtwdeoxy=[]
all_flour_corr_acc_mean_ccoxy_dtwdeoxy=[]
all_flour_corr_acc_std_ccoxy_dtwdeoxy=[]
all_nested_scores_flour_corrdtw_fuse = []
all_nested_scores_vit_corrdtw_fuse =[]

mean_tpr_vit_ccoxy_dtwdeoxy_fuse =[]
mean_tpr_flour_ccoxy_dtwdeoxy_fuse =[]
mean_fpr_vit_ccoxy_dtwdeoxy_fuse =[]
mean_fpr_flour_ccoxy_dtwdeoxy_fuse =[]

std_tpr_vit_ccoxy_dtwdeoxy_fuse =[]
std_fpr_vit_ccoxy_dtwdeoxy_fuse =[]
std_tpr_flour_ccoxy_dtwdeoxy_fuse =[]
std_fpr_flour_ccoxy_dtwdeoxy_fuse =[]

mean_auc_vit_ccoxy_dtwdeoxy_fuse =[]
mean_auc_flour_ccoxy_dtwdeoxy_fuse =[]
mean_fpr = np.linspace(0, 1, 100)

for name, classifier in zip(names, classifiers):
    if name in ('Linear SVM', 'RBF SVM',
                'Gradient Boosting', 'AdaBoost', 'Logistic Regression',
                ):
        verbose = 0
    else:
        verbose = 0
    print(name)
    time.sleep(0.1)
    NUM_TRIALS = 30
    nested_scores_vit_ccoxy_dtwdeoxy_fuse = np.zeros(NUM_TRIALS)
    nested_scores_flour_ccoxy_dtwdeoxy_fuse= np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        param_search = GridSearchCV(
            estimator=classifier, param_grid=params_classifiers[name],
            verbose=verbose, cv =inner_cv)

        # Nested CV with parameter optimization
        nested_score_vit_cccoxy_dtwdeoxy_fuse = cross_val_score(param_search, X=vit_ccoxy_dtw_deoxy_fuse, y=class_vit, cv=outer_cv)
        nested_scores_vit_ccoxy_dtwdeoxy_fuse[i] = nested_score_vit_cccoxy_dtwdeoxy_fuse.mean()

        # Nested CV with parameter optimization

        nested_score_flour_ccoxy_dtwdeoxy_fuse = cross_val_score(param_search, X=flour_ccoxy_dtwdeoxy_fuse, y=class_flour, cv=outer_cv)
        nested_scores_flour_ccoxy_dtwdeoxy_fuse[i] = nested_score_flour_ccoxy_dtwdeoxy_fuse.mean()
        

        # Nested CV Prediction
        prob_vit_ccoxy_dtwdeoxy_fuse = cross_val_predict(param_search, X=vit_ccoxy_dtw_deoxy_fuse, y=class_vit, cv=outer_cv,method='predict_proba')
        
        prob_flour_ccoxy_dtwdeoxy_fuse= cross_val_predict(param_search, X=flour_ccoxy_dtwdeoxy_fuse, y=class_flour, cv=outer_cv,method='predict_proba')
        
        
        fprs_vit_ccoxy_dtwdeoxy_fuse,tprs_vit_ccoxy_dtwdeoxy_fuse,_=roc_curve(class_vit,prob_vit_ccoxy_dtwdeoxy_fuse[:,1])
        fprs_flour_ccoxy_dtwdeoxy_fuse,tprs_flour_ccoxy_dtwdeoxy_fuse,_=roc_curve(class_flour,prob_flour_ccoxy_dtwdeoxy_fuse[:,1])
        
        
        tpr_vit_ccoxy_dtwdeoxy_fuse=np.interp(mean_fpr,fprs_vit_ccoxy_dtwdeoxy_fuse,tprs_vit_ccoxy_dtwdeoxy_fuse)
        tpr_vit_ccoxy_dtwdeoxy_fuse[0] =0.0


        tpr_flour_ccoxy_dtwdeoxy_fuse=np.interp(mean_fpr,fprs_flour_ccoxy_dtwdeoxy_fuse,tprs_flour_ccoxy_dtwdeoxy_fuse)
        tpr_flour_ccoxy_dtwdeoxy_fuse[0] =0.0              
        
        all_fprs_vit_ccoxy_dtwdeoxy_fuse.append(fprs_vit_ccoxy_dtwdeoxy_fuse)
        all_fprs_flour_ccoxy_dtwdeoxy_fuse.append(fprs_flour_ccoxy_dtwdeoxy_fuse)

        all_tprs_vit_ccoxy_dtwdeoxy_fuse.append(tpr_vit_ccoxy_dtwdeoxy_fuse)
        all_tprs_flour_ccoxy_dtwdeoxy_fuse.append(tpr_flour_ccoxy_dtwdeoxy_fuse)
        
    
    m_tpr = np.mean(all_tprs_vit_ccoxy_dtwdeoxy_fuse,axis=0)
    std_tpr = np.std(all_tprs_vit_ccoxy_dtwdeoxy_fuse,axis=0)
    m_tpr [-1] =1
    
    mean_tpr_vit_ccoxy_dtwdeoxy_fuse.append(m_tpr)
    std_tpr_vit_ccoxy_dtwdeoxy_fuse.append(std_tpr)
    mean_fpr_vit_ccoxy_dtwdeoxy_fuse.append(mean_fpr)
    mean_auc_vit_ccoxy_dtwdeoxy_fuse.append(auc(mean_fpr,m_tpr))
    
    m_tpr = np.mean(all_tprs_flour_ccoxy_dtwdeoxy_fuse,axis=0)
    std_tpr = np.std(all_tprs_flour_ccoxy_dtwdeoxy_fuse,axis=0)
    m_tpr [-1] =1
    
    mean_tpr_flour_ccoxy_dtwdeoxy_fuse.append(m_tpr)
    std_tpr_flour_ccoxy_dtwdeoxy_fuse.append(std_tpr)
    mean_fpr_flour_ccoxy_dtwdeoxy_fuse.append(mean_fpr)
    mean_auc_flour_ccoxy_dtwdeoxy_fuse.append(auc(mean_fpr,m_tpr))
    
    
    all_fprs_flour_ccoxy_dtwdeoxy_fuse=[]
    all_fprs_vit_ccoxy_dtwdeoxy_fuse =[]

    all_tprs_flour_ccoxy_dtwdeoxy_fuse =[]
    all_tprs_vit_ccoxy_dtwdeoxy_fuse =[]
    
        
    all_nested_scores_flour_corrdtw_fuse.append(nested_scores_flour_ccoxy_dtwdeoxy_fuse)
    all_nested_scores_vit_corrdtw_fuse.append(nested_scores_vit_ccoxy_dtwdeoxy_fuse)
    
    
    print("Vital_corrOxy_DTW_Deoxy : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_vit_ccoxy_dtwdeoxy_fuse.mean(), nested_scores_vit_ccoxy_dtwdeoxy_fuse.std()))
    all_vit_acc_mean_ccoxy_dtwdeoxy_fuse.append(nested_scores_vit_ccoxy_dtwdeoxy_fuse.mean())
    all_vit_acc_std_ccoxy_dtwdeoxy_fuse.append(nested_scores_vit_ccoxy_dtwdeoxy_fuse.std())
    

    print("Flour_corrOxy_DTW_Deoxy : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_ccoxy_dtwdeoxy_fuse.mean(), nested_scores_flour_ccoxy_dtwdeoxy_fuse.std()))
    all_flour_acc_mean_ccoxy_dtwdeoxy_fuse.append(nested_scores_flour_ccoxy_dtwdeoxy_fuse.mean())
    all_flour_acc_std_ccoxy_dtwdeoxy_fuse.append(nested_scores_flour_ccoxy_dtwdeoxy_fuse.std())
    
    


chance = np.linspace(0,1,100)
fig, axs = plt.subplots(1, 1,figsize=(25, 25),sharex=False, sharey=False,constrained_layout=True)
#fig.suptitle('ROC Curves of Classification of ..',fontsize=24,verticalalignment='bottom',fontweight='bold')
fs=38
legends =['Nearest Neighbor','Linear SVM','RBF SVM','Gradient Boosting','AdaBoost','Naive Bayes','Linear Discriminant Analysis','Quadratic Discriminant Analysis','Logistic Regression']

for k in range(0,9):
    axs.plot(mean_fpr_flour_ccoxy_dtwdeoxy_fuse[k],mean_tpr_flour_ccoxy_dtwdeoxy_fuse[k],linewidth=8)#,yerr = std_tpr_flour_corr_oxy[k])
    auc_fuse = [mean_auc_flour_ccoxy_dtwdeoxy_fuse[k]]
    legends[k]=legends[k] +', AUC:'+ str("{:.2f}".format(auc_fuse[0]))
    
axs.plot(chance,chance,linewidth=8)   
axs.set_xlabel('False Positive Rate (FPR)',fontsize=fs,fontweight='bold')
axs.set_ylabel('True Positive Rate (TPR)',fontsize=fs,fontweight='bold')
plt.setp(axs.get_xticklabels(), fontsize=fs, fontweight="bold")
plt.setp(axs.get_yticklabels(), fontsize=fs, fontweight="bold")
axs.set_title('ROC Curve of CC-ΔHbO & DTW-ΔHb Fusion',fontsize=fs,fontweight='bold')
legends.append('Chance Level')
axs.legend(legends,fontsize=fs)


fig, axs = plt.subplots(1, 1,figsize=(50, 25),sharex=False, sharey=False,constrained_layout=True)
#fig.suptitle('Flourishing Classification Accuracy Plots',fontsize=24,verticalalignment='center_baseline',fontweight='bold')
k=0
legends =['Nearest \n Neighbor','Linear SVM','RBF SVM','Gradient \n Boosting','AdaBoost','Naive \n Bayes','LDA','QDA','Logistic \n Regression']
fs=50

data = all_nested_scores_flour_corrdtw_fuse
axs.violinplot(data,showmeans=True,showextrema=True,vert=True)
plt.setp(axs.get_xticklabels(), fontsize=fs, fontweight="bold")
plt.setp(axs.get_yticklabels(), fontsize=fs, fontweight="bold")
axs.set_xlabel('Classifier'+ '\n' ,fontsize=fs,fontweight='bold',labelpad=10)
axs.set_ylabel('Accuracy',fontsize=fs,fontweight='bold')
axs.set_title('Flourishing Classification Accuracy Plots',fontsize=fs,fontweight='bold')
set_axis_style(axs, legends)


print('----------------------%%% For Level 2-Fused (CC-Deoxy + DTW-Oxy for flourishing) Features %%%--------------------------')

flour_cc_dtw_fuse= np.concatenate((corr_dxy_new_flour, dtw_oxy_new_flour),axis=1)


# vit_ccoxy_dtwdeoxy_fuse = LinearSVC(C=0.1, penalty="l1", dual=False).fit(vit_cc_dtw_fuse, class_vit)
# vit_ccoxy_dtwdeoxy_fuse = SelectFromModel(vit_ccoxy_dtwdeoxy_fuse, prefit=True)
# vit_ccoxy_dtwdeoxy_fuse = vit_ccoxy_dtwdeoxy_fuse.transform(vit_cc_dtw_fuse)

# flour_ccoxy_dtwdeoxy_fuse = LinearSVC(C=0.1, penalty="l1", dual=False).fit(flour_cc_dtw_fuse, class_flour)
# flour_ccoxy_dtwdeoxy_fuse = SelectFromModel(flour_ccoxy_dtwdeoxy_fuse, prefit=True)
# flour_ccoxy_dtwdeoxy_fuse = flour_ccoxy_dtwdeoxy_fuse.transform(flour_cc_dtw_fuse)


flour_ccdxy_dtwoxy_fuse = flour_cc_dtw_fuse

all_flour_acc_mean_ccdxy_dtwoxy_fuse=[]
all_flour_acc_std_ccdxy_dtwoxy_fuse=[]



all_fprs_flour_ccdxy_dtwoxy_fuse=[]
all_tprs_flour_ccdxy_dtwoxy_fuse=[]
all_flour_corr_acc_mean_ccdxy_dtwoxy=[]
all_flour_corr_acc_std_ccdxy_dtwoxy=[]
all_nested_scores_flour_corrdtw_fuse = []

mean_tpr_flour_ccdxy_dtwoxy_fuse =[]
mean_fpr_flour_ccdxy_dtwoxy_fuse =[]

std_tpr_vit_ccdxy_dtwoxy_fuse =[]
std_fpr_vit_ccdxy_dtwoxy_fuse =[]
std_tpr_flour_ccdxy_dtwoxy_fuse =[]
std_fpr_flour_ccdxy_dtwoxy_fuse =[]

mean_auc_vit_ccdxy_dtwoxy_fuse =[]
mean_auc_flour_ccdxy_dtwoxy_fuse =[]
mean_fpr = np.linspace(0, 1, 100)

for name, classifier in zip(names, classifiers):
    if name in ('Linear SVM', 'RBF SVM',
                'Gradient Boosting', 'AdaBoost', 'Logistic Regression',
                ):
        verbose = 0
    else:
        verbose = 0
    print(name)
    time.sleep(0.1)
    NUM_TRIALS = 30
    nested_scores_flour_ccdxy_dtwoxy_fuse= np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        param_search = GridSearchCV(
            estimator=classifier, param_grid=params_classifiers[name],
            verbose=verbose, cv =inner_cv)

        # Nested CV with parameter optimization

        # Nested CV with parameter optimization

        nested_score_flour_ccdxy_dtwoxy_fuse = cross_val_score(param_search, X=flour_ccdxy_dtwoxy_fuse, y=class_flour, cv=outer_cv)
        nested_scores_flour_ccdxy_dtwoxy_fuse[i] = nested_score_flour_ccdxy_dtwoxy_fuse.mean()
        

        # Nested CV Prediction
        
        prob_flour_ccdxy_dtwoxy_fuse= cross_val_predict(param_search, X=flour_ccdxy_dtwoxy_fuse, y=class_flour, cv=outer_cv,method='predict_proba')
        
        
        fprs_flour_ccdxy_dtwoxy_fuse,tprs_flour_ccdxy_dtwoxy_fuse,_=roc_curve(class_flour,prob_flour_ccdxy_dtwoxy_fuse[:,1])


        tpr_flour_ccdxy_dtwoxy_fuse=np.interp(mean_fpr,fprs_flour_ccdxy_dtwoxy_fuse,tprs_flour_ccdxy_dtwoxy_fuse)
        tpr_flour_ccdxy_dtwoxy_fuse[0] =0.0              
        
        all_fprs_flour_ccdxy_dtwoxy_fuse.append(fprs_flour_ccdxy_dtwoxy_fuse)

        all_tprs_flour_ccdxy_dtwoxy_fuse.append(tpr_flour_ccdxy_dtwoxy_fuse)
        
    

    
    m_tpr = np.mean(all_tprs_flour_ccdxy_dtwoxy_fuse,axis=0)
    std_tpr = np.std(all_tprs_flour_ccdxy_dtwoxy_fuse,axis=0)
    m_tpr [-1] =1
    
    mean_tpr_flour_ccdxy_dtwoxy_fuse.append(m_tpr)
    std_tpr_flour_ccdxy_dtwoxy_fuse.append(std_tpr)
    mean_fpr_flour_ccdxy_dtwoxy_fuse.append(mean_fpr)
    mean_auc_flour_ccdxy_dtwoxy_fuse.append(auc(mean_fpr,m_tpr))
    
    
    all_fprs_flour_ccdxy_dtwoxy_fuse=[]

    all_tprs_flour_ccdxy_dtwoxy_fuse =[]
    
        
    all_nested_scores_flour_corrdtw_fuse.append(nested_scores_flour_ccdxy_dtwoxy_fuse)
    
    

    print("Flour_corrDeOxy_DTW_Oxy : Average acc. of {:6f} with std. dev. of {:6f}." 
          .format(nested_scores_flour_ccdxy_dtwoxy_fuse.mean(), nested_scores_flour_ccdxy_dtwoxy_fuse.std()))
    all_flour_acc_mean_ccdxy_dtwoxy_fuse.append(nested_scores_flour_ccdxy_dtwoxy_fuse.mean())
    all_flour_acc_std_ccdxy_dtwoxy_fuse.append(nested_scores_flour_ccdxy_dtwoxy_fuse.std())
    
    


chance = np.linspace(0,1,100)
fig, axs = plt.subplots(1, 1,figsize=(25, 25),sharex=False, sharey=False,constrained_layout=True)
#fig.suptitle('ROC Curves of Classification of ..',fontsize=24,verticalalignment='bottom',fontweight='bold')
fs=38
legends =['Nearest Neighbor','Linear SVM','RBF SVM','Gradient Boosting','AdaBoost','Naive Bayes','Linear Discriminant Analysis','Quadratic Discriminant Analysis','Logistic Regression']

for k in range(0,9):
    axs.plot(mean_fpr_flour_ccdxy_dtwoxy_fuse[k],mean_tpr_flour_ccdxy_dtwoxy_fuse[k],linewidth=8)#,yerr = std_tpr_flour_corr_oxy[k])
    auc_fuse = [mean_auc_flour_ccdxy_dtwoxy_fuse[k]]
    legends[k]=legends[k] +', AUC:'+ str("{:.2f}".format(auc_fuse[0]))
    
axs.plot(chance,chance,linewidth=8)   
axs.set_xlabel('False Positive Rate (FPR)',fontsize=fs,fontweight='bold')
axs.set_ylabel('True Positive Rate (TPR)',fontsize=fs,fontweight='bold')
plt.setp(axs.get_xticklabels(), fontsize=fs, fontweight="bold")
plt.setp(axs.get_yticklabels(), fontsize=fs, fontweight="bold")
axs.set_title('ROC Curve of CC-ΔHb & DTW-ΔHbO Fusion',fontsize=fs,fontweight='bold')
legends.append('Chance Level')
axs.legend(legends,fontsize=fs)


fig, axs = plt.subplots(1, 1,figsize=(50, 25),sharex=False, sharey=False,constrained_layout=True)
#fig.suptitle('Flourishing Classification Accuracy Plots',fontsize=24,verticalalignment='center_baseline',fontweight='bold')
k=0
legends =['Nearest \n Neighbor','Linear SVM','RBF SVM','Gradient \n Boosting','AdaBoost','Naive \n Bayes','LDA','QDA','Logistic \n Regression']
fs=50

data = all_nested_scores_flour_corrdtw_fuse
axs.violinplot(data,showmeans=True,showextrema=True,vert=True)
plt.setp(axs.get_xticklabels(), fontsize=fs, fontweight="bold")
plt.setp(axs.get_yticklabels(), fontsize=fs, fontweight="bold")
axs.set_xlabel('Classifier'+ '\n' ,fontsize=fs,fontweight='bold',labelpad=10)
axs.set_ylabel('Accuracy',fontsize=fs,fontweight='bold')
axs.set_title('Flourishing Classification Accuracy Plots',fontsize=fs,fontweight='bold')
set_axis_style(axs, legends)