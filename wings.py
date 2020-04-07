# seaborn
# fancyimpute
#
#
# import pandas as pd
# import numpy as np
# import random as rnd
# import math as m
# import seaborn as sns
# import matplotlib.pyplot as plt
# from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
# import gc
# import os

# os.getcwd()
# os.chdir("/mnt/960CCAFB0CCAD581/Kaggle/Titanic")
# os.getcwd()
# gc.collect()
#
# combined = pd.concat([test_df,train_df],ignore_index=True)
# combined = combined.set_index('PassengerId')
#
#
#
# def KNN_cls(df):
#     param_grid = dict(n_neighbors=list(range(1,20)),weights = ['uniform', 'distance'])
#     knn = KNeighborsClassifier()
#     grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
#     grid.fit(df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df.train == 1,['Survived']],verbose=FALSE)
#     prediction = pd.DataFrame(grid.predict(df.loc[df['train'] == 0].drop('Survived', axis=1)))
#     prediction.columns = df.loc[(df.train == 0),['Survived']].columns
#     prediction.index = df.loc[df['train'] == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = KNN_cls(combined_ii)
# prediction.to_csv('Submit_ii_knn.csv')
#
# prediction = KNN_cls(combined_knn)
# prediction.to_csv('Submit_knn_knn.csv')
#
# prediction = KNN_cls(combined_si)
# prediction.to_csv('Submit_si_knn.csv')
#
# def RF_cls(df):
#     RF_optimal = RandomForestClassifier(n_estimators=optimal_k)
#     RF_optimal.fit(df.loc[df.train == 1].drop('Survived', axis=1),df.loc[df.train == 1,['Survived']])
#     prediction = pd.DataFrame(RF_optimal.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
#     prediction.columns = ['Survived']
#     prediction.index = df.loc[df.train == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = RF_cls(combined_ii)
# prediction.to_csv('Submit_ii_rf.csv')
#
# prediction = RF_cls(combined_knn)
# prediction.to_csv('Submit_knn_rf.csv')
#
# prediction = RF_cls(combined_si)
# prediction.to_csv('Submit_si_rf.csv')
#
# def NB_cls(df):
#     scoring = 'accuracy'
#     k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#     NB = GaussianNB()
#     NB.fit(df.loc[df.train == 1].drop('Survived', axis=1), df.loc[df.train == 1,['Survived']])
#     prediction = pd.DataFrame(NB.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
#     prediction.columns = ['Survived']
#     prediction.index = df.loc[df.train == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = NB_cls(combined_ii)
# prediction.to_csv('Submit_ii_nb.csv')
#
# prediction = NB_cls(combined_knn)
# prediction.to_csv('Submit_knn_nb.csv')
#
# prediction = NB_cls(combined_si)
# prediction.to_csv('Submit_si_nb.csv')
#
# def SVM_cls(df):
#     scoring = 'accuracy'
#     k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#     SV = SVC()
#     SV.fit(df.loc[df.train == 1].drop('Survived', axis=1), df.loc[df.train == 1,['Survived']])
#     prediction = pd.DataFrame(SV.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
#     prediction.columns = ['Survived']
#     prediction.index = df.loc[df.train == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = SVM_cls(combined_ii)
# prediction.to_csv('Submit_ii_svm.csv')
#
# prediction = SVM_cls(combined_knn)
# prediction.to_csv('Submit_knn_svm.csv')
#
# prediction = SVM_cls(combined_si)
# prediction.to_csv('Submit_si_svm.csv')
#
# def GBM_cls(df):
#     param_grid = dict(learning_rate = [0.9,0.45,0.225,0.112,0.056,0.028,0.014,0.007,0.0035,0.0017,0.00087,0.00043,0.00022,0.0001],n_estimators = [10,30,50,80,100,140,170,200],subsample = [1,0.8,0.6,0.3,0.1],max_depth = [2,3,4,5,6,7,8])
#     gbm = GradientBoostingClassifier()
#     grid = GridSearchCV(gbm, param_grid, cv=10, scoring='accuracy', return_train_score=False)
#     grid.fit(df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df.train == 1,['Survived']])
#     prediction = pd.DataFrame(grid.predict(df.loc[df['train'] == 0].drop('Survived', axis=1)))
#     prediction.columns = df.loc[(df.train == 0),['Survived']].columns
#     prediction.index = df.loc[df['train'] == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# def GBM_cls(df):
#     scoring = 'accuracy'
#     k_fold = KFold(n_splits=10, shuffle=True)
#     cv_scores = []
#     for k1 in range(1,100):
#         score = cross_val_score(GradientBoostingClassifier( learning_rate = np.log(m.exp(k1/(k1*k1 +1)))),df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df['train'] == 1,['Survived']], cv=k_fold, scoring=scoring)
#         cv_scores.append(score.mean())
#         if (len(cv_scores) > 10 and abs(score.mean()-sum(cv_scores[-5:])/5) < 0.0005):
#             break
#         optimal_k = cv_scores.index(max(cv_scores)) + 1
#         lr = np.log(m.exp(optimal_k/(optimal_k*optimal_k +1)))
#         print("learninbg rate")
#     cv_scores = []
#     for k2 in range(1,20):
#         score = cross_val_score(GradientBoostingClassifier( learning_rate = lr,n_estimators=k2*10),df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df['train'] == 1,['Survived']], cv=k_fold, scoring=scoring)
#         cv_scores.append(score.mean())
#         if (len(cv_scores) > 10 and abs(score.mean()-sum(cv_scores[-5:])/5) < 0.0005):
#             break
#         print("n_estimators")
#     cv_scores = []
#     for k3 in range(1,k1):
#         for k4 in range(1,k2):
#             score = cross_val_score(GradientBoostingClassifier( learning_rate = np.log(m.exp(k3/(k3*k3 +1))),n_estimators=k4*10),df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df['train'] == 1,['Survived']], cv=k_fold, scoring=scoring)
#             cv_scores.append([score.mean(),np.log(m.exp(k3/(k3*k3 +1))),k4*10])
#             print("lr_ne_grid")
#     scores = pd.DataFrame(cv_scores, columns=['score','lr','n_est'])
#     scores = scores.nlargest(5,'score',keep='first')
#     scores = scores.reset_index().drop('index', axis=1)
#     cv_scores = []
#     for k in range(1,10):
#         for index,row in scores.iterrows():
#             score = cross_val_score(GradientBoostingClassifier(learning_rate=row['lr'],n_estimators=row['n_est'].astype(int),subsample=(11-k)/10),df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df['train'] == 1,['Survived']], cv=k_fold, scoring=scoring)
#             cv_scores.append([score.mean(),row['lr'],row['n_est'],(11-k)/10])
#             print("subsample")
#     scores = pd.DataFrame(cv_scores, columns=['score','lr','n_est','subsample'])
#     scores = scores.nlargest(5,'score',keep='first')
#     scores = scores.reset_index().drop('index', axis=1)
#     cv_scores = []
#     for k in range(2,9):
#         for index,row in scores.iterrows():
#             score = cross_val_score(GradientBoostingClassifier(learning_rate=row['lr'],n_estimators=row['n_est'].astype(int),subsample=row['subsample'],max_depth=k),df.loc[df['train'] == 1].drop('Survived', axis=1),df.loc[df['train'] == 1,['Survived']], cv=k_fold, scoring=scoring)
#             cv_scores.append([score.mean(),row['lr'],row['n_est'],row['subsample'],k])
#             print("max_depth")
#     scores = pd.DataFrame(cv_scores, columns=['score','lr','n_est','subsample','max_depth'])
#     scores = scores.nlargest(1,'score',keep='first')
#     scores = scores.reset_index().drop('index', axis=1)
#     gbm = GradientBoostingClassifier(learning_rate=scores.loc[0].lr,n_estimators=scores.loc[0].n_est.astype(int),subsample=scores.loc[0].subsample,max_depth = scores.loc[0].max_depth.astype(int))
#     gbm.fit(df.loc[df.train == 1].drop('Survived', axis=1),df.loc[df.train == 1,['Survived']].astype(int))
#     prediction = pd.DataFrame(gbm.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
#     prediction.columns = ['Survived']
#     prediction.index = df.loc[df.train == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = GBM_cls(combined_ii)
# prediction.to_csv('Submit_ii_GBM.csv')
#
# prediction = GBM_cls(combined_knn)
# prediction.to_csv('Submit_knn_GBM.csv')
#
# prediction = GBM_cls(combined_si)
# prediction.to_csv('Submit_si_GBM.csv')
#
#
# gbm = GradientBoostingClassifier()
# gbm.fit(combined_ii.loc[combined_ii.train == 1].drop('Survived', axis=1),combined_ii.loc[combined_ii.train == 1,['Survived']].astype(int))
# prediction = pd.DataFrame(gbm.predict(combined_ii.loc[combined_ii.train == 0].drop('Survived', axis=1)))
# prediction.columns = ['Survived']
# prediction.index = combined_ii.loc[combined_ii.train == 0].index
# prediction.Survived = prediction.Survived.round().astype(int)
# prediction.to_csv('Submit_ii_GBM_ne.csv')
#
#
# gbm.get_params(deep=True)
# print(gbm.feature_importances_)
# len(gbm.feature_importances_)
# len(combined_ii.columns)
#
#
#
#
#
#
# x = combined_ii.Age
# y = combined_ii.Survived
#
# def mygini(x,y,m,b):
#     try:
#         float(x[1])
#         z = 1
#     except:
#         z = 0
#         if z = 1:
#             df = pd.DataFrame({'x' : x,'y' : y})
#             +
#             len(x.unique())
#
#
# gc.collect()
#
#
# mygini(combined_ii.loc[combined_ii.train == 1].Age
# ,combined_ii.loc[combined_ii.train == 1].Survived,5%,10)
#
