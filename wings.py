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
# # Feature creation title
#
# combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
# combined.loc[combined["Title"] == 'Mr','Title2'] = 'Mr'
# combined.loc[combined["Title"] == 'Mrs','Title2'] = 'Mrs'
# combined.loc[combined["Title"] == 'Miss','Title2'] = 'Miss'
# combined.loc[combined["Title"] == 'Master','Title2'] = 'Master'
# combined.loc[combined["Title"] == 'Ms','Title2'] = 'Ms'
# combined.loc[combined["Title"] == 'Mlle','Title2'] = 'Miss'
# combined.loc[combined["Title"] == 'Ms','Title2'] = 'Miss'
# combined.loc[combined["Title"] == 'Mme','Title2'] = 'Mrs'
# combined.loc[combined["Title2"].isna(),'Title2'] = 'Rare'
# df_title = pd.get_dummies(combined.Title2 , prefix = 'Title')
# combined = pd.concat([combined,df_title],axis=1)
# combined['Name_len'] = combined.Name.str.len()
# combined['Name_space'] = combined.Name.str.count(' ')
# combined = combined.drop(['Name'], axis=1)
# combined = combined.drop(['Title'], axis=1)
# combined = combined.drop(['Title2'], axis=1)
# del df_title
# gc.collect()
#
# # Missing Value treatment for Cabin
#
# combined['Cabin_new'] = combined.Cabin.str[:1]
# combined = combined.drop(['Cabin'], axis=1)
# df_cabin = pd.get_dummies(combined.Cabin_new,prefix = 'Cabin', dummy_na = True)
# combined = pd.concat([combined,df_cabin],axis=1)
# combined = combined.drop(['Cabin_new'], axis=1)
# del df_cabin
# gc.collect()
#
# # Feature creation Sex
#
# combined['Gender'] = combined['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# df_sex = pd.get_dummies(combined.Sex , prefix = 'Sex')
# combined = pd.concat([combined,df_sex],axis=1)
# combined = combined.drop(['Sex'], axis=1)
# del df_sex
# gc.collect()
#
# # Family
#
# combined['FamilySize'] = combined['SibSp'] + combined['Parch'] +1
# combined['withsomebody'] = combined['SibSp'] + combined['Parch']
# combined["isalone"] = combined['withsomebody']
# combined["isalone"].loc[combined['withsomebody'] > 0] = 0
# combined["isalone"].loc[combined['withsomebody'] == 0] = 1
#
#
# # Missing Value treatment for Embarked
#
# combined['Embarked'] = combined['Embarked'].fillna('C')
# df_Embarked = pd.get_dummies(combined.Embarked , prefix = 'Embarked')
# combined = pd.concat([combined,df_Embarked],axis=1)
# combined = combined.drop(['Embarked'], axis=1)
# del df_Embarked
# gc.collect()
#
#
# # Missing Value treatment for Ticket
#
# new = combined["Ticket"].str.split(" ", n = 2, expand = True)
# new[3] = np.where(new[2].isna(),new[1],new[2])
# new['Ticket1'] = np.where(new[3].isna(),new[0],new[3])
# new['Ticket2'] = new[0].str.extract('([A-Za-z]+)',expand=False)
# new['T_length'] = new.Ticket1.str.len()
# new['T_First'] = new.Ticket1.str[:1]
# new = new.drop([0], axis=1)
# new = new.drop([1], axis=1)
# new = new.drop([2], axis=1)
# new = new.drop([3], axis=1)
# new = new.drop(['Ticket1'], axis=1)
# combined = pd.concat([combined,new],axis=1)
# combined = combined.drop(['Ticket'], axis=1)
#
# combined.loc[combined['T_length'] < 5,'T_l_new'] = 'S'
# combined.loc[combined['T_length'] == 5,'T_l_new'] = 'M'
# combined.loc[combined['T_length'] > 5,'T_l_new'] = 'L'
#
# combined.loc[combined['T_First'] == '1','T_f_new'] = 'S'
# combined.loc[combined['T_First'] == '2','T_f_new'] = 'M'
# combined.loc[combined['T_f_new'].isna(),'T_f_new'] = 'L'
#
# combined['High_ticket'] = combined['Ticket2'].isin(['PP','PC','C','P'])
#
# df_t1 = pd.get_dummies(combined.T_l_new, prefix = 'T_l')
# combined = pd.concat([combined,df_t1],axis=1)
# del df_t1
# df_t2 = pd.get_dummies(combined.T_f_new, prefix = 'T_F')
# combined = pd.concat([combined,df_t2],axis=1)
# del df_t2
# del new
# gc.collect()
#
# combined = combined.drop(['T_l_new'], axis=1)
# combined = combined.drop(['T_f_new'], axis=1)
# combined = combined.drop(['T_First'], axis=1)
# combined = combined.drop(['T_length'], axis=1)
# combined = combined.drop(['Ticket2'], axis=1)
#
# # interaction between class and age
# combined['Age*Class'] = combined["Age"]*combined["Pclass"]
#
# # interaction between class and child
# combined.loc[combined['Age'] < 16,'ischild'] = 1
# combined.loc[combined.ischild.isna(),'ischild'] = 0
# combined["Child*Class"] = combined["ischild"]*combined["Pclass"]
#
# # interaction between class and gender
# combined["Gender*Class"] = combined["Gender"]*combined["Pclass"]
#
# # Missing Value treatment for Age
#
#
#
# combined_knn = pd.DataFrame(KNN(k=100).fit_transform(combined.drop('Survived', axis=1)))
# combined_knn.columns = combined.drop('Survived', axis=1).columns
# combined_knn.index = combined.index
# combined_knn = pd.concat([combined_knn,combined.Survived],axis=1)
#
# Submit_knn = combined_knn.loc[(combined_knn.train== 0),['Survived']]
# Submit_knn.Survived = Submit_knn.Survived.round().astype(int)
# Submit_knn.to_csv('Submit_knn.csv')
# combined_knn.loc[combined_knn['train'] == 0,'Survived'] = float('NaN')
#
# combined_ii = pd.DataFrame(IterativeImputer().fit_transform(combined.drop('Survived', axis=1)))
# combined_ii.columns = combined.drop('Survived', axis=1).columns
# combined_ii.index = combined.index
# combined_ii.index = combined.index
# combined_ii = pd.concat([combined_ii,combined.Survived],axis=1)
#
# Submit_ii = combined_ii.loc[(combined_ii.train== 0),['Survived']]
# Submit_ii.Survived = Submit_ii.Survived.round().astype(int)
# Submit_ii.to_csv('Submit_ii.csv')
# combined_ii.loc[combined_ii['train'] == 0,'Survived'] = float('NaN')
#
# #combined_nnm = pd.DataFrame(NuclearNormMinimization().fit_transform(combined))
# #combined_nnm.columns = combined.columns
# #combined_nnm.index = combined.index
# #Submit_nnm = combined_nnm.loc[(combined_nnm.train== 0),['Survived']]
# #Submit_nnm.Survived = Submit_nnm.Survived.round().astype(int)
# #Submit_nnm.to_csv('Submit_nnm.csv')
# #combined_nnm.loc[combined_nnm['train'] == 0,'Survived'] = float('NaN')
#
# combined_si = pd.DataFrame(SoftImpute().fit_transform(combined.drop('Survived', axis=1)))
# combined_si.columns = combined.drop('Survived', axis=1).columns
# combined_si.index = combined.index
# combined_si.index = combined.index
# combined_si = pd.concat([combined_si,combined.Survived],axis=1)
#
# Submit_si = combined_si.loc[(combined_si.train== 0),['Survived']]
# Submit_si.Survived = Submit_si.Survived.round().astype(int)
# Submit_si.loc[Submit_si['Survived'] == 2,'Survived'] = 1
# Submit_si.to_csv('Submit_si.csv')
# combined_si.loc[combined_si['train'] == 0,'Survived'] = float('NaN')
#
# datasets_all = [combined_knn,combined_ii,combined_si]
#
# for dataset in datasets_all:
#     dataset.loc[ dataset['Age'] <= 16, 'Age_new'] = 'A',
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age_new'] = 'B',
#     dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age_new'] = 'C',
#     dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age_new'] = 'D',
#     dataset.loc[ dataset['Age'] > 62, 'Age_new'] = 'E'
#
# df_age = pd.get_dummies(combined_si.Age_new,prefix = 'Age')
# combined_si = pd.concat([combined_si,df_age],axis=1)
# combined_si = combined_si.drop(['Age_new'], axis=1)
#
# df_age = pd.get_dummies(combined_ii.Age_new,prefix = 'Age')
# combined_ii = pd.concat([combined_ii,df_age],axis=1)
# combined_ii = combined_ii.drop(['Age_new'], axis=1)
#
# df_age = pd.get_dummies(combined_knn.Age_new,prefix = 'Age')
# combined_knn = pd.concat([combined_knn,df_age],axis=1)
# combined_knn = combined_knn.drop(['Age_new'], axis=1)
#
# del df_age
# gc.collect()
#
# def DT_cls(df):
#     scoring = 'accuracy'
#     k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#     DT = DecisionTreeClassifier()
#     DT.fit(df.loc[df.train == 1].drop('Survived', axis=1), df.loc[df.train == 1,['Survived']])
#     prediction = pd.DataFrame(DT.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
#     prediction.columns = ['Survived']
#     prediction.index = df.loc[df.train == 0].index
#     prediction.Survived = prediction.Survived.round().astype(int)
#     gc.collect()
#     return prediction;
#
# prediction = DT_cls(combined_ii)
# prediction.to_csv('Submit_ii_dt.csv')
#
# prediction = DT_cls(combined_knn)
# prediction.to_csv('Submit_knn_dt.csv')
#
# prediction = DT_cls(combined_si)
# prediction.to_csv('Submit_si_dt.csv')
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
