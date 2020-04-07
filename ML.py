def DT_cls(df):
    scoring = 'accuracy'
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    DT = DecisionTreeClassifier()
    DT.fit(df.loc[df.train == 1].drop('Survived', axis=1), df.loc[df.train == 1,['Survived']])
    prediction = pd.DataFrame(DT.predict(df.loc[df.train == 0].drop('Survived', axis=1)))
    prediction.columns = ['Survived']
    prediction.index = df.loc[df.train == 0].index
    prediction.Survived = prediction.Survived.round().astype(int)
    gc.collect()
    return prediction;

prediction = DT_cls(combined_ii)
prediction.to_csv('Submit_ii_dt.csv')

prediction = DT_cls(combined_knn)
prediction.to_csv('Submit_knn_dt.csv')

prediction = DT_cls(combined_si)
prediction.to_csv('Submit_si_dt.csv')
