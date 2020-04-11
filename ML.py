import sklearn.tree
import pandas


def DT_cls(config):
    df = pandas.read_csv(config.feature_file, delimiter=',')

    DT = sklearn.tree.DecisionTreeClassifier()
    DT.fit(df.loc[df._data_ == 1].drop('Survived', axis=1), df.loc[df._data_ == 1,['Survived']])
    prediction = pandas.DataFrame(DT.predict(df.loc[df._data_ == 0].drop('Survived', axis=1)))
    prediction.columns = ['Survived']
    prediction.index = df.loc[df._data_ == 0].index
    prediction.Survived = prediction.Survived.round().astype(int)
    prediction.Source = 'SKL_DT'
    DT_cls_result = prediction

    return prediction;

