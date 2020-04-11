import sklearn.tree
import pandas


def DT_cls(config):

    df = pandas.read_csv(config.feature_file, delimiter=',')

    DT = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = 0.1)
    DT.fit(df.loc[df._data_ == 1].drop(config.target, axis=1).drop(config.primary, axis=1), df.loc[df._data_ == 1,[config.target]])
    prediction = pandas.DataFrame(DT.predict_proba(df.loc[df._data_ == 0].drop(config.target, axis=1).drop(config.primary, axis=1))[:, 1])
    prediction.columns = ['prob']
    prediction['Source'] = 'SKL_DT'
    prediction[config.primary] = df.loc[df._data_ == 0].index

    prediction.loc[prediction['prob'] <= config.threshold,config.primary] = 0
    prediction.loc[prediction['prob'] > config.threshold,config.primary] = 1

    out = subprocess.check_output(["ls", location]).decode("utf-8")
    if 'raw_data' in out.split('\n'):

    prediction.to_csv(path_or_buf = '/home/swayush/ML/file.csv', index=False)
    mode = 'a', header = False

    #    prediction[config.target] = prediction.prob.round().astype(int)
#    DT_cls_result = prediction

    return prediction;
