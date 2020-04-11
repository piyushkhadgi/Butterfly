import sklearn.tree
import pandas
import subprocess


def DT_cls(config):

    df = pandas.read_csv(config.feature_file, delimiter=',')

    DT = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = 0.1)
    DT.fit(df.loc[df._data_ == 1].drop(config.target, axis=1).drop(config.primary, axis=1), df.loc[df._data_ == 1,[config.target]])
    prediction = pandas.DataFrame(DT.predict_proba(df.loc[df._data_ == 0].drop(config.target, axis=1).drop(config.primary, axis=1))[:, 1])
    prediction.columns = ['prob']
    prediction[config.primary] = df.loc[df._data_ == 0].index

    prediction.loc[prediction['prob'] <= config.threshold,config.primary] = 0
    prediction.loc[prediction['prob'] > config.threshold,config.primary] = 1

    out = subprocess.check_output(["ls",config.proj_path]).decode("utf-8")
    if 'result.csv' in out.split('\n'):
        res = pandas.read_csv(config.result_file, delimiter=',')
        temp = pandas.DataFrame(filter(lambda x: x.startswith('SKL_DT_'), res.algo.unique()), columns=['algo'])
        temp['series'] = pandas.to_numeric(temp['algo'].str[-4:])
        if len(temp.index) == 0:
            prediction['algo'] = 'SKL_DT_0001'
            prediction.to_csv(path_or_buf=config.result_file, index=False, mode='a', header=False)
        else:
            prediction['algo'] = 'SKL_DT_' + f"{temp.series.max()+1:04}"
            prediction.to_csv(path_or_buf=config.result_file, index=False, mode = 'a', header = False)
    else:
        prediction['algo'] = 'SKL_DT_0001'
        prediction.to_csv(path_or_buf=config.result_file, index=False)




    #    prediction[config.target] = prediction.prob.round().astype(int)
#    DT_cls_result = prediction

    return prediction;

