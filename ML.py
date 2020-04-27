import sklearn.tree
import sklearn.ensemble
import pandas
import subprocess
import time


def RF_cls(config, min_samples_leaf, n_estimators):
    df = pandas.read_csv(config.feature_file, delimiter=',')
    prediction = pandas.DataFrame(columns=[config.primary])
    prediction[config.primary] = df.loc[df._data_ == 0].index + 1

    RF = sklearn.ensemble.RandomForestClassifier(min_samples_leaf = min_samples_leaf, random_state = 1234, n_estimators = n_estimators)
    RF.fit(df.loc[df._data_ == 1].drop(config.target, axis=1).drop(config.primary, axis=1), df.loc[df._data_ == 1,[config.target]])
    prediction['prob'] = pandas.DataFrame(RF.predict_proba(df.loc[df._data_ == 0].drop(config.target, axis=1).drop(config.primary, axis=1))[:, 1])
    prediction.prob=round(prediction.prob, 4)
    prediction.loc[prediction['prob'] <= config.threshold,config.target] = 0
    prediction.loc[prediction['prob'] > config.threshold,config.target] = 1
    prediction[config.primary] = prediction[config.primary].round().astype(int)
    prediction[config.target] = prediction[config.target].round().astype(int)
    write_to_memory(prediction,config,algo="SKL_RF",para = 'nodesize=' + str(min_samples_leaf) + 'n_estimators=' + str(n_estimators))
    return



def DT_cls(config,min_samples_leaf):

    df = pandas.read_csv(config.feature_file, delimiter=',')
    prediction = pandas.DataFrame(columns=[config.primary])
    prediction[config.primary] = df.loc[df._data_ == 0].index + 1
    DT = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = min_samples_leaf, random_state = 1234)
    DT.fit(df.loc[df._data_ == 1].drop(config.target, axis=1).drop(config.primary, axis=1), df.loc[df._data_ == 1,[config.target]])
    prediction['prob'] = pandas.DataFrame(DT.predict_proba(df.loc[df._data_ == 0].drop(config.target, axis=1).drop(config.primary, axis=1))[:, 1])
    prediction.prob=round(prediction.prob, 4)
    prediction.loc[prediction['prob'] <= config.threshold,config.target] = 0
    prediction.loc[prediction['prob'] > config.threshold,config.target] = 1
    prediction[config.primary] = prediction[config.primary].round().astype(int)
    prediction[config.target] = prediction[config.target].round().astype(int)

    write_to_memory(prediction,config,algo="SKL_DT",para = 'nodesize=' + str(min_samples_leaf))
    return


def write_to_memory(prediction,config,algo,para):

    prediction['message'] = para
    out = subprocess.check_output(["ls", config.proj_path]).decode("utf-8")
    if 'result.csv' in out.split('\n'):
        prob_list = []
        res = pandas.read_csv(config.result_file, delimiter=',')
        unique_algo = res.algo.unique()
        for x in unique_algo:
            prob_list.append(res.loc[res['algo'] == x, config.target])
        pred_list = prediction[config.target]
        for every_prob_list in prob_list:
            comp = pred_list.eq(every_prob_list)
            if sum(comp)/len(comp) > 0.95:
                print('Repeat sequence')
                prediction[[config.primary, config.target]].to_csv(config.temp_loc + algo + '_repeat.csv', index=False)
                return
        temp = pandas.DataFrame(filter(lambda x: x.startswith(algo), res.algo.unique()), columns=['algo'])
        temp['series'] = pandas.to_numeric(temp['algo'].str[-4:])

        if len(temp.index) == 0:
            prediction['algo'] = algo + '_0001'
            prediction = sumbit_result(prediction, config, para, config.temp_loc + algo + '_0001.csv')
            prediction.to_csv(path_or_buf=config.result_file, index=False, mode='a', header=False)
        else:
            prediction['algo'] = algo + '_' + f"{temp.series.max()+1:04}"
            prediction = sumbit_result(prediction, config, para, config.temp_loc + algo + '_' + f"{temp.series.max()+1:04}" +'.csv')
            prediction.to_csv(path_or_buf=config.result_file, index=False, mode = 'a', header = False)
    else:
        prediction['algo'] = algo + '_0001'
        prediction =  sumbit_result(prediction, config, para, config.temp_loc + algo + '_0001.csv')
        prediction.to_csv(path_or_buf=config.result_file, index=False)
    return


def sumbit_result(prediction,config,para,name):
    prediction[[config.primary, config.target]].to_csv(name, index=False)
    try:
        out = subprocess.check_output(
            ['kaggle', 'competitions', 'submit', config.project, '-f', name, '-m',
             para]).decode("utf-8")
        print(out)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(format(e.output))
    if out[:10] != 'Successful':
        print('Error')
        return
    score = 'pending'
    n = 10
    i = 1
    while score == 'pending' or score == 'None':
        print("Lets wait for " + str(n) + " seconds for results to get evaluated. Loop 1")
        time.sleep(n)
        out = subprocess.check_output(['kaggle', 'competitions', 'submissions', config.project]).decode("utf-8")
        out = out.splitlines()
        out = " ".join(out[2].split()).split(" ")
        score = out[5]
        i = i + 1
    prediction['score'] = score
    return prediction


