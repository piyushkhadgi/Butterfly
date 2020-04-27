from ML import DT_cls
from ML import RF_cls
from Base import read
from Util import describe, feature_create, feature_plot
from Base import ProjectConfig

if __name__ == '__main__':
    config = ProjectConfig(project='titanic', source='kaggle', threshold=0.383838, target='Survived',
                           primary='PassengerId')

    #    read(config)
    #    feature_create(config)
    #    feature_plot(config)

    RF_cls(config, min_samples_leaf=0.05, n_estimators=100)
    RF_cls(config, min_samples_leaf=0.05, n_estimators=500)
    RF_cls(config, min_samples_leaf=0.05, n_estimators=250)
    RF_cls(config, min_samples_leaf=0.05, n_estimators=750)


# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit
