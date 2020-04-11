from ML import DT_cls
from Base import read
from Util import describe, feature_create, feature_plot
from Base import ProjectConfig

if __name__ == '__main__':

    config = ProjectConfig(project='titanic', source='kaggle', threshold = 0.383838, target = 'Survived', primary = 'PassengerId')

#    read(config)
#    print(describe(config).head(10))
#    feature_create(config)
#    feature_plot(config)


    DT_cls(config)



#    df = describe(config)
    #key = df.index.tolist()


# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


