from fundamentals import read
from help import describe, feature_create
from fundamentals import ProjectConfig

if __name__ == '__main__':

    config = ProjectConfig()
#    read(config)
    print(describe(config).head(10))
    feature_create(config)

#    df = describe(config)
    #key = df.index.tolist()


# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


