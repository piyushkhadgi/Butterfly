from bones import install
from bones import read
from flesh import describe
from bones import ProjectConfig
import pandas

packages = ['subprocess', 'pandas']

install(packages)

config = ProjectConfig()

#read(config)
df = describe(config)


key = df.index.tolist()


# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


