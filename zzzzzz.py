from bones import install
from bones import read
from bones import describe_raw
from bones import ProjectConfig
import pandas

packages = ['subprocess', 'pandas']

install(packages)

config = ProjectConfig()

#read(config)
describe_raw(config)


# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


