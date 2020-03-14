from bones import install
from bones import read
from bones import describe
from bones import ProjectConfig

packages = ['subprocess', 'pandas']

install(packages)

config = ProjectConfig()

#read(config)
describe(config)





# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


