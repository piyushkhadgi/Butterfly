from Bones import install
from Bones import read
from Bones import ProjectConfig

packages = ['subprocess','pandas']

install(packages)

config = ProjectConfig()

df = read(config)




# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


