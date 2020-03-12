from Bones import Install
from Bones import Read
from Bones import ProjectConfig

Packages = ['subprocess','pandas']

Install(Packages)

Config = ProjectConfig()

Df = Read(Config)




# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


