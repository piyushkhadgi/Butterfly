from skeleton import install
from skeleton import read
from skeleton import project_config

packages = ['subprocess','pandas']

install(packages)

config = project_config()

df = read(config)

print(df)

# todo: Feature understanding
# todo: Missing Value
# todo: Feature engineering
# todo: Feature understanding
# todo: Sampling code

# todo: Train
# todo: validate
# todo: predict
# todo: Store prediction
# todo: Submit


