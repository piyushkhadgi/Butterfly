import subprocess
import pandas
import pip

class ProjectConfig:
    """ class created to store all parameters and locations of this project."""
    project: str = ''
    source: str = ''
    root_path: str = ''
    raw_file: str=''

    def __init__(self, project='titanic', source='kaggle', root_path='/home/swayush/ML/'):
        self.project = project
        self.source = source
        self.root_path = root_path
        self.raw_file = root_path + source[0:1] + '_' + project + '/raw.csv'

def install(packages):
    """ Function to install any missing package."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            pip.main(['install', package])

def read(config):
    """ Function to read the modeling data from competition page and store on your system."""
    if config.source == 'kaggle':
        folder = 'k_' + config.project
        file_name = config.project + '.zip'
        location = config.root_path + folder

        out = subprocess.check_output(["ls", config.root_path]).decode("utf-8")
        if folder not in out.split('\n'):
            subprocess.check_output(["mkdir", location]).decode("utf-8")
        out = subprocess.check_output(["ls"]).decode("utf-8")
        if file_name in out.split('\n'):
            subprocess.check_output(["rm", file_name]).decode("utf-8")
        subprocess.check_output(['kaggle', 'competitions', 'download', config.project]).decode("utf-8")
        subprocess.check_output(['cp', file_name, location]).decode("utf-8")
        subprocess.check_output(["rm", file_name]).decode("utf-8")
        out = subprocess.check_output(["ls", location]).decode("utf-8")
        if 'raw_data' in out.split('\n'):
            out = subprocess.check_output(["rm", '-r', location + '/raw_data']).decode("utf-8")
        subprocess.check_output(["mkdir", location + '/raw_data']).decode("utf-8")
        subprocess.check_output(["unzip", location + '/' + file_name, '-d', location + '/raw_data']).decode("utf-8")
        df_trn = pandas.read_csv(location + '/raw_data/train.csv')
        df_trn['_data_'] = 'train'
        df_tst = pandas.read_csv(location + '/raw_data/test.csv')
        df_tst['_data_'] = 'test'
        df = df_trn.append(df_tst, ignore_index=True)
        df.to_csv(config.raw_file, index=False)
    else:
        print('Invalid Source')
    return None

