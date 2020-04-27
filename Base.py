import pip
import subprocess
import pandas

#pip.main(['install', package])

class ProjectConfig:
    """ class created to store all parameters and locations of this project."""
    project: str = ''
    source: str = ''
    root_path: str = ''
    proj_path: str = ''
    raw_file: str=''
    feature_file: str=''
    result_file: str=''
    threshold: float=0.5
    target: str=''
    primary: str=''


    def __init__(self, project='titanic', source='kaggle', root_path='/home/swayush/ML/', threshold = 0.5, target = 'Survived', primary = 'PassengerId'):
        self.project = project
        self.source = source
        self.root_path = root_path
        self.proj_path = root_path + source[0:1] + '_' + project
        self.raw_file = self.proj_path + '/raw.csv'
        self.feature_file = self.proj_path + '/feature.csv'
        self.result_file = self.proj_path + '/result.csv'
        self.temp_loc = self.proj_path + '/temp/'
        self.threshold = threshold
        self.target = target
        self.primary = primary


def read(config):

    """ Function to read the modeling data from competition page and store on your system."""
    if config.source == 'kaggle':
        folder = 'k_' + config.project
        file_name = config.project + '.zip'
#        location = config.proj_path

        out = subprocess.check_output(["ls", config.root_path]).decode("utf-8")
        if folder not in out.split('\n'):
            subprocess.check_output(["mkdir", config.proj_path]).decode("utf-8")

        out = subprocess.check_output(["ls", config.proj_path]).decode("utf-8")
        if 'histographs' not in out.split('\n'):
            subprocess.check_output(["mkdir", config.proj_path + '/histographs']).decode("utf-8")
        if 'temp' not in out.split('\n'):
            subprocess.check_output(["mkdir", config.temp_loc]).decode("utf-8")
        out = subprocess.check_output(["ls"]).decode("utf-8")
        if file_name in out.split('\n'):
            subprocess.check_output(["rm", file_name]).decode("utf-8")
        subprocess.check_output(['kaggle', 'competitions', 'download', config.project]).decode("utf-8")
        subprocess.check_output(['cp', file_name, config.proj_path]).decode("utf-8")
        subprocess.check_output(["rm", file_name]).decode("utf-8")
        out = subprocess.check_output(["ls", config.proj_path]).decode("utf-8")
        if 'raw_data' in out.split('\n'):
            out = subprocess.check_output(["rm", '-r', config.proj_path + '/raw_data']).decode("utf-8")
        subprocess.check_output(["mkdir", config.proj_path + '/raw_data']).decode("utf-8")
        subprocess.check_output(["unzip", config.proj_path + '/' + file_name, '-d', config.proj_path + '/raw_data']).decode("utf-8")
        df_trn = pandas.read_csv(config.proj_path + '/raw_data/train.csv')
        df_trn['_data_'] = 1
        config.threshold = df_trn[config.target].mean()
        df_tst = pandas.read_csv(config.proj_path + '/raw_data/test.csv')
        df_tst['_data_'] = 0
        df = df_trn.append(df_tst, ignore_index=True)
        df.to_csv(config.raw_file, index=False)
    else:
        print('Invalid Source')
    return None

