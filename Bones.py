import subprocess
import pandas

class project_config:
    project: str = ''
    source: str = ''
    root_path: str = ''

    def __init__(self, project='titanic', source='kaggle', root_path='/home/swayush/ML/'):
        self.project = project
        self.source = source
        self.root_path = root_path


def install(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            pip.main(['install', package])

def read(config):
    if config.source == 'kaggle':
        folder = 'k_' + config.project
        file_name = config.project + '.zip'

        out = subprocess.check_output(["ls", config.root_path]).decode("utf-8")
        if folder not in out.split('\n'):
            subprocess.check_output(["mkdir", config.root_path + folder]).decode("utf-8")
        out = subprocess.check_output(["ls"]).decode("utf-8")
        if file_name in out.split('\n'):
            subprocess.check_output(["rm", file_name]).decode("utf-8")
        subprocess.check_output(['kaggle', 'competitions', 'download', config.project]).decode("utf-8")
        subprocess.check_output(['cp', file_name, config.root_path + folder]).decode("utf-8")
        subprocess.check_output(["rm", file_name]).decode("utf-8")
        out = subprocess.check_output(["ls", config.root_path + folder]).decode("utf-8")
        if 'raw_data' in out.split('\n'):
            out = subprocess.check_output(["rm", '-r', config.root_path + folder + '/raw_data']).decode("utf-8")
        subprocess.check_output(["mkdir", config.root_path + folder + '/raw_data']).decode("utf-8")
        subprocess.check_output(["unzip", config.root_path + folder + '/' + file_name, '-d', config.root_path + folder + '/raw_data']).decode("utf-8")
        df_trn = pandas.read_csv(config.root_path + folder + '/raw_data/train.csv')
        df_trn['_data_'] = 'train'
        df_tst = pandas.read_csv(config.root_path + folder + '/raw_data/test.csv')
        df_tst['_data_'] = 'test'
        df = df_trn.append(df_tst, ignore_index=True)
        df.to_csv(config.root_path + folder+'/raw.csv',index=False)

    else:
        print('Invalid Source')
        df = None
    return(df)
