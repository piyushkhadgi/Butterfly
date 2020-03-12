import subprocess
import pandas

class ProjectConfig:
    Project: str = ''
    Source: str = ''
    RootPath: str = ''

    def __init__(self, Project='titanic', Source='kaggle', RootPath='/home/swayush/ML/'):
        self.Project = Project
        self.Source = Source
        self.RootPath = RootPath


def Install(Packages):
    for Package in Packages:
        try:
            __import__(Package)
        except ImportError:
            pip.main(['install', Package])

def Read(Config):
    if Config.Source == 'kaggle':
        Folder = 'k_' + Config.Project
        FileName = Config.Project + '.zip'

        Out = subprocess.check_output(["ls", Config.RootPath]).decode("utf-8")
        if Folder not in Out.split('\n'):
            subprocess.check_output(["mkdir", Config.RootPath + Folder]).decode("utf-8")
        Out = subprocess.check_output(["ls"]).decode("utf-8")
        if FileName in Out.split('\n'):
            subprocess.check_output(["rm", FileName]).decode("utf-8")
        subprocess.check_output(['kaggle', 'competitions', 'download', Config.Project]).decode("utf-8")
        subprocess.check_output(['cp', FileName, Config.RootPath + Folder]).decode("utf-8")
        subprocess.check_output(["rm", FileName]).decode("utf-8")
        Out = subprocess.check_output(["ls", Config.RootPath + Folder]).decode("utf-8")
        if 'raw_data' in Out.split('\n'):
            Out = subprocess.check_output(["rm", '-r', Config.RootPath + Folder + '/raw_data']).decode("utf-8")
        subprocess.check_output(["mkdir", Config.RootPath + Folder + '/raw_data']).decode("utf-8")
        subprocess.check_output(["unzip", Config.RootPath + Folder + '/' + FileName, '-d', Config.RootPath + Folder + '/raw_data']).decode("utf-8")
        Df_trn = pandas.read_csv(Config.RootPath + Folder + '/raw_data/train.csv')
        Df_trn['_data_'] = 'train'
        Df_tst = pandas.read_csv(Config.RootPath + Folder + '/raw_data/test.csv')
        Df_tst['_data_'] = 'test'
        Df = Df_trn.append(Df_tst, ignore_index=True)
        Df.to_csv(Config.RootPath + Folder+'/raw.csv',index=False)

    else:
        print('Invalid Source')
        Df = None
    return(Df)
