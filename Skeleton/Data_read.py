import subprocess
import pandas

import pip



def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


project = 'titanic'

folder = 'k_'+project
file_name = project+'.zip'
root_path = '/home/swayush/ML/'

out = subprocess.check_output(["ls",root_path]).decode("utf-8")
if folder not in out.split('\n'): out = subprocess.check_output(["mkdir",root_path+folder]).decode("utf-8")
out = subprocess.check_output(["ls"]).decode("utf-8")
if file_name in out.split('\n'): subprocess.check_output(["rm",file_name]).decode("utf-8")
out = subprocess.check_output(['kaggle','competitions','download',project]).decode("utf-8")
out = subprocess.check_output(['cp',file_name,root_path+folder]).decode("utf-8")
out = subprocess.check_output(["rm",file_name]).decode("utf-8")
out = subprocess.check_output(["ls",root_path+folder]).decode("utf-8")
if 'raw_data' in out.split('\n'): out = subprocess.check_output(["rm",'-r',root_path+folder+'/raw_data']).decode("utf-8")
out = subprocess.check_output(["mkdir",root_path+folder+'/raw_data']).decode("utf-8")
out = subprocess.check_output(["unzip",root_path+folder+'/'+file_name,'-d',root_path+folder+'/raw_data']).decode("utf-8")
df_trn = pd.read_csv(root_path+folder+'/raw_data/train.csv')
df_trn['_data_'] = 'train'
df_tst = pd.read_csv(root_path+folder+'/raw_data/test.csv')
df_tst['_data_'] = 'test'
df = df_trn.append(df_tst, ignore_index=True)
print(df.head())

# todo: Ascertain path exists
# todo: Download data if not present
# todo: Unzip and create final file

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






