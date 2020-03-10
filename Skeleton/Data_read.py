import subprocess

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
if 'raw_data' not in out.split('\n'): out = subprocess.check_output(["mkdir",root_path+folder+'/raw_data']).decode("utf-8")
#out = subprocess.check_output(["unzip",root_path+folder+'/'+file_name,'-d '+root_path+folder+'/raw_data').decode("utf-8")

#out = subprocess.check_output(["pwd"]).decode("utf-8")
#if file_name not in out.split(' '): print('out')
print('-d '+root_path+folder+'/raw_data')

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






