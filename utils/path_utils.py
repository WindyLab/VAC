import os
import shutil
from datetime import datetime

def join_and_create(path, folder):
    full_path = os.path.join(path, folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path,exist_ok=True)
    else:
        shutil.rmtree(full_path,ignore_errors=True)
        os.makedirs(full_path)
    return full_path

def get_training_time():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

def create_path_to_folder(checkpoint_dir):
    parent_folder = get_training_time()
    models_path = join_and_create(checkpoint_dir, parent_folder)
    return models_path

def path_to_model(models_path,epoch=None):
    if epoch is None:
        model_file_name = "Final_Model.pth"
    else:
        model_file_name = "Model_Ep{}.pth".format(epoch)
    return os.path.join(models_path, model_file_name)

def path_to_conf(conf_file):
    return os.path.join( 'confs', conf_file)