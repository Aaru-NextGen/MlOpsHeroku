import os
import pickle

from constants import model_dir

def load_pickle(file_name, dir_path=model_dir):
    pickle_obj_path = os.path.join(dir_path, file_name)
    return pickle.load(open(pickle_obj_path, 'rb'))

def pickle_obj(obj, file_name, dir=model_dir):
    lb_path = os.path.join(dir, file_name)
    pickle.dump(obj, open(lb_path, 'wb'))