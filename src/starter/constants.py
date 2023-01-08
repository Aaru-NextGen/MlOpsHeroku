import os
import pickle

CLEAN_DATA = 'census.csv'
MODEL_NAME = 'rf_model.pkl'
ENCODER_NAME = 'encoder.pkl'
LB = 'lb.pkl'

project_dir = os.path.abspath(
    os.path.dirname(
        os.path.abspath(os.path.dirname(__file__)
        )
    )
)
model_dir = os.path.join(project_dir, 'model')
data_dir = os.path.join(project_dir, 'data')

def load_pickle(file_name, dir_path=model_dir):
    pickle_obj_path = os.path.join(dir_path, file_name)
    return pickle.load(open(pickle_obj_path, 'rb'))

def pickle_obj(obj, file_name, dir=model_dir):
    lb_path = os.path.join(dir, file_name)
    pickle.dump(obj, open(lb_path, 'wb'))