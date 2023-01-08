import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml import data, model

CLEAN_DATA = 'census.csv'
MODEL_NAME = 'rf_model.pkl'
ENCODER_NAME = 'encoder.pkl'
LB = 'lb.pkl'

project_dir = 'src'
data_dir = os.path.join(project_dir, 'data')
model_dir = os.path.join(project_dir, 'model')
input_data = pd.read_csv(os.path.join(data_dir, CLEAN_DATA))

def pickle_obj(obj, file_name):
    lb_path = os.path.join(model_dir, file_name)
    pickle.dump(obj, open(lb_path, 'wb'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(input_data, test_size=0.20, random_state=24)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

pickle_obj(encoder, ENCODER_NAME)
pickle_obj(lb, LB)

# Train and save a model.
rf_model = model.train_model(X_train, y_train)

pickle_obj(rf_model, MODEL_NAME)

X_test, y_test, encoder, lb = data.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


preds = model.inference(rf_model, X_test)

print('precision: {}, recall: {}, fbeta: {}'.format(
    *model.compute_model_metrics(y_test, preds)
))