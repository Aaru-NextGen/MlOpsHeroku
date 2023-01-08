import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import ml
from constants import CLEAN_DATA, MODEL_NAME, ENCODER_NAME, LB, data_dir, pickle_obj, project_dir


input_data = pd.read_csv(os.path.join(data_dir, CLEAN_DATA))
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

X_train, y_train, encoder, lb = ml.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

pickle_obj(encoder, ENCODER_NAME)
pickle_obj(lb, LB)

# Train and save a model.
rf_model = ml.train_model(X_train, y_train)

pickle_obj(rf_model, MODEL_NAME)

X_test, y_test, encoder, lb = ml.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


preds = ml.inference(rf_model, X_test)

print('precision: {}, recall: {}, fbeta: {}'.format(
    *ml.compute_model_metrics(y_test, preds)
))