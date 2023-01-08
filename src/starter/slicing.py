import os

import pandas as pd
from ml import process_data, compute_model_metrics, inference
from constants import data_dir, CLEAN_DATA, MODEL_NAME, load_pickle


def slice_metrics(model, data, slice_feature, slice_file, categorical_features=[]):
    """
    Output the performance of the model on slices of the data
    Inputs
    ------
    model : Machine learning model
        Trained machine learning model.
    data : pd.DataFrame
        Dataframe containing the features and label.
    slice_feature: str
        Name of the feature used to make slices (categorical features)
    slice_file: path
        path of the txt file where this slicing information is being save
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    Returns
    -------
    None
    """
    with open(slice_file, "a") as f:
        f.write("Performance on all unique slices on salary data based on {}\n".format(slice_feature))
        f.write("*****************************************************\n")
        X, y, _, _ = process_data(
            data, categorical_features=categorical_features, label="salary", training=True
        )
        preds = inference(model, X)

        for slice_value in data[slice_feature].unique():
            slice_index = data.index[data[slice_feature] == slice_value]
            
            f.write("{} = {}\n".format(slice_feature, slice_value))
            f.write('data size: {}\n'.format(len(slice_index)))
            f.write('precision: {}, recall: {}, fbeta: {}\n'.format(
                *compute_model_metrics(y[slice_index], preds[slice_index])
            ))
            f.write('-------------------------------------------------\n')

if __name__ == '__main__':
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
    data = pd.read_csv(os.path.join(data_dir, CLEAN_DATA))

    model = load_pickle(MODEL_NAME)

    file_path = os.path.join(os.path.dirname(__file__), "slice_output.txt") 
    if os.path.exists(file_path):
        os.remove(file_path)
    for feature in cat_features:
        slice_metrics(model, data, feature, file_path, categorical_features=cat_features)