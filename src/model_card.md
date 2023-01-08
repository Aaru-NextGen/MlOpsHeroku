# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A ``RandomForestClassifier`` from the ``Scikit-Learn`` module is employed.
All of the hyperparameters are the classifier's default parameter. 

## Intended Use
Based on the data provided by the employees, the model is used to categorise the wage of employees into two ranges: ``=50K`` and ``>50K``.


Users may use this model to forecast the kind of compensation by applying it to information about their employees in the provided manner. 

## Training Data
A publicly accessible dataset from the Census Bureau is utilised to train and assess the model.


There are enough data to train a well performing model since the dataset comprises a large number of features and a fair number of instances.


Categorical aspects of the data are encoded using ``OneHotEncoder`` for both training and assessment, and the target is converted using ``LabelBinarizer`` 

## Evaluation Data
After preprocessing, the original dataset is divided into training and evaluation data, with evaluation data size set at 20%. 

## Metrics

- precision: 0.729
- recall: 0.627
- fbeta: 0.674

## Caveats and Recommendations
The show was prepared on information of individuals generally from the USA. Subsequently it isn't prescribed to utilize the demonstrate to anticipate the compensation sort for individuals from other locales within the world, which might have exceptionally diverse highlight conveyances
