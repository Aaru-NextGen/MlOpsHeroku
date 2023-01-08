import json
import requests


data ={
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "While",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(
    'https://fastapitestservice.onrender.com/inferance',
    data=json.dumps(data)
)

print("Response Status Code : {}".format(response.status_code))
print("Response Inferance : {}".format(response.json()))