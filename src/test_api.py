import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

WELCOME_MESSAGE = "Hello User. Thank you for visiting, Please use /infarance for predictions"

def test_home_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == WELCOME_MESSAGE

def test_inferance_endpoint_success_case_1():
    request_body = json.dumps({
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
    })
    res = client.post("/inferance", data=request_body)
    assert res.status_code == 200
    assert res.json() == "<=50K"

def test_inferance_endpoint_success_case_2():
    request_body = json.dumps({
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 707516,
        "education": "Bachelors",
        "education-num": 19,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "black",
        "sex": "Male",
        "capital-gain": 21704,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })
    res = client.post("/inferance", data=request_body)
    assert res.status_code == 200
    assert res.json() == ">50K"

def test_inferance_validation():
    request_body = json.dumps({
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
        # "native-country": "United-States"
    })
    res = client.post("/inferance", data=request_body)
    assert res.status_code == 422 # Unprocessable Entity


if __name__ == "__main__":
    test_home_endpoint()
    test_inferance_endpoint_success_case_1()
    test_inferance_endpoint_success_case_2()
    test_inferance_validation()