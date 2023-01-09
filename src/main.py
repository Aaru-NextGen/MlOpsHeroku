# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml import process_data, inference
from starter.constants import ENCODER_NAME, LB, MODEL_NAME, load_pickle

app = FastAPI()

class InferanceReq(BaseModel):
    age: int = Field(alias="age")
    workclass: str = Field(alias='workclass')
    fnlgt: int = Field(alias='fnlgt')
    education: str = Field(alias='education')
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str = Field(alias='occupation')
    relationship: str = Field(alias='relationship')
    race: str = Field(alias='race')
    sex: str = Field(alias='sex')
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
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
        }

def preprocess_request_parameters(request):
    request = request.__dict__
    new_request = {}
    for key, value in request.items():
        key = key.replace("_", "-")
        new_request[key] = value
    return new_request

@app.get("/")
async def index():
    return 'Hello User. Thank you for visiting, Please use /infarance for predictions'

@app.post("/inferance")
async def get_inferance(request: InferanceReq):
    request = preprocess_request_parameters(request) 
    data = pd.DataFrame.from_dict([request])

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

    encoder = load_pickle(ENCODER_NAME)    
    lb = load_pickle(LB)
    model = load_pickle(MODEL_NAME)

    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, 
        training=False, encoder=encoder, lb=lb)

    pred = inference(model, X)[0]
    return '<=50K' if pred == 0 else '>50K'