from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from fastapi.encoders import jsonable_encoder
from torch.utils.data import DataLoader
import torch
from typing import List, Optional
from pytorch import get_device, split_sets_random, pop_target

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'NN is all ready to go!'

def format_features(brewery_name: str,	review_overall: int, review_aroma: int,
                    review_appearance: int, review_palate:int, review_taste:int):
  return {
        'brewery_name': [brewery_name],
        'review_overall': [overall],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste]
    }

@app.get("/beers/type/")
def predict(brewery_name: str,	review_overall: int, review_aroma: int,
            review_appearance: int, review_palate:int, review_taste:int):
    features = format_features(brewery_name,overall,aroma,appearance,palate,taste)
    obs = pd.DataFrame(features)
    pred = nn_pipe.predict(obs)
    return JSONResponse(pred.tolist())

@app.get("/model/architecture/")
async def print_model():
    model = get_device()
    return {'model archtecture': str(model)}




