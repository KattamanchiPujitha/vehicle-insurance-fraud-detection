# main.py
# FastAPI backend for the Insurance Claim Fraud Detection project.
# It loads the saved model and gives a fraud risk score for one claim.

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent          # the folder where this file is saved


# 1. create the app
app = FastAPI(title="Insurance Claim Fraud Detection API")


# 2. load the saved model (same idea as heart.joblib in the heart disease project)
saved = joblib.load(BASE_DIR / "fraud_model.joblib")
model = saved["model"]              # the trained model (converter + Gradient Boosting)
features = saved["features"]        # the 12 column names, in the same order as in training
threshold = saved["threshold"]      # a score at or above this is called "Fraud"


# 3. the data the API expects = the 12 values typed on the HTML page
class ClaimData(BaseModel):
    PolicyType: str
    Fault: str
    AddressChange_Claim: str
    Deductible: int
    VehiclePrice: str
    PastNumberOfClaims: str
    Make: str
    MonthClaimed: str
    Age: int
    Month: str
    AccidentArea: str
    NumberOfSuppliments: str


# 4. home page: when someone opens the website, send them index.html
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


# 5. prediction: the HTML page sends the 12 values here and gets the score back
@app.post("/predict")
def predict(claim: ClaimData):
    # one-row table with the columns in the same order as in training
    row = pd.DataFrame([claim.model_dump()])[features]

    # the model gives a fraud score between 0 and 1
    score = float(model.predict_proba(row)[0][1])

    # decide the result
    if score >= threshold:
        level = "High"          # send for investigation
    elif score >= threshold * 0.6:
        level = "Medium"        # look a little closer
    else:
        level = "Low"

    return {
        "risk_score": round(score * 100, 1),        # 0 to 100
        "threshold": round(threshold * 100, 1),
        "prediction": "Fraud" if score >= threshold else "Not Fraud",
        "risk_level": level,
    }
