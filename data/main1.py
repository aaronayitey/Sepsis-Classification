from fastapi import FastAPI
import uvicorn
import json
from pydantic import BaseModel
import joblib
import json
import imblearn
import pandas as pd
from xgboost import XGBClassifier
from fastapi import FastAPI, Query, Request, HTTPException



app = FastAPI()

# loading my best model with joblib 
pipeline = joblib.load('./sepsis_classification_pipeline.joblib')
encoder = joblib.load('./label_encoder.joblib')
model = joblib.load('./random_forest_model.joblib')

###
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sepsis Prediction using FastAPI"}

def classify(prediction):
    if prediction == 0:
        return "Patient does not have sepsis"
    else:
        return "Patient has sepsis"
        
@app.post("/predict/")
async def predict_sepsis(
    request: Request,
    Age: float  = Query(..., description='Age'),
    Body_Mass_Index_BMI: float  = Query(..., description='Body Mass Index BMI'),
    Diastolic_Blood_Pressure: float  = Query(..., description='Diastolic Blood Pressure'),
    Plasma_Glucose: float  = Query(..., description='Plasma Glucose'),
    Triceps_Skinfold_Thickness: float  = Query(..., description='Triceps Skinfold Thickness'),
    Elevated_Glucose: float  = Query(..., description='Elevated Glucose'),
    Diabetes_Pedigree_Function: float  = Query(..., description='Diabetes Pedigree Function'),
    Insulin_Levels: float  = Query(..., description='Insulin Levels')
):
    input_data = [Age, Body_Mass_Index_BMI, Diastolic_Blood_Pressure, Plasma_Glucose, \
                  Triceps_Skinfold_Thickness, Elevated_Glucose, Diabetes_Pedigree_Function, Insulin_Levels]

    input_df = pd.DataFrame([input_data], columns=[
        'Age', 'Body_Mass_Index_BMI', 'Diastolic_Blood_Pressure', 'Plasma_Glucose', \
                  'Triceps_Skinfold_Thickness', 'Elevated_Glucose', 'Diabetes_Pedigree_Function', 'Insulin_Levels'
    ])

    pred = pipeline.predict(input_df)
    output = classify(pred[0])

    response = {
        "prediction": output
    }

    return response

    # Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)