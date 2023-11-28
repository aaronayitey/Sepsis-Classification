from fastapi import FastAPI, Query, Request, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd

pipeline = joblib.load('./sepsis_classification_pipeline.joblib')
encoder = joblib.load('./label_encoder.joblib')
model = joblib.load('./random_forest_model.joblib')

app = FastAPI()

class Item(BaseModel):
    Age: float  = Query(..., description='Age')
    Body_Mass_Index_BMI: float  = Query(..., description='Body_Mass_Index_BMI')
    Diastolic_Blood_Pressure: float  = Query(..., description='Diastolic_Blood_Pressure')
    Plasma_Glucose: float  = Query(..., description='Plasma_Glucose')
    Triceps_Skinfold_Thickness: float  = Query(..., description='Triceps_Skinfold_Thickness')
    Elevated_Glucose: float  = Query(..., description='Elevated_Glucose')
    Diabetes_Pedigree_Function: float  = Query(..., description='Diabetes_Pedigree_Function')
    Insulin_Levels: float  = Query(..., description='Insulin_Levels')

@app.post("/predict")
async def predict_sepsis(item: Item):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict()])

        # Make predictions directly using the pipeline
        predictions = pipeline.predict(input_data)

        # Decode predictions using the label encoder
        decoded_predictions = encoder.inverse_transform(predictions)

        return {"prediction": f'Patient is {decoded_predictions[0]}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
