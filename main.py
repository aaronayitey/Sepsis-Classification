from fastapi import FastAPI, Query, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd


pipeline = joblib.load('./sepsis_classification_pipeline.joblib')
encoder = joblib.load('./label_encoder.joblib')
model = joblib.load('./random_forest_model.joblib')

app = FastAPI()


class features(BaseModel):
    Age: int
    Body_Mass_Index_BMI: float
    Diastolic_Blood_Pressure: float
    Plasma_Glucose: float
    Triceps_Skinfold_Thickness: float
    Elevated_Glucose: float
    Diabetes_Pedigree_Function: float
    Insulin_Levels: float


@app.post("/predict")
async def predict_sepsis(item: features):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict()])

        input_data = pipeline.named_steps.preprocessor.transform(input_data)

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Decode predictions using the label encoder
        decoded_predictions = encoder.inverse_transform(predictions)

        return {"prediction": f'Patient is {decoded_predictions[0]}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
