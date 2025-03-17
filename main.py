import os
from typing import List
from fastapi import FastAPI, File, UploadFile
import shutil
from model import predict


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

@app.post("/predict/")
async def predict_skin_disease(files: List[UploadFile] = File(...)):
    """Handles multiple file uploads and returns predictions"""
    results = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        predicted_disease = predict(file_path)
        
        results.append({"filename": file.filename, "disease": predicted_disease})

    return {"predictions": results}
