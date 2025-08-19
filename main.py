# main.py
import os
import shutil
import tempfile
import traceback
import json
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from data_processor import analyze_task

# --- CONFIG ---
load_dotenv()

app = FastAPI(title="Data Analyst Agent", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- API ENDPOINT ---
@app.post("/api/")
async def api_endpoint(files: List[UploadFile] = File(...)):
    
    # Find questions.txt and save all files to temp locations
    saved_files = {}
    questions_text = ""
    try:
        q_file = next((f for f in files if f.filename == "questions.txt"), None)
        if not q_file:
            raise HTTPException(status_code=400, detail="questions.txt is required.")
        
        questions_text = (await q_file.read()).decode("utf-8")
        
        for f in files:
            # We don't need to save files for the current tasks, but this makes it extensible
            pass

        # Run the analysis
        result = analyze_task(questions_text=questions_text, files={})
        
        # FastAPI can't directly serialize numpy types, so we convert to a JSON string then back
        # This is a robust way to handle complex data types
        json_compatible_result = json.loads(pd.Series([result]).to_json(orient='values'))[0]
        return JSONResponse(content=json_compatible_result)

    except Exception as e:
        print(f"A critical error occurred: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"error": str(e)})