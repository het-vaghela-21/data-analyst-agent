# tools.py
import pandas as pd
import numpy as np
import io
import sys
import re
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import base64
import requests
import duckdb

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes a DataFrame scraped from Wikipedia."""
    print("INFO: Cleaning and standardizing DataFrame.")
    clean_df = df.copy()
    
    # Handle multi-level column headers by joining them
    if isinstance(clean_df.columns, pd.MultiIndex):
        clean_df.columns = ['_'.join(map(str, col)).strip() for col in clean_df.columns]

    # Standardize column names to snake_case
    new_cols = {}
    for col in clean_df.columns:
        new_col = re.sub(r'[^a-zA-Z0-9]+', '_', str(col)).lower().strip('_')
        new_cols[col] = new_col
    clean_df.rename(columns=new_cols, inplace=True)

    # Convert all columns to numeric where possible, coercing errors
    for col in clean_df.columns:
        # Don't try to convert columns that seem purely textual
        if 'name' in col or 'title' in col or 'building' in col or 'city' in col or 'country' in col:
            continue
        # Remove citations like [5] before converting
        clean_df[col] = clean_df[col].astype(str).str.replace(r'\[.*?\]', '', regex=True)
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    print(f"INFO: Cleaned columns are: {clean_df.columns.tolist()}")
    return clean_df


def execute_python_code(code: str, df: pd.DataFrame = None) -> (str, pd.DataFrame):
    """Executes a string of Python code with a DataFrame."""
    output_buffer = io.StringIO()
    
    exec_globals = {
        "pd": pd, "np": np, "plt": plt, "requests": requests, "duckdb": duckdb,
        "df": df.copy() if df is not None else None,
        "io": io, "base64": base64, "re": re
    }

    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        exec(code, exec_globals)
    except Exception as e:
        print(f"Execution Error: {e}")
    finally:
        sys.stdout = original_stdout
    
    printed_output = output_buffer.getvalue().strip()
    final_df = exec_globals.get('df')

    return printed_output, final_df