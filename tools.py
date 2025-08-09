# tools.py
import pandas as pd
import numpy as np
import io
import sys
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import base64
import requests # Make sure requests is available
import duckdb # Make sure duckdb is available

# This is our single, powerful tool. It can scrape, query, analyze, plot, and more.
def execute_python_code(code: str, df: pd.DataFrame = None) -> (str, pd.DataFrame):
    """
    Executes a string of Python code.
    - Can take an optional pandas DataFrame as input, available as `df`.
    - Returns the captured print output (stdout) and the potentially modified DataFrame.
    """
    output_buffer = io.StringIO()
    
    # The environment in which the code will be executed
    # We include all necessary libraries for the AI to use.
    exec_globals = {
        "pd": pd,
        "np": np,

        "plt": plt,
        "requests": requests,
        "duckdb": duckdb,

        "df": df.copy() if df is not None else None,
        
        "io": io,
        "base64": base64
    }

    # Redirect stdout to the buffer to capture print() statements
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        # Execute the code in the prepared environment
        exec(code, exec_globals)
    except Exception as e:
        # If the code fails, print the error message
        print(f"Execution Error: {e}")
    finally:
        # Always restore the original stdout
        sys.stdout = original_stdout
    
    # Get any text that was printed
    printed_output = output_buffer.getvalue().strip()
    
    # Get the dataframe, which might have been created or modified by the code
    final_df = exec_globals.get('df')

    return printed_output, final_df