# tools.py
import pandas as pd
import numpy as np
import io
import sys
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import base64

# This is our single, powerful tool. It can scrape, analyze, plot, and more.
def execute_python_code(code: str, df: pd.DataFrame = None) -> (str, pd.DataFrame):
    """
    Executes a string of Python code.
    - Can take an optional pandas DataFrame as input, available as `df`.
    - Returns the captured print output (stdout) and the modified DataFrame.
    """
    # Create a buffer to capture print statements
    buffer = io.StringIO()
    
    # The environment in which the code will be executed
    # We include all necessary libraries here.
    local_vars = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df.copy() if df is not None else None,
        "io": io,
        "base64": base64
    }

    # Redirect stdout to the buffer
    original_stdout = sys.stdout
    sys.stdout = buffer
    
    try:
        # Execute the code
        exec(code, local_vars)
    except Exception as e:
        # If there's an error, print it to the buffer and return
        print(e)
    finally:
        # Always restore stdout
        sys.stdout = original_stdout
    
    # Get the printed output
    printed_output = buffer.getvalue().strip()
    
    # Get the potentially modified DataFrame from the execution environment
    final_df = local_vars.get('df')

    return printed_output, final_df