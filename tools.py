import pandas as pd
import io
import duckdb
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import base64

# tools.py

import pandas as pd
# (Keep all your other functions like run_python_code_on_data, etc.)

def scrape_web_table(url: str) -> pd.DataFrame:
    """
    Scrapes the first table at a URL, cleans it for analysis, and returns a pandas DataFrame.
    """
    try:
        tables = pd.read_html(url, flavor='lxml')
        df = tables[0]

        # --- DATA CLEANING STEPS ---
        # 1. Rename columns for easier access by the AI
        if 'Worldwide gross' in df.columns:
            df.rename(columns={'Worldwide gross': 'Gross'}, inplace=True)
        if 'Reference(s)' in df.columns:
            df.rename(columns={'Reference(s)': 'Reference'}, inplace=True)


        # 2. Clean the 'Gross' column (e.g., "$2.9 billion" -> 2900000000)
        if 'Gross' in df.columns:
            df['Gross'] = df['Gross'].astype(str).str.replace(r'\$', '', regex=True).str.replace(r'\[.*?\]', '', regex=True)
            
            def convert_gross_to_number(gross_str):
                gross_str = gross_str.lower()
                if 'billion' in gross_str:
                    return float(gross_str.replace('billion', '')) * 1_000_000_000
                elif 'million' in gross_str:
                    return float(gross_str.replace('million', '')) * 1_000_000
                else:
                    return pd.to_numeric(gross_str.replace(',', ''), errors='coerce')
            
            df['Gross'] = df['Gross'].apply(convert_gross_to_number)

        # 3. Clean other key columns to ensure they are purely numeric
        for col in ['Year', 'Rank', 'Peak']:
            if col in df.columns:
                # errors='coerce' will turn any non-numeric value into a blank (NaN)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Drop any rows that have blank values in our key columns after cleaning
        df.dropna(subset=['Gross', 'Year', 'Rank', 'Peak'], inplace=True)
        
        print("Data scraped and cleaned successfully.")
        return df
        
    except Exception as e:
        return f"Error scraping or cleaning table: {e}"

# (Your other functions like run_python_code_on_data, run_duckdb_query, etc. stay below)
def run_python_code_on_data(code: str, dataframe: pd.DataFrame) -> str:
    """
    Executes a snippet of Python code with a DataFrame named 'df' available.
    Returns the printed output of the code.
    """
    buffer = io.StringIO()
    local_vars = {'df1': dataframe, 'pd': pd}

    with redirect_stdout(buffer):
        try:
            exec(code, {'__builtins__': __builtins__}, local_vars)
        except Exception as e:
            print(f"Execution Error: {e}")

    return buffer.getvalue().strip()

def run_duckdb_query(query: str) -> pd.DataFrame:
    """
    Executes a DuckDB query and returns the result as a pandas DataFrame.
    """
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        result_df = con.execute(query).fetchdf()
        return result_df
    except Exception as e:
        return f"DuckDB Error: {e}"

def create_scatterplot_with_regression(dataframe: pd.DataFrame, x_col: str, y_col: str) -> str:
    """
    Generates a scatterplot with a dotted red regression line.
    Returns a base64 encoded data URI of the plot image.
    """
    try:
        df_copy = dataframe.copy()
        df_copy[x_col] = pd.to_numeric(df_copy[x_col], errors='coerce')
        df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
        df_copy.dropna(subset=[x_col, y_col], inplace=True)

        if df_copy.empty:
            return "Error: No data available for plotting after cleaning."

        x = df_copy[x_col]
        y = df_copy[y_col]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6)

        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, 'r--', label=f'y={m:.2f}x+{b:.2f}')

        plt.title(f'Scatter Plot of {y_col} vs. {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        if len(image_base64) > 95000:
            return "Error: Generated image is too large."

        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"Plotting Error: {e}"