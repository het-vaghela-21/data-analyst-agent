import pandas as pd
import io
import duckdb
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import base64

def scrape_web_table(url: str) -> pd.DataFrame:
    """
    Scrapes the first table found at a given URL and returns it as a pandas DataFrame.
    """
    try:
        # lxml is a fast parser, good to specify
        tables = pd.read_html(url, flavor='lxml')
        print(f"Found {len(tables)} tables. Returning the first one.")
        return tables[0]
    except Exception as e:
        return f"Error scraping table: {e}"

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