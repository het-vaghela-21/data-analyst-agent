# tools.py
import pandas as pd
import duckdb
import io
import base64
import matplotlib.pyplot as plt
import numpy as np

def scrape_web_table(url: str):
    """Scrapes and cleans the Wikipedia table of highest-grossing films."""
    try:
        df = pd.read_html(url, flavor='lxml')[0]
        if 'Worldwide gross' in df.columns:
            df.rename(columns={'Worldwide gross': 'Gross'}, inplace=True)
        if 'Gross' in df.columns:
            df['Gross'] = df['Gross'].astype(str).str.replace(r'\$', '', regex=True).str.replace(r'\[.*?\]', '', regex=True)
            def convert_gross_to_number(s):
                s = s.lower()
                if 'billion' in s: return float(s.replace('billion', '')) * 1e9
                if 'million' in s: return float(s.replace('million', '')) * 1e6
                return pd.to_numeric(s.replace(',', ''), errors='coerce')
            df['Gross'] = df['Gross'].apply(convert_gross_to_number)
        for col in ['Year', 'Rank', 'Peak', 'Title']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') if col != 'Title' else df[col]
        df.dropna(subset=['Gross', 'Year', 'Rank', 'Peak'], inplace=True)
        return df
    except Exception as e:
        return f"Error scraping or cleaning table: {e}"

def run_duckdb_query(query: str):
    """Runs a live DuckDB query against the remote S3 dataset."""
    try:
        print(f"INFO: Running live DuckDB query: {query}")
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
        result_df = con.execute(query).fetchdf()
        return result_df
    except Exception as e:
        return f"DuckDB Error: {e}"

def create_scatterplot_with_regression(dataframe: pd.DataFrame, x_col: str, y_col: str):
    """Creates a general-purpose scatterplot with a regression line."""
    try:
        df_copy = dataframe.copy()
        df_copy[x_col] = pd.to_numeric(df_copy[x_col], errors='coerce')
        df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
        df_copy.dropna(subset=[x_col, y_col], inplace=True)

        if df_copy.empty: return "Error: No valid data to plot."

        x = df_copy[x_col]
        y = df_copy[y_col]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, 'r--')
        plt.title(f'Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"Plotting Error: {e}"