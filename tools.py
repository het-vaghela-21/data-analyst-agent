# tools.py
import pandas as pd
import duckdb
import io
import base64
import re
import matplotlib.pyplot as plt
import numpy as np

def scrape_web_table(url: str):
    """Scrapes and cleans the Wikipedia table of highest-grossing films."""
    try:
        df = pd.read_html(url, flavor='lxml')[0]
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {
            next(c for c in df.columns if re.search(r'worldwide|gross', c, re.I)): 'Gross',
            next(c for c in df.columns if re.search(r'year', c, re.I)): 'Year',
            next(c for c in df.columns if re.search(r'rank', c, re.I)): 'Rank',
            next(c for c in df.columns if re.search(r'peak', c, re.I)): 'Peak',
            next(c for c in df.columns if re.search(r'title|film', c, re.I)): 'Title'
        }
        df.rename(columns=rename_map, inplace=True)
        def safe_to_number(s: any) -> float:
            s = str(s).lower()
            s = re.sub(r'[$,₹£]', '', s)
            s = s.replace(",", "")
            if 'billion' in s: return float(re.sub(r'[a-z]', '', s).strip()) * 1e9
            if 'million' in s: return float(re.sub(r'[a-z]', '', s).strip()) * 1e6
            try: return float(s)
            except (ValueError, TypeError): return np.nan
        df['Gross'] = df['Gross'].apply(safe_to_number)
        for col in ['Year', 'Rank', 'Peak']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Gross', 'Year', 'Rank', 'Peak'], inplace=True)
        return df
    except Exception as e:
        return f"Error scraping or cleaning table: {e}"

def run_duckdb_query(query: str):
    """Runs a LIVE DuckDB query against the remote S3 dataset."""
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

        plt.figure(figsize=(6, 4), dpi=120)
        plt.scatter(x, y, alpha=0.7, s=20)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m * x + b, 'r--')
        plt.title(f'Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        plt.close('all')

        if len(data) > 95000:
             return "Error: Plot image is too large (>100kB)."
        return f"data:image/png;base64,{data}"
    except Exception as e:
        return f"Plotting Error: {e}"