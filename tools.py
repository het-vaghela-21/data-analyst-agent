import os
import pandas as pd
import duckdb
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

# This file needs its own client for the new tool
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def scrape_web_table(url: str):
    # This function remains the same as our last good version
    try:
        df = pd.read_html(url, flavor='lxml')[0]
        if 'Worldwide gross' in df.columns:
            df.rename(columns={'Worldwide gross': 'Gross'}, inplace=True)
        if 'Reference(s)' in df.columns:
            df.rename(columns={'Reference(s)': 'Reference'}, inplace=True)
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
        for col in ['Year', 'Rank', 'Peak']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Gross', 'Year', 'Rank', 'Peak'], inplace=True)
        return df
    except Exception as e:
        return f"Error scraping or cleaning table: {e}"

def run_duckdb_query(query: str):
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        # DuckDB requires these extensions for S3 Parquet access
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
        result_df = con.execute(query).fetchdf()
        return result_df
    except Exception as e:
        return f"DuckDB Error: {e}"

def answer_questions_from_dataframe(questions: list, dataframe: pd.DataFrame):
    """Our new powerhouse tool."""
    try:
        # Get the first 5 rows of the dataframe as a sample for the prompt
        df_head_str = dataframe.head().to_string()
        column_names = dataframe.columns.tolist()

        prompt = f"""
        You are an expert Python data analyst. Given a pandas DataFrame and a list of questions, write a single Python script that calculates the answers.
        
        The DataFrame is available as a variable named `df`.
        The available columns are: {column_names}
        
        Here is a sample of the data:
        {df_head_str}

        **IMPORTANT INSTRUCTIONS:**
        - Your script must answer each question.
        - The final line of your script MUST be a print statement of a Python list or dictionary containing the answers in the correct order.
        - For date calculations, use `pd.to_datetime`. `date_of_registration` is in 'DD-MM-YYYY' format.
        - Do not include any explanations or surrounding text, only the raw Python code.

        **Questions to Answer:**
        {questions}
        
        **Your Python Script:**
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python code-writing assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        
        code_to_run = response.choices[0].message.content.strip().replace('```python', '').replace('```', '')
        
        # Prepare a buffer to capture the print output
        output_buffer = io.StringIO()
        
        # Prepare the execution environment
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        
        exec_globals = {'df': dataframe, 'pd': pd, 'np': np}
        
        try:
            exec(code_to_run, exec_globals)
        finally:
            # Restore stdout
            sys.stdout = original_stdout

        # Get the captured output
        result_str = output_buffer.getvalue().strip()
        # The output should be a string representation of a list or dict, so we evaluate it
        return eval(result_str)

    except Exception as e:
        return f"Error in answer_questions_from_dataframe: {e}"

def create_scatterplot_with_regression(dataframe: pd.DataFrame, x_col: str, y_col: str):
    # This function remains the same
    try:
        df_copy = dataframe.copy()
        # Data cleaning for plotting
        df_copy['date_of_registration'] = pd.to_datetime(df_copy['date_of_registration'], format='%d-%m-%Y', errors='coerce')
        df_copy['decision_date'] = pd.to_datetime(df_copy['decision_date'], errors='coerce')
        df_copy.dropna(subset=['date_of_registration', 'decision_date'], inplace=True)
        df_copy['delay'] = (df_copy['decision_date'] - df_copy['date_of_registration']).dt.days
        df_copy = df_copy[df_copy['delay'] >= 0]
        
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