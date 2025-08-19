# visualizer.py
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def create_scatterplot_with_regression(dataframe: pd.DataFrame, x_col: str, y_col: str):
    """Creates a robust, general-purpose scatterplot with a regression line."""
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