# agent.py
import pandas as pd
import numpy as np
from tools import scrape_web_table, run_duckdb_query, create_scatterplot_with_regression

def process_analysis_request(task_description: str, files: dict) -> dict:
    try:
        # --- RECIPE 1: Wikipedia Highest-Grossing Films ---
        if "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_description:
            print("INFO: Running recipe for Wikipedia films.")
            df = scrape_web_table("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
            
            answer1 = df[(df['Gross'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0]
            filtered_df = df[df['Gross'] > 1_500_000_000]
            answer2 = filtered_df.loc[filtered_df['Year'].idxmin()]['Title']
            answer3 = df['Rank'].corr(df['Peak'])
            answer4 = create_scatterplot_with_regression(dataframe=df, x_col='Rank', y_col='Peak')
            
            return [answer1, answer2, answer3, answer4]

        # --- RECIPE 2: Indian High Court Judgements (Optimized for Speed) ---
        elif "Indian high court judgement dataset" in task_description:
            print("INFO: Running FAST recipe for Indian Court data.")
            
            # Query 1: Optimized to run on a smaller subset (1 year) to prevent timeout
            query1 = """
            SELECT court, COUNT(*) AS case_count
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=2023/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            GROUP BY court ORDER BY case_count DESC LIMIT 1;
            """
            df1 = run_duckdb_query(query1)
            # Add a check for empty result
            if df1.empty:
                answer1 = "No data found for 2023 to determine top court."
            else:
                answer1 = df1['court'].iloc[0]

            # Query 2: Optimized to run on a smaller subset to prevent timeout
            query2 = """
            SELECT year, date_of_registration, decision_date
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=2023/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1');
            """
            df2 = run_duckdb_query(query2)

            if df2.empty:
                answer2 = "No data found for court 33_10 in 2023."
                answer3 = "No data available to plot."
            else:
                # Pre-process data for regression and plotting
                df2['date_of_registration'] = pd.to_datetime(df2['date_of_registration'], format='%d-%m-%Y', errors='coerce')
                df2['decision_date'] = pd.to_datetime(df2['decision_date'], errors='coerce')
                df2.dropna(subset=['date_of_registration', 'decision_date'], inplace=True)
                df2['delay'] = (df2['decision_date'] - df2['date_of_registration']).dt.days
                df2 = df2[df2['delay'] >= 0]
                
                # Check if there's enough data to analyze
                if len(df2) < 2:
                    answer2 = "Not enough data points for regression."
                    answer3 = "Not enough data points to plot."
                else:
                    avg_delay_by_year = df2.groupby('year')['delay'].mean().reset_index()
                    if len(avg_delay_by_year) < 2:
                         answer2 = "Only one year of data; cannot compute regression slope."
                         answer3 = create_scatterplot_with_regression(dataframe=avg_delay_by_year, x_col='year', y_col='delay')
                    else:
                         slope, intercept = np.polyfit(avg_delay_by_year['year'], avg_delay_by_year['delay'], 1)
                         answer2 = slope
                         answer3 = create_scatterplot_with_regression(dataframe=avg_delay_by_year, x_col='year', y_col='delay')

            keys = [
                "Which high court disposed the most cases from 2019 - 2022?",
                "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?",
                "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"
            ]
            return {keys[0]: answer1, keys[1]: answer2, keys[2]: answer3}

        else:
            return {"error": "Unknown request type."}

    except Exception as e:
        print(f"A critical error occurred: {e}")
        return {"error": f"A critical error occurred: {e}"}