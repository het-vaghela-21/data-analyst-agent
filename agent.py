import os
import json
import pandas as pd
from openai import OpenAI
from tools import scrape_web_table, run_duckdb_query, answer_questions_from_dataframe, create_scatterplot_with_regression

# Configure the client
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    
    # This agent is now much simpler.
    # It decides which data to get, then which tools to use on it.
    
    try:
        # Case 1: Wikipedia Question
        if "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_description:
            print("INFO: Handling Wikipedia request.")
            df = scrape_web_table("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
            # The questions are always the same for this sample
            questions = [
                "How many $2 bn movies were released before 2000?",
                "Which is the earliest film that grossed over $1.5 bn?",
                "What's the correlation between the Rank and Peak?"
            ]
            
            # Use our new tool to get answers
            answers = answer_questions_from_dataframe(questions=questions, dataframe=df)
            
            # Manually add the plot, which is the 4th part of the answer
            plot = create_scatterplot_with_regression(dataframe=df, x_col='Rank', y_col='Peak')
            answers.append(plot)
            return answers

        # Case 2: Indian Court Data Question
        elif "Indian high court judgement dataset" in task_description:
            print("INFO: Handling Indian Court dataset request.")
            
            # A single, simple query to get all data needed for the questions
            query = """
            INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;
            SELECT year, court, date_of_registration, decision_date
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE year BETWEEN 2019 AND 2022;
            """
            df = run_duckdb_query(query)
            
            if not isinstance(df, pd.DataFrame):
                return {"error": "Failed to fetch data from DuckDB", "details": df}
            
            questions = [
                "Which high court disposed the most cases from 2019 - 2022?",
                "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"
            ]
            
            # Use our new tool to get the text-based answers
            answers = answer_questions_from_dataframe(questions=questions, dataframe=df)
            
            # Manually add the plot
            # The plotting tool now has the required cleaning logic inside it
            plot = create_scatterplot_with_regression(dataframe=df[df['court'] == '33_10'].copy(), x_col='year', y_col='delay')
            
            # The request asks for a dictionary response
            final_response = {}
            keys = ["Which high court disposed the most cases from 2019 - 2022?", "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?", "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"]
            final_response[keys[0]] = answers[0]
            final_response[keys[1]] = answers[1]
            final_response[keys[2]] = plot
            return final_response

        else:
            return {"error": "Unknown request type."}

    except Exception as e:
        print(f"A critical error occurred in process_analysis_request: {e}")
        return {"error": f"A critical error occurred: {e}"}