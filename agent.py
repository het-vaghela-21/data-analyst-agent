import os
import json
import pandas as pd
from openai import OpenAI
from tools import scrape_web_table, run_python_code_on_data, run_duckdb_query, create_scatterplot_with_regression

# Configure the client to use the correct AI Proxy URL
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    data_context = {}
    data_source_summary = "No specific data source identified." # Default value

    # Step 1: Load data from the appropriate source
    if "data.csv" in files:
        csv_file = files['data.csv']
        df = pd.read_csv(csv_file)
        data_context['df1'] = df
        column_names = df.columns.tolist()
        data_source_summary = f"Data is from 'data.csv', loaded as DataFrame 'df1'. The available columns are: {column_names}"

    elif "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_description:
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        df = scrape_web_table(url)
        data_context['df1'] = df
        # Make the agent data-aware by getting column names from the cleaned data
        column_names = df.columns.tolist()
        data_source_summary = f"Data scraped from Wikipedia, loaded as DataFrame 'df1'. The available columns are: {column_names}"

    elif "Indian high court judgement dataset" in task_description:
        data_source_summary = "Data is in the remote Indian high court dataset. Use 'run_duckdb_query'."
    
    # Step 2: Create a plan using the LLM with full context
    prompt = f"""
    You are a data analyst agent. Your task is to create a step-by-step plan in JSON format to answer the user's request.
    Use the exact column names provided in the Data Context.
    The user's final desired output format is described in the request (e.g., a JSON array of strings, a JSON object).
    Your plan should only use the tools provided.

    **User Request:**
    ---
    {task_description}
    ---
    
    **Data Context:**
    ---
    {data_source_summary}
    The main dataframe, if available, is named 'df1'.
    ---

    **Available Tools:**
    1. run_python_code_on_data(code: str, dataframe_name: str): Use this for all calculations, analysis, filtering, and correlations on data that is ALREADY loaded in a pandas DataFrame. The dataframe is available as 'df1' inside the code. The code MUST use a print() statement to return a result.
    2. run_duckdb_query(query: str): ONLY use this for querying large, remote datasets like the Indian court dataset when explicitly mentioned in the request. Do NOT use this for data already in a DataFrame.
    3. create_scatterplot_with_regression(dataframe_name: str, x_col: str, y_col: str): Generates a scatterplot from a DataFrame.

    **Response Format:**
    Your response must be a single JSON object with a key "plan", which is an array of steps. Each step is an object with "tool_name" and "args".
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst agent that creates JSON plans using the exact column names provided."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    plan_data = json.loads(response.choices[0].message.content)
    plan = plan_data.get("plan", [])

    if not plan:
        raise ValueError("LLM failed to generate a valid plan.")

    # Step 3: Execute the plan
    results = []
    for step in plan:
        tool_name = step['tool_name']
        args = step['args']
        
        print(f"Executing tool: {tool_name} with args: {args}")

        if tool_name == "run_python_code_on_data":
            df_name = args.pop('dataframe_name')
            result = run_python_code_on_data(dataframe=data_context[df_name], **args)
        
        elif tool_name == "run_duckdb_query":
            result_df = run_duckdb_query(**args)
            if isinstance(result_df, pd.DataFrame):
                result = result_df.to_string()
                data_context['duckdb_result'] = result_df
            else:
                result = result_df
        
        elif tool_name == "create_scatterplot_with_regression":
            df_name = args.pop('dataframe_name')
            result = create_scatterplot_with_regression(dataframe=data_context[df_name], **args)
        
        else:
            result = f"Error: Unknown tool '{tool_name}'"
            
        results.append(result)
    
    # Step 4: Format and return the final response
    if "respond with a JSON object" in task_description.lower():
        try:
            question_keys = [line.split(':')[0].strip().strip('"') for line in task_description.split('\n') if ':' in line and '?' in line]
            return {key: res for key, res in zip(question_keys, results)}
        except Exception:
            return {"error": "Failed to format response as object.", "results": results}
    else:
        return results