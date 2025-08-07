import os
import json
import pandas as pd
from openai import OpenAI
from tools import scrape_web_table, run_python_code_on_data, run_duckdb_query, create_scatterplot_with_regression

# Configure the client
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    data_context = {}
    data_source_summary = "No specific data source identified."

    # Step 1: Load initial data if provided
    if "data.csv" in files:
        df = pd.read_csv(files['data.csv'])
        data_context['df1'] = df
        column_names = df.columns.tolist()
        data_source_summary = f"Data is from 'data.csv', loaded as DataFrame 'df1'. The available columns are: {column_names}"
    elif "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_description:
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        df = scrape_web_table(url)
        data_context['df1'] = df
        column_names = df.columns.tolist()
        data_source_summary = f"Data scraped from Wikipedia, loaded as DataFrame 'df1'. The available columns are: {column_names}"
    elif "Indian high court judgement dataset" in task_description:
        data_source_summary = "Data is in the remote Indian high court dataset. Use 'run_duckdb_query'."
    
    # Step 2: Create a plan using the LLM
    prompt = f"""
    You are a data analyst agent. Create a JSON plan to answer the user's request.
    Use the exact column names provided in the Data Context if available.

    **User Request:**
    ---
    {task_description}
    ---
    
    **Data Context:**
    ---
    {data_source_summary}
    ---

    **Available Tools:**
    1. run_python_code_on_data(code: str, dataframe_name: str): Use for analysis on a DataFrame. The dataframe_name can be 'df1' (if loaded initially) or 'query_result' (if from a duckdb query). The dataframe is available as 'df1' inside the code. The code MUST use a print() statement.
    2. run_duckdb_query(query: str): Use for querying remote datasets. The result is saved as a DataFrame named 'query_result' for the next step.
    3. create_scatterplot_with_regression(dataframe_name: str, x_col: str, y_col: str): Generates a plot. Can use 'df1' or 'query_result' as the dataframe_name.

    **Response Format:**
    Your response must be a single JSON object with a key "plan", which is an array of steps. Each step must have "tool_name" and "args".
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst agent that creates JSON plans."},
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
        args = step.get('args', {})
        
        print(f"Executing tool: {tool_name} with args: {args}")

        if tool_name == "run_python_code_on_data":
            # This logic automatically uses the result of a previous query if the AI forgets
            if 'dataframe_name' not in args and 'query_result' in data_context:
                print("INFO: 'dataframe_name' missing. Auto-using 'query_result'.")
                args['dataframe_name'] = 'query_result'
            
            df_name = args.pop('dataframe_name')
            result = run_python_code_on_data(dataframe=data_context[df_name], **args)
        
        elif tool_name == "run_duckdb_query":
            result_df = run_duckdb_query(**args)
            if isinstance(result_df, pd.DataFrame):
                data_context['query_result'] = result_df
                result = f"Query successful, result stored in 'query_result'. Rows: {len(result_df)}"
            else:
                result = result_df
        
        elif tool_name == "create_scatterplot_with_regression":
            df_name = args.pop('dataframe_name')
            result = create_scatterplot_with_regression(dataframe=data_context[df_name], **args)
        
        else:
            result = f"Error: Unknown tool '{tool_name}'"
            
        results.append(result)
    
    # Step 4: Format and return final response
    final_results = [res for res in results if not isinstance(res, str) or not res.startswith("Query successful")]

    if "respond with a JSON object" in task_description.lower():
        try:
            json_block_in_prompt = task_description.split('```json')[1].split('```')[0]
            question_keys = json.loads(json_block_in_prompt).keys()
            return {key: res for key, res in zip(question_keys, final_results)}
        except Exception as e:
            print(f"Error formatting JSON object response: {e}")
            return {"error": "Failed to format response as object.", "results": final_results}
    else:
        return final_results