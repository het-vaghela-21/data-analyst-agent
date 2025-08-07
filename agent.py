import os
import json
import pandas as pd
from openai import OpenAI
from tools import scrape_web_table, run_python_code_on_data, run_duckdb_query, create_scatterplot_with_regression

# Configure the client to use the AI Proxy
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    data_context = {}

    # Simple router to load initial data
    if "data.csv" in files:
        csv_file = files['data.csv']
        df = pd.read_csv(csv_file)
        data_context['df1'] = df
        data_source_summary = "Data is from 'data.csv', loaded as DataFrame 'df1'."
    elif "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_description:
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        df = scrape_web_table(url)
        data_context['df1'] = df
        data_source_summary = f"Data scraped from Wikipedia, loaded as DataFrame 'df1'."
    elif "Indian high court judgement dataset" in task_description:
        data_source_summary = "Data is in the remote Indian high court dataset. Use 'run_duckdb_query'."
    else:
        data_source_summary = "No specific data source identified."

    # Define the plan-generation prompt
    prompt = f"""
    You are a data analyst agent. Your task is to create a step-by-step plan in JSON format to answer the user's request.
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
    1. run_python_code_on_data(code: str, dataframe_name: str): Executes python code on a named dataframe. The dataframe is available as 'df'. The code MUST use a print() statement to return a result.
    2. run_duckdb_query(query: str): Executes a DuckDB SQL query. Returns a dataframe.
    3. create_scatterplot_with_regression(dataframe_name: str, x_col: str, y_col: str): Generates a scatterplot.

    **Response Format:**
    Your response must be a single JSON object with a key "plan", which is an array of steps. Each step is an object with "tool_name" and "args".
    The arguments in "args" MUST match the function signatures of the tools.

    Example Plan:
    {{
      "plan": [
        {{
          "tool_name": "run_python_code_on_data",
          "args": {{
            "dataframe_name": "df1",
            "code": "print(df[df['Year'] < 2000].shape[0])"
          }}
        }},
        {{
          "tool_name": "create_scatterplot_with_regression",
          "args": {{
            "dataframe_name": "df1",
            "x_col": "Rank",
            "y_col": "Peak"
          }}
        }}
      ]
    }}
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
            result = result_df.to_string() 
            data_context['duckdb_result'] = result_df
        elif tool_name == "create_scatterplot_with_regression":
            df_name = args.pop('dataframe_name')
            result = create_scatterplot_with_regression(dataframe=data_context[df_name], **args)
        else:
            result = f"Error: Unknown tool '{tool_name}'"

        results.append(result)

    # Simple formatting based on request type
    if "respond with a JSON object" in task_description.lower():
        try:
            question_keys = [line.split(':')[0].strip().strip('"') for line in task_description.split('\n') if ':' in line and '?' in line]
            return {key: res for key, res in zip(question_keys, results)}
        except Exception:
            return {"error": "Failed to format response as object.", "results": results}
    else:
        return results