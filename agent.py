# agent.py
import os
import json
import pandas as pd
from openai import OpenAI
from tools import execute_python_code

# Configure the client
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    
    # --- PLANNER PHASE ---
    # 1. Ask the AI to create a high-level plan
    planner_prompt = f"""
    You are a data analysis planner. Based on the user's request, create a concise, step-by-step plan of the actions needed.
    Do not write code. Provide a numbered list of simple actions in plain English.

    User Request: "{task_description}"
    """
    
    planner_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a data analysis planner."}, {"role": "user", "content": planner_prompt}],
        temperature=0,
    )
    plan_str = planner_response.choices[0].message.content
    print(f"INFO: Generated Plan:\n{plan_str}")

    # --- EXECUTOR PHASE ---
    dataframe_state = None # This will hold the data as it's processed
    final_answers = []
    
    # Loop through the plan and execute each step
    for step in plan_str.split('\n'):
        # Ignore empty lines or lines that are not part of the plan
        if not step.strip() or not step.strip()[0].isdigit():
            continue

        print(f"\n--- EXECUTING STEP: {step} ---")

        # For each step, ask the AI to write the specific Python code
        executor_prompt = f"""
        You are an expert Python programmer who writes simple, single-purpose code snippets.
        A pandas DataFrame is available under the variable name `df`. If `df` is None, you may need to load data first.
        The user's overall goal is: "{task_description}"
        Your current task is to write the Python code for this single step: "{step}"

        Your code must perform this step.
        - To scrape a table from a URL, use `df = pd.read_html(url, flavor='lxml')[0]`.
        - To run a DuckDB query on the remote dataset, use `con = duckdb.connect(); con.execute('INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;'); df = con.execute('YOUR_SQL_QUERY').df(); con.close()`.
        - To create a plot, save it to a base64 string and print it: `buf = io.BytesIO(); plt.savefig(buf, format='png'); print(f"data:image/png;base64,{{base64.b64encode(buf.getvalue()).decode('ascii')}}")`
        - The final result of any calculation step MUST be printed to the console using `print()`.
        - ONLY write the raw Python code. Do not add explanations.
        """
        
        executor_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python code-writing assistant."},
                {"role": "user", "content": executor_prompt}
            ],
            temperature=0,
        )
        code_to_execute = executor_response.choices[0].message.content.strip().replace('```python', '').replace('```', '')
        print(f"Generated Code:\n{code_to_execute}")

        # Execute the generated code using our powerful tool
        printed_output, dataframe_state = execute_python_code(code=code_to_execute, df=dataframe_state)
        
        print(f"Result from print(): {printed_output}")
        if printed_output:
            final_answers.append(printed_output)

    # Return the collected print statements from each step
    return final_answers