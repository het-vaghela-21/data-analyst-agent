# agent.py
import os
import json
import pandas as pd
from openai import OpenAI
from tools import execute_python_code, clean_dataframe

# Configure the client
client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPROXY_TOKEN")
)

def process_analysis_request(task_description: str, files: dict) -> dict:
    
    # --- PLANNER PHASE ---
    planner_prompt = f"""
    You are a data analysis planner. Based on the user's request, create a concise, step-by-step plan.
    Do not write code. Provide a numbered list of simple actions in plain English.
    The first step should always be to load the data from the specified URL.

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
    dataframe_state = None
    final_answers = []
    
    plan_steps = [step for step in plan_str.split('\n') if step.strip() and step.strip()[0].isdigit()]

    for i, step in enumerate(plan_steps):
        print(f"\n--- EXECUTING STEP: {step} ---")

        # Create the prompt for the code-writing AI
        executor_prompt = f"""
        You are an expert Python programmer who writes simple code snippets to accomplish a single task.
        A pandas DataFrame is available as the variable `df`.
        The user's overall goal is: "{task_description}"
        Your current task is to write the Python code for this single step: "{step}"
        """
        # For analysis steps (after data is loaded), provide the exact column names
        if dataframe_state is not None:
            executor_prompt += f"\nThe DataFrame `df` has the following columns, use them exactly: {dataframe_state.columns.tolist()}"

        executor_prompt += """
        \n**Instructions:**
        - To scrape a table, use the code: `url = "YOUR_URL"; df = pd.read_html(url, flavor='lxml')[0]`
        - To create a plot, save it to a base64 string and print it, for example: `buf = io.BytesIO(); plt.savefig(buf, format='png'); print(f"data:image/png;base64,{{base64.b64encode(buf.getvalue()).decode('ascii')}}")`
        - The final result of any calculation step MUST be printed to the console using `print()`.
        - ONLY write the raw Python code.
        """
        
        executor_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a Python code-writing assistant."}, {"role": "user", "content": executor_prompt}],
            temperature=0,
        )
        code_to_execute = executor_response.choices[0].message.content.strip().replace('```python', '').replace('```', '')
        print(f"Generated Code:\n{code_to_execute}")

        # Execute the generated code
        printed_output, dataframe_state = execute_python_code(code=code_to_execute, df=dataframe_state)
        
        # After the first step (loading data), automatically clean the DataFrame
        if i == 0 and dataframe_state is not None:
            dataframe_state = clean_dataframe(dataframe_state)

        print(f"Result from print(): {printed_output}")
        if printed_output:
            final_answers.append(printed_output)

    # Return the collected answers
    return final_answers