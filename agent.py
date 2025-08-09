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
    You are a data analysis planner. Based on the user's request, create a high-level, step-by-step plan of what needs to be done.
    Do not write code. Just provide a numbered list of actions in plain English.
    
    User Request: "{task_description}"
    """
    
    planner_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a data analysis planner."}, {"role": "user", "content": planner_prompt}],
        temperature=0,
    )
    plan = planner_response.choices[0].message.content
    print(f"INFO: Generated Plan:\n{plan}")

    # --- EXECUTOR PHASE ---
    # 2. Loop through the plan and execute each step
    dataframe_state = None # This will hold the data as it's processed
    final_answers = []
    
    for step in plan.split('\n'):
        if not step.strip() or not step.strip()[0].isdigit():
            continue

        print(f"\n--- EXECUTING STEP: {step} ---")

        # 3. For each step, ask the AI to write the specific Python code
        executor_prompt = f"""
        You are an expert Python programmer. You write clean, simple Python code to accomplish a single task.
        A pandas DataFrame is available under the variable name `df`. If `df` is None, you may need to load data first.
        The user's overall goal is: "{task_description}"
        The current step to accomplish is: "{step}"

        Write the Python code to perform this step.
        - To scrape a table from a URL, use `pd.read_html(url)[0]`.
        - To create a plot, use matplotlib and save it to a base64 string like this:
          `buf = io.BytesIO(); plt.savefig(buf, format='png'); print(f"data:image/png;base64,{{base64.b64encode(buf.getvalue()).decode('ascii')}}")`
        - The final result of a step should be printed to the console using `print()`.
        - Do not write any explanations, just the raw Python code.
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

        # 4. Execute the generated code using our powerful tool
        printed_output, dataframe_state = execute_python_code(code=code_to_execute, df=dataframe_state)
        
        print(f"Result: {printed_output}")
        if printed_output:
            final_answers.append(printed_output)

    # 5. Return the collected answers
    # Attempt to parse the final output into the requested format (list or dict)
    try:
        if "JSON array" in task_description:
            return [json.loads(a) if a.startswith('[') else a for a in final_answers]
        elif "JSON object" in task_description:
            # This is a simple heuristic; might need improvement
            final_dict = {}
            for a in final_answers:
                try:
                    final_dict.update(json.loads(a))
                except:
                    pass
            return final_dict
        return final_answers
    except:
        return final_answers