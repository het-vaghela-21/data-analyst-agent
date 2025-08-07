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
    # Initialize the agent's state
    data_context = {}
    history = []
    max_steps = 7 # Set a limit to prevent infinite loops

    # --- Main Agent Loop ---
    for i in range(max_steps):
        print(f"\n--- Step {i+1} ---")
        
        # 1. REASON: Create a prompt with the history and ask for the NEXT action
        prompt = f"""
        You are a data analyst agent. Your goal is to answer the user's request by executing a sequence of tool calls.
        You will work step-by-step. I will provide you with the user's request and a history of the actions you have already taken and their results.
        Decide on the single next tool to call to move closer to the final answer.

        **User's Full Request:**
        ---
        {task_description}
        ---

        **History of Actions Taken So Far:**
        ---
        {history}
        ---

        **Available Tools:**
        1. run_python_code_on_data(code: str, dataframe_name: str): For analysis on a DataFrame (e.g., 'df1', 'query_result_1', etc.). The dataframe is available as 'df1' in the code.
        2. run_duckdb_query(query: str): For querying remote datasets. The result is a DataFrame.
        3. create_scatterplot_with_regression(dataframe_name: str, x_col: str, y_col: str): Generates a plot.
        4. finish(final_answers: list or dict): Call this FINAL tool when you have all the answers. The argument should be a JSON list or object that directly answers the user's request.

        **Your Task:**
        Respond with a single JSON object representing the next tool call. Example: {{"tool_name": "run_duckdb_query", "args": {{"query": "SELECT * FROM ..."}}}}
        If you have all the necessary information to answer the user's request, call the "finish" tool.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst agent that decides the next action to take."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            action = json.loads(response.choices[0].message.content)
            tool_name = action['tool_name']
            args = action.get('args', {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            history.append(f"Invalid action response from LLM: {response.choices[0].message.content}")
            continue

        print(f"Action: {tool_name} with args: {args}")

        # If the agent decides it's finished, return the answer
        if tool_name == "finish":
            return args.get("final_answers", {"error": "Agent finished without providing an answer."})

        # 2. ACT: Execute the chosen tool
        if tool_name == "run_python_code_on_data":
            df_name = args.pop('dataframe_name')
            observation = run_python_code_on_data(dataframe=data_context[df_name], **args)
        elif tool_name == "run_duckdb_query":
            observation = run_duckdb_query(**args)
            if isinstance(observation, pd.DataFrame):
                # Save the result with a unique name
                query_result_name = f"query_result_{i+1}"
                data_context[query_result_name] = observation
                observation = f"Query successful. Result saved as DataFrame '{query_result_name}'. Columns: {observation.columns.tolist()}"
        elif tool_name == "create_scatterplot_with_regression":
            df_name = args.pop('dataframe_name')
            observation = create_scatterplot_with_regression(dataframe=data_context[df_name], **args)
        else:
            observation = f"Error: Unknown tool '{tool_name}'"
        
        print(f"Observation: {observation}")
        
        # 3. OBSERVE: Add the action and observation to history for the next loop
        history.append(f"Action: {action}, Observation: {observation}")

    return {"error": "Agent exceeded maximum number of steps."}