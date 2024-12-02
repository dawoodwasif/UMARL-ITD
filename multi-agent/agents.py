import numpy as np
import pandas as pd
import logging
import re
from typing import List, Dict
from collections import defaultdict
from scipy.stats import dirichlet
from openai import OpenAI  


# Initialize logging
logging.basicConfig(level=logging.INFO)



class UncertaintyAwareRLToolBuilderAgent:
    def __init__(self, api_key, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, max_retries=10, num_actions=3):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.tools = {}
        self.q_table = defaultdict(lambda: defaultdict(lambda: 1e-3))  # Initialize Q-values to a small positive value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_retries = max_retries
        self.num_actions = num_actions  # Number of available actions
        self.uncertainties = {"vacuity": [], "dissonance": [], "entropy": []}

    def choose_action(self, state: str) -> int:
        """Choose an action based on the state and Q-table using Dirichlet probabilities."""
        action_probabilities = self.get_action_probabilities(state)
        
        # Handle invalid probabilities
        if np.any(np.isnan(action_probabilities)):
            logging.error(f"Action probabilities contain NaN for state '{state}'. Defaulting to uniform probabilities.")
            action_probabilities = np.full(self.num_actions, 1 / self.num_actions)
        
        action = np.random.choice(range(self.num_actions), p=action_probabilities)
        logging.info(f"Chosen action with probabilities {action_probabilities}: {action}")
        return action

    def get_action_probabilities(self, state: str) -> List[float]:
        """Compute action probabilities using Dirichlet distribution."""
        action_counts = [self.q_table[state].get(a, 0) + 1e-3 for a in range(self.num_actions)]
        
        # Ensure all parameters are positive for the Dirichlet distribution
        if any(count <= 0 for count in action_counts):
            logging.warning(f"Invalid Dirichlet parameters for state '{state}': {action_counts}. Fixing to minimum value.")
            action_counts = [max(count, 1e-3) for count in action_counts]
        
        probabilities = dirichlet.rvs(action_counts, size=1).flatten()
        
        # Normalize probabilities to ensure they sum to 1
        probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.full(self.num_actions, 1 / self.num_actions)
        
        return probabilities


    def execute_with_retry(self, func, df, max_retries=3, **kwargs):
        """
        Executes a function with retry mechanism. Fixes known issues like index mismatch.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return func(df, **kwargs)
            except ValueError as ve:
                logging.error(f"Execution attempt {attempt + 1} failed: {ve}")
                if "incompatible index of inserted column" in str(ve):
                    df = df.reset_index(drop=True)  # Reset index to fix the mismatch
                    logging.info("Index mismatch detected. Resetting dataframe index and retrying.")
                else:
                    logging.error(f"Unhandled ValueError: {ve}")
                    raise ve
            except Exception as e:
                logging.error(f"Execution attempt {attempt + 1} failed: {e}")
            attempt += 1
        logging.error(f"Function execution failed after {max_retries} attempts.")
        return None

    def update_uncertainty_measures(self, action_probabilities: List[float]):
        """Update vacuity, dissonance, and entropy measures."""
        vacuity = 1 - sum(action_probabilities) / self.num_actions
        dissonance = sum(
            abs(a_i - a_j)
            for i, a_i in enumerate(action_probabilities)
            for j, a_j in enumerate(action_probabilities)
            if i != j
        ) / (2 * self.num_actions)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in action_probabilities) / np.log2(self.num_actions)

        self.uncertainties["vacuity"].append(vacuity)
        self.uncertainties["dissonance"].append(dissonance)
        self.uncertainties["entropy"].append(entropy)

        logging.info(f"Updated uncertainty measures: Vacuity={vacuity}, Dissonance={dissonance}, Entropy={entropy}")

    def generate_tool_code(self, task_name: str, requirements: str, prompt_style: str, columns: List[str], custom_prompt: str = None) -> str:
        """
        Generate code for a specific tool using GPT.
        Accepts a custom prompt if provided.
        """
        prompt = custom_prompt or f"""
        Generate a complete, well-formatted Python function named '{task_name}' to detect {requirements}.
        Ensure column names ({', '.join(columns)}) are validated, with detailed error handling.
        """

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )
                raw_text = response.choices[0].message.content.strip()
                code = self.extract_code(raw_text)
                if code and self.is_code_valid(code):
                    logging.info(f"Generated code after {attempt + 1} attempt(s)")
                    return code
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                continue

            logging.warning(f"Retry {attempt + 1}/{self.max_retries}: Syntax issue, retrying...")
        return None

    @staticmethod
    def extract_code(response_text: str) -> str:
        """Extract Python code block from GPT response."""
        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        return code_match.group(1) if code_match else response_text

    @staticmethod
    def is_code_valid(code: str) -> bool:
        """Check if the generated code is syntactically valid."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError as e:
            logging.error(f"Syntax issue: {e}")
            return False

    def evaluate_tool(self, tool_name: str):
        tool_func = self.tools.get(tool_name)
        if tool_func:
            test_data = self._get_test_data(tool_name)  # Use the defined method
            try:
                return self.execute_with_retry(tool_func, test_data)
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")
                return -5
        logging.error(f"Tool {tool_name} not found.")
        return -10

    def create_tool(self, task_name: str, requirements: str, columns: List[str]):
        """Main function to create and evaluate a tool."""
        state = task_name
        action = self.choose_action(state)

        tool_code = self.generate_tool_code(task_name, requirements, "default_prompt", columns)
        if tool_code is None:
            logging.error(f"Code generation failed for {task_name} after maximum retries.")
            self.update_q_table(state, action, -10)
            return

        try:
            # Debug: Print the generated code
            print("Generated Code:")
            print(tool_code)
            
            exec(tool_code, globals())
            func_name = task_name.replace(' ', '_')
            # Debug: Check if the function exists in globals
            print(f"Looking for function: {func_name} in globals.")
            self.tools[task_name] = globals().get(func_name)
            
            if self.tools[task_name] is None:
                logging.error(f"Function {func_name} was not created successfully.")
                self.update_q_table(state, action, -10)
                return

            reward = self.evaluate_tool(func_name)
            self.update_q_table(state, action, reward)
        except Exception as e:
            logging.error(f"Tool generation failed for {task_name}: {e}")
        self.update_q_table(state, action, -10)

    def update_q_table(self, state: str, action: int, reward: float):
        """Update Q-table based on the RL update rule, ensuring non-negative Q-values."""
        current_q = self.q_table[state][action]
        best_future_q = max(self.q_table[state].values(), default=0)
        new_q = max(
            current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q),
            1e-3  # Ensure Q-values are at least 1e-3
        )
        self.q_table[state][action] = new_q
        logging.info(f"Q-table updated: State={state}, Action={action}, Q-value={new_q}")

    def _get_test_data(self, tool_name: str):
        # Example test data based on expected input
        return pd.DataFrame({
            "user": ["WCR0044", "LRG0155"],
            "date": ["2024-01-02 05:02:50", "2024-01-02 06:33:00"],
            "pc": ["PC-9174", "PC-0450"],
            "activity": ["Logon", "Logoff"],
            "content": ["Login successful", "User logged out"]
        })


# Decomposer Agent
class DecomposerAgent:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def decompose_task(self) -> Dict[str, str]:
        log_types = ['logon', 'psychometric', 'file_access', 'email', 'device']
        subtasks = {log_type: f"{self.task_name}_{log_type}" for log_type in log_types}
        logging.info(f"Decomposed task '{self.task_name}' into subtasks: {list(subtasks.values())}")
        return subtasks


# Executor Node
class ExecutorNode:
    def __init__(self, tool_name: str, tool_func):
        self.tool_name = tool_name
        self.tool_func = tool_func

    def execute(self, df: pd.DataFrame, *args, **kwargs):
        if not callable(self.tool_func):
            logging.error(f"Tool {self.tool_name} is not callable.")
            return None
        try:
            return self.tool_func(df, *args, **kwargs)
        except Exception as e:
            logging.error(f"Error executing tool {self.tool_name}: {e}")
            return None


# Anomaly Aggregator Agent
class AnomalyAggregatorAgent:
    def aggregate(self, results: Dict[str, pd.DataFrame]):
        aggregated_results = {}
        for log_type, result in results.items():
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    aggregated_results[log_type] = {
                        "anomalies": None,
                        "reasons": f"No anomalies detected (empty DataFrame) for {log_type}"
                    }
                else:
                    aggregated_results[log_type] = {
                        "anomalies": result,
                        "reasons": f"Flagged {len(result)} suspicious entries for {log_type}"
                    }
            elif result is None:
                aggregated_results[log_type] = {
                    "anomalies": None,
                    "reasons": f"No anomalies detected or tool failed for {log_type}"
                }
            else:
                logging.warning(f"Unexpected result type for {log_type}: {type(result).__name__}")
                aggregated_results[log_type] = {
                    "anomalies": None,
                    "reasons": f"Tool returned an invalid result for {log_type}"
                }
        logging.info(f"Final Aggregated Anomalies:\n{aggregated_results}")
        return aggregated_results
