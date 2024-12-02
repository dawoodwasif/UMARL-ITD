import argparse
import logging
import os
from datetime import datetime
from typing import Dict
import pandas as pd
from agents import (
    DecomposerAgent,
    AnomalyAggregatorAgent,
    ExecutorNode,
    UncertaintyAwareRLToolBuilderAgent
)
from utils import safe_execute_tool, validate_and_prepare_data, load_and_subset_data
from schema import tool_specs
from sample_data import data_sources

# Set up logging
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Anomaly Detection Pipeline.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "benchmark"],
        required=True,
        help="Mode to run the pipeline: 'test' or 'benchmark'."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="Number of rows to use for each dataset (for benchmarking)."
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for OpenAI. Defaults to the 'OPENAI_API_KEY' environment variable."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the RL agent."
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.9,
        help="Discount factor for the RL agent."
    )
    parser.add_argument(
        "--exploration_rate",
        type=float,
        default=0.1,
        help="Exploration rate for the RL agent."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum retries for the RL agent."
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        default=3,
        help="Number of actions for the RL agent."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="pipeline.log",
        help="Log file to store output logs."
    )
    return parser.parse_args()

# Initialize agents
def initialize_agents(task_name: str, args):
    decomposer_agent = DecomposerAgent(task_name)
    anomaly_aggregator_agent = AnomalyAggregatorAgent()
    rl_tool_builder_agent = UncertaintyAwareRLToolBuilderAgent(
        api_key=args.openai_api_key,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        max_retries=args.max_retries,
        num_actions=args.num_actions
    )
    return decomposer_agent, anomaly_aggregator_agent, rl_tool_builder_agent

# Tool code generation
def generate_tool_code(agent, task_name: str, requirements: str, log_type: str) -> str:
    spec = tool_specs[log_type]
    prompt = f"""
    Generate a robust Python function named '{task_name}'.
    Objective: {spec['objective']}
    Input Schema: The DataFrame must include these columns: {', '.join(spec['required_columns'])}.
    Example Input: {spec['example_input']}
    Expected Output: {spec['example_output']}
    Key Requirements:
    - Validate input schema with error handling for missing or invalid columns.
    - Handle edge cases, such as empty DataFrames or columns with incorrect datatypes.
    - Include detailed comments for each step.
    """
    return agent.generate_tool_code(task_name, requirements, "default_prompt", spec["required_columns"], prompt)

# Execute pipeline
def execute_pipeline(decomposer_agent, rl_tool_builder_agent, anomaly_aggregator_agent):
    subtasks = decomposer_agent.decompose_task()
    executor_nodes = {}
    results = {}

    for log_type, subtask_name in subtasks.items():
        logging.info(f"Creating tool for subtask: {subtask_name}")
        spec = tool_specs.get(log_type, {})
        if not spec:
            logging.error(f"No specification found for log type: {log_type}")
            continue

        df = data_sources.get(log_type, pd.DataFrame())
        df = validate_and_prepare_data(df, spec["required_columns"])

        tool_code = generate_tool_code(rl_tool_builder_agent, subtask_name, spec["objective"], log_type)
        if tool_code is None:
            logging.error(f"Failed to generate tool for {subtask_name}.")
            continue

        try:
            exec(tool_code, globals())
            tool_func = globals().get(subtask_name)
            if tool_func:
                executor_nodes[log_type] = ExecutorNode(subtask_name, tool_func)
            else:
                logging.error(f"Function {subtask_name} not found in globals.")
        except Exception as e:
            logging.error(f"Error loading generated tool for {subtask_name}: {e}")

    for log_type, executor_node in executor_nodes.items():
        df = data_sources.get(log_type, pd.DataFrame())
        results[log_type] = safe_execute_tool(executor_node.tool_func, df) if executor_node.tool_func else None

    return anomaly_aggregator_agent.aggregate(results)

# Execute pipeline with a subset
def execute_pipeline_with_subset(data_dict, subset_size, decomposer_agent, rl_tool_builder_agent, anomaly_aggregator_agent):
    filtered_data_dict = {k: v for k, v in data_dict.items() if k in tool_specs}
    logging.info(f"Using subset size: {subset_size}")
    subset_data = load_and_subset_data(filtered_data_dict, subset_size)

    for log_type, df in subset_data.items():
        data_sources[log_type] = validate_and_prepare_data(df, tool_specs[log_type]["required_columns"])

    return execute_pipeline(decomposer_agent, rl_tool_builder_agent, anomaly_aggregator_agent)

# Main function
def main():
    args = parse_arguments()
    setup_logging(args.log_file)

    decomposer_agent, anomaly_aggregator_agent, rl_tool_builder_agent = initialize_agents("Detect_Suspicious_Activity", args)

    if args.mode == "test":
        final_results = execute_pipeline(decomposer_agent, rl_tool_builder_agent, anomaly_aggregator_agent)
        logging.info("Final Aggregated Anomalies:")
        logging.info(final_results)

    elif args.mode == "benchmark":
        from read_dataset import data_dict
        start_time = datetime.now()
        final_results = execute_pipeline_with_subset(data_dict, args.subset_size, decomposer_agent, rl_tool_builder_agent, anomaly_aggregator_agent)
        end_time = datetime.now()
        execution_time = end_time - start_time

        logging.info(f"Pipeline execution time: {execution_time}")
        logging.info("Final Aggregated Anomalies:")
        logging.info(final_results)

if __name__ == "__main__":
    main()
