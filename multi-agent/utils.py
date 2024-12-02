import logging
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
import random

def safe_execute_tool(tool_func, df: pd.DataFrame, **kwargs):
    """
    Safely execute the tool function and ensure it returns a DataFrame.
    """
    try:
        result = tool_func(df, **kwargs)
        if not isinstance(result, pd.DataFrame):  # Ensure the result is a DataFrame
            raise ValueError(f"Tool did not return a valid DataFrame. Got {type(result).__name__} instead.")
        return result
    except Exception as e:
        logging.error(f"Unhandled exception during tool execution for {tool_func.__name__}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def validate_and_prepare_data(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Validate and prepare input data for the tool.
    Ensures all required columns exist and are in correct formats.
    """
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' is missing in input data. Filling with NaN.")
            df[col] = None  # Add missing columns with NaN values

    # Ensure datetime columns are properly formatted
    for col in required_columns:
        if "date" in col or "timestamp" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].fillna(pd.Timestamp.now())  # Correct assignment
            except Exception as e:
                logging.error(f"Error converting column '{col}' to datetime: {e}")
                df[col] = pd.Timestamp.now()  # Default to current time if conversion fails

    return df


def load_and_subset_data(data_dict, subset_size=None):
    """
    Load and optionally subset data from the provided data dictionary.
    
    Args:
        data_dict (dict): Dictionary containing DataFrames for different log types.
        subset_size (int): Number of rows to select for each DataFrame. If None, uses the entire dataset.
    
    Returns:
        dict: Dictionary containing DataFrames (subset if specified).
    """
    subset_data = {}
    for log_type, df in tqdm(data_dict.items(), desc="Loading and Subsetting Data"):
        if subset_size:
            # Ensure reproducibility for subsets
            random.seed(42)
            subset_data[log_type] = df.sample(n=min(subset_size, len(df)), random_state=42)
        else:
            subset_data[log_type] = df
    return subset_data
