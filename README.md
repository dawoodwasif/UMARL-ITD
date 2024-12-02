# UMARL-ITD: Uncertainty-Aware Multi-Agent Reinforcement Learning for Insider Threat Detection


# UMARL-ITD: Uncertainty-Aware Multi-Agent Reinforcement Learning for Insider Threat Detection

## Overview
UMARL-ITD is an advanced multi-agent system designed for insider threat detection. Leveraging Uncertainty-Aware Reinforcement Learning (UMARL), this framework uses state-of-the-art techniques to detect suspicious activities in large-scale datasets, such as the CERT Insider Threat Dataset. The system decomposes tasks, applies RL to build tools dynamically, and aggregates results for actionable insights.

### Key Features
- **Uncertainty-Aware RL**: A novel approach to managing uncertainty in tool-building decisions using Dirichlet distributions.
- **Multi-Agent Architecture**: Includes specialized agents for task decomposition, anomaly aggregation, and RL-based tool building.
- **Scalable Pipeline**: Handles large datasets with options for subsetting and parallel execution.
- **Customizable Framework**: Designed with modularity for easy extension and experimentation.

## Repository Structure
```
├── agents.py           # Defines Decomposer, Anomaly Aggregator, Executor Node, and RL Tool Builder agents.
├── main.py             # Entry point for running the pipeline with command-line arguments.
├── pipeline.py         # Core pipeline logic for tool generation and anomaly aggregation.
├── read_dataset.py     # Script for loading datasets dynamically.
├── sample_data.py      # Sample data used for testing the pipeline.
├── schema.py           # Specifications for each log type and its schema.
├── umarl-demo.ipynb    # Interactive Jupyter Notebook showcasing the framework.
├── umarl-flow.ipynb    # Interactive Jupyter Notebook to visualize flow of agents.
├── utils.py            # Utility functions for data preparation and execution safety.
```

## Prerequisites
Ensure the following tools and libraries are installed:
- Python 3.8+
- Required Python packages:
  ```bash
  pip install pandas numpy openai tqdm scipy
  ```
- OpenAI API Key: Set your API key as an environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key"
  ```

## Usage

### 1. Running the Pipeline
Use the `main.py` script to execute the pipeline. Parameters can be customized via command-line arguments.

#### Command-line Arguments
| Argument             | Type    | Default               | Description                                        |
|----------------------|---------|-----------------------|--------------------------------------------------|
| `--mode`            | string  | (required)            | Pipeline mode: `test` or `benchmark`.            |
| `--subset_size`     | int     | `None`                | Number of rows to subset for benchmarking.       |
| `--openai_api_key`  | string  | `OPENAI_API_KEY` env  | API key for OpenAI.                              |
| `--learning_rate`   | float   | `0.1`                 | Learning rate for the RL agent.                  |
| `--discount_factor` | float   | `0.9`                 | Discount factor for the RL agent.                |
| `--exploration_rate`| float   | `0.1`                 | Exploration rate for the RL agent.               |
| `--max_retries`     | int     | `10`                  | Maximum retries for the RL agent.                |
| `--num_actions`     | int     | `3`                   | Number of actions for the RL agent.              |
| `--log_file`        | string  | `pipeline.log`        | Log file to save execution outputs.              |

#### Example Command
```bash
python main.py --mode benchmark --subset_size 100 --log_file pipeline.log
```

### 2. Interactive Demo
Run the Jupyter Notebook `umarl-demo.ipynb` to interactively explore the pipeline.
```bash
jupyter notebook umarl-demo.ipynb
```

### 3. Customizing the Pipeline
Modify `schema.py` to add new log types or update existing schema specifications.

## Dataset
- **CERT Insider Threat Dataset**: Place the dataset files (`logon.csv`, `file.csv`, etc.) in the root directory.

## Agents Overview

### 1. `DecomposerAgent`
Decomposes high-level tasks into subtasks based on log types (e.g., `logon`, `email`, etc.).

### 2. `UncertaintyAwareRLToolBuilderAgent`
Applies reinforcement learning to build tools dynamically based on schema specifications.

### 3. `ExecutorNode`
Executes generated tools safely on the datasets.

### 4. `AnomalyAggregatorAgent`
Aggregates results across subtasks and generates a summary of detected anomalies.

## Pipeline Flow
1. **Task Decomposition**: Decomposes `Detect_Suspicious_Activity` into subtasks like `logon`, `email`, etc.
2. **Tool Generation**: Uses RL to generate tools for each subtask.
3. **Tool Execution**: Applies tools to the corresponding log datasets.
4. **Anomaly Aggregation**: Aggregates results into a final report.

## Logging
All outputs, warnings, and errors are logged to the specified log file using the `--log_file` argument.

## Expected Results
The pipeline outputs aggregated anomalies detected across different log types. For example:
```
Final Aggregated Anomalies:
{
  'logon': {
    'anomalies': DataFrame containing suspicious logon activities,
    'reasons': 'Flagged 4 suspicious entries for logon'
  },
  'email': {
    'anomalies': None,
    'reasons': 'No anomalies detected'
  },
  'file_access': {
    'anomalies': DataFrame containing suspicious file deletions,
    'reasons': 'Flagged 2 suspicious entries for file_access'
  }
}
```
### Key Metrics
- **Number of Suspicious Entries Flagged**: The total number of anomalies detected in each log type.
- **Uncertainty Metrics**:
  - *Vacuity*: Measures missing or uninformative data.
  - *Dissonance*: Quantifies conflicting predictions.
  - *Entropy*: Reflects the uncertainty of predictions.
- **Execution Time**: Time taken for the pipeline to execute.

By integrating the benchmark dataset in `feature_extraction`, UMARL-ITD can be compared to other insider threat detection techniques to validate its efficiency and accuracy.

## License
This project is licensed under the MIT License.

## Contributions
Contributions are welcome! Feel free to submit pull requests or open issues for bugs and feature requests.
