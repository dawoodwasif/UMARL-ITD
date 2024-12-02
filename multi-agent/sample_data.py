import pandas as pd
import numpy as np

data_sources = {
    "logon": pd.DataFrame([
        ["WCR0044", "2024-01-02 05:02:50", "PC-9174", "Logon", "Login successful"],
        ["WCR0044", "2024-01-02 06:15:30", "PC-9174", "Logon", "Login successful"],
        ["LRG0155", "2024-01-02 06:33:00", "PC-0450", "Logoff", "User logged out"],
        ["XTR0011", "2024-01-02 07:45:00", "PC-1234", "Logon", "Login successful"],
        ["FLX0909", "2024-01-02 08:15:00", "PC-9987", "Logon", "Multiple failed login attempts"],
    ], columns=["user", "date", "pc", "activity", "content"]),

    "psychometric": pd.DataFrame([
        {"user": "WCR0044", "trait": "openness", "value": 0.8},
        {"user": "LRG0155", "trait": "conscientiousness", "value": 0.6},
        {"user": "FLX0909", "trait": "neuroticism", "value": 0.9},
    ]),

    "file_access": pd.DataFrame([
        {"user": "WCR0044", "file": "confidential.pdf", "access_type": "read", "timestamp": "2024-01-02 05:15:00"},
        {"user": "LRG0155", "file": "project_plan.docx", "access_type": "write", "timestamp": "2024-01-02 06:40:00"},
        {"user": "FLX0909", "file": "budget.xlsx", "access_type": "delete", "timestamp": "2024-01-02 07:30:00"},
    ]),

    "email": pd.DataFrame([
        {"user": "LRG0155", "email_content": "Suspicious email content", "timestamp": "2024-01-02 06:50:00"},
        {"user": "FLX0909", "email_content": "Confidential project details", "timestamp": "2024-01-02 07:15:00"},
        {"user": "XTR0011", "email_content": "Unauthorized access attempt", "timestamp": "2024-01-02 08:00:00"},
    ]),

    "device": pd.DataFrame([
        {"user": "XTR0011", "device": "USB", "action": "connect", "timestamp": "2024-01-02 07:00:00"},
        {"user": "FLX0909", "device": "External HDD", "action": "disconnect", "timestamp": "2024-01-02 08:30:00"},
        {"user": "WCR0044", "device": "Printer", "action": "connect", "timestamp": "2024-01-02 09:00:00"},
    ]),
}

# Standardize column names to lowercase
for log_type, df in data_sources.items():
    data_sources[log_type].columns = df.columns.str.lower()

# Convert relevant columns to datetime format
datetime_columns = {
    "logon": ["date"],
    "file_access": ["timestamp"],
    "email": ["timestamp"],
    "device": ["timestamp"]
}
for log_type, cols in datetime_columns.items():
    for col in cols:
        if col in data_sources[log_type].columns:
            data_sources[log_type][col] = pd.to_datetime(data_sources[log_type][col], errors='coerce')